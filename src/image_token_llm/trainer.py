"""
Comprehensive training script for multi-modal reasoning model.
Supports knowledge distillation, continuous learning, and graph-based reasoning.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import (
    EvaluatorConfig,
    GraphRAGConfig,
    ImageTokenizerConfig,
    SimulatorConfig,
)
from .continuous_learning import IncrementalLearner, OnlineLearningLoop
from .data_loaders import create_dataloader
from .graph_attention import GraphRAGEnhanced
from .knowledge_transfer import (
    KnowledgeDistillationLoss,
    OllamaDistillationPipeline,
    TeacherStudentAdapter,
)
from .vision_encoder import TripletEncoder


class MultiModalReasoningTrainer:
    """
    Main trainer for multi-modal image-based reasoning model.
    Integrates vision encoding, graph reasoning, knowledge distillation,
    and continuous learning.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        learning_rate: float = 1e-4,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        use_distillation: bool = True,
        teacher_model: str = "llama2",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize configs
        self.img_config = ImageTokenizerConfig(
            embedding_dim=embedding_dim,
            patch_size=16,
        )
        self.graph_config = GraphRAGConfig(
            top_k_neighbors=8,
            max_hops=3,
        )

        # Initialize models
        print("Initializing models...")
        self.triplet_encoder = TripletEncoder(self.img_config).to(device)
        self.graph_rag = GraphRAGEnhanced(
            self.graph_config,
            embedding_dim=embedding_dim,
            device=device,
        )

        # Initialize continuous learning
        self.incremental_learner = IncrementalLearner(
            triplet_encoder=self.triplet_encoder,
            graph_rag=self.graph_rag,
            learning_rate=learning_rate,
        )

        self.online_loop = OnlineLearningLoop(
            learner=self.incremental_learner,
            consolidation_frequency=100,
        )

        # Knowledge distillation setup
        self.use_distillation = use_distillation
        if use_distillation:
            print(f"Setting up knowledge distillation from {teacher_model}...")
            self.distillation_pipeline = OllamaDistillationPipeline(
                teacher_model=teacher_model,
            )
            self.distillation_loss = KnowledgeDistillationLoss()
            self.teacher_adapter = TeacherStudentAdapter(
                teacher_dim=4096,
                student_dim=embedding_dim,
            ).to(device)

        # Optimizer
        all_params = list(self.triplet_encoder.parameters())
        if use_distillation:
            all_params += list(self.teacher_adapter.parameters())

        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=learning_rate,
            weight_decay=0.01,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
        )

        # Metrics
        self.global_step = 0
        self.writer: Optional[SummaryWriter] = None

    def train(
        self,
        num_epochs: int = 10,
        batch_size: int = 16,
        dataset_name: str = "synthetic",
        max_samples: Optional[int] = None,
        log_interval: int = 10,
        save_interval: int = 100,
        enable_tensorboard: bool = True,
    ) -> None:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            dataset_name: Dataset to use
            max_samples: Max samples per epoch (for debugging)
            log_interval: Steps between logging
            save_interval: Steps between checkpoints
            enable_tensorboard: Enable TensorBoard logging
        """
        if enable_tensorboard:
            self.writer = SummaryWriter(log_dir="./runs/multimodal_training")

        print(f"Creating {dataset_name} dataloader...")
        train_loader = create_dataloader(
            dataset_name=dataset_name,
            split="train",
            batch_size=batch_size,
            max_samples=max_samples,
        )

        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            epoch_metrics = self._train_epoch(
                train_loader,
                epoch,
                log_interval,
                save_interval,
            )

            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Loss: {epoch_metrics['loss']:.4f}"
            )

            # Save checkpoint after each epoch
            self.save_checkpoint(f"epoch_{epoch+1}.pt")

        if self.writer:
            self.writer.close()

        print("Training complete!")

    def _train_epoch(
        self,
        dataloader,
        epoch: int,
        log_interval: int,
        save_interval: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.triplet_encoder.train()
        epoch_metrics = {
            "loss": 0.0,
            "recon_loss": 0.0,
            "distill_loss": 0.0,
        }
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            metrics = self._train_step(batch)

            # Accumulate metrics
            for key, value in metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value

            num_batches += 1

            # Logging
            if self.global_step % log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{metrics.get('loss', 0):.4f}",
                    "step": self.global_step,
                })

                if self.writer:
                    for key, value in metrics.items():
                        self.writer.add_scalar(
                            f"train/{key}",
                            value,
                            self.global_step,
                        )

            # Checkpointing
            if self.global_step % save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")

            self.global_step += 1

        # Average epoch metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)

        return epoch_metrics

    def _train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        what = batch["what"].to(self.device)
        action = batch["action"].to(self.device)
        result = batch["result"].to(self.device)
        triplets = batch["triplet"]

        # Forward pass through encoder
        fused_emb, components = self.triplet_encoder(what, action, result)

        # Reconstruction loss
        what_comp = components[:, 0, :]
        action_comp = components[:, 1, :]
        result_comp = components[:, 2, :]

        recon_loss = (
            nn.functional.mse_loss(what_comp, fused_emb)
            + nn.functional.mse_loss(action_comp, fused_emb)
            + nn.functional.mse_loss(result_comp, fused_emb)
        ) / 3.0

        total_loss = recon_loss
        metrics = {"loss": total_loss.item(), "recon_loss": recon_loss.item()}

        # Knowledge distillation (if enabled)
        if self.use_distillation and self.global_step % 5 == 0:
            # Periodically distill from teacher
            distill_loss = self._distillation_step(fused_emb, triplets)
            total_loss = total_loss + 0.3 * distill_loss
            metrics["distill_loss"] = distill_loss.item()

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.triplet_encoder.parameters(),
            max_norm=1.0,
        )
        self.optimizer.step()
        self.scheduler.step()

        # Update graph with learned embeddings
        for i, triplet in enumerate(triplets):
            if isinstance(triplet, tuple) and len(triplet) == 3:
                emb = fused_emb[i].detach()
                self.graph_rag.ingest([triplet], [emb])

        return metrics

    def _distillation_step(
        self,
        student_embeddings: torch.Tensor,
        triplets,
    ) -> torch.Tensor:
        """Distillation from teacher model."""
        # Generate teacher knowledge for a subset
        batch_size = min(4, len(triplets))
        sample_triplets = triplets[:batch_size]

        # Extract knowledge from teacher via Ollama
        teacher_texts = [
            f"{t[0]} {t[1]} {t[2]}" if isinstance(t, tuple) else str(t)
            for t in sample_triplets
        ]

        try:
            teacher_embs = self.distillation_pipeline.distill_embeddings(
                teacher_texts,
                batch_size=batch_size,
            )
            teacher_embs = teacher_embs.to(self.device)

            # Adapt teacher embeddings to student space
            teacher_projected = self.teacher_adapter(teacher_embs)

            # MSE loss between student and adapted teacher
            student_subset = student_embeddings[:batch_size]
            distill_loss = nn.functional.mse_loss(
                student_subset,
                teacher_projected,
            )

            return distill_loss

        except Exception as e:
            print(f"Distillation error: {e}")
            return torch.tensor(0.0, device=self.device)

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = {
            "global_step": self.global_step,
            "triplet_encoder": self.triplet_encoder.state_dict(),
            "graph_reasoning_net": self.graph_rag.reasoning_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        if self.use_distillation:
            checkpoint["teacher_adapter"] = self.teacher_adapter.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.triplet_encoder.load_state_dict(checkpoint["triplet_encoder"])
        self.graph_rag.reasoning_net.load_state_dict(
            checkpoint["graph_reasoning_net"]
        )
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_step = checkpoint["global_step"]

        if self.use_distillation and "teacher_adapter" in checkpoint:
            self.teacher_adapter.load_state_dict(checkpoint["teacher_adapter"])

        print(f"Loaded checkpoint: {checkpoint_path}")


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train multi-modal model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--dataset", type=str, default="synthetic")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--use-distillation", action="store_true")
    parser.add_argument("--teacher-model", type=str, default="llama2")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize trainer
    trainer = MultiModalReasoningTrainer(
        embedding_dim=args.embedding_dim,
        learning_rate=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        use_distillation=args.use_distillation,
        teacher_model=args.teacher_model,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
