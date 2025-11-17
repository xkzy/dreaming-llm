"""Continuous learning system with experience replay and incremental updates."""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .graph_attention import GraphRAGEnhanced
from .vision_encoder import TripletEncoder


class ExperienceBuffer:
    """
    Replay buffer for storing and sampling past experiences.
    Supports prioritized sampling based on importance.
    """

    def __init__(self, max_size: int = 10000, priority_alpha: float = 0.6) -> None:
        self.max_size = max_size
        self.priority_alpha = priority_alpha
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=max_size)
        self.priorities: Deque[float] = deque(maxlen=max_size)

    def add(
        self,
        experience: Dict[str, Any],
        priority: float = 1.0,
    ) -> None:
        """Add an experience to the buffer."""
        self.buffer.append(experience)
        self.priorities.append(priority ** self.priority_alpha)

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor, List[int]]:
        """
        Sample a batch of experiences with prioritized sampling.
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent
        
        Returns:
            experiences: List of sampled experiences
            weights: Importance sampling weights
            indices: Indices of sampled experiences
        """
        if len(self.buffer) == 0:
            return [], torch.tensor([]), []

        priorities = torch.tensor(list(self.priorities), dtype=torch.float32)
        probs = priorities / priorities.sum()

        batch_size = min(batch_size, len(self.buffer))
        indices = torch.multinomial(probs, batch_size, replacement=False)

        experiences = [self.buffer[i] for i in indices]

        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        return experiences, weights, indices.tolist()

    def update_priorities(
        self,
        indices: List[int],
        priorities: List[float],
    ) -> None:
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority ** self.priority_alpha

    def __len__(self) -> int:
        return len(self.buffer)


class IncrementalLearner(nn.Module):
    """
    Incremental learning system that continuously adapts to new knowledge.
    Uses elastic weight consolidation (EWC) to prevent catastrophic forgetting.
    """

    def __init__(
        self,
        triplet_encoder: TripletEncoder,
        graph_rag: GraphRAGEnhanced,
        learning_rate: float = 1e-4,
        ewc_lambda: float = 0.4,
        buffer_size: int = 10000,
    ) -> None:
        super().__init__()
        self.triplet_encoder = triplet_encoder
        self.graph_rag = graph_rag
        self.learning_rate = learning_rate
        self.ewc_lambda = ewc_lambda

        self.experience_buffer = ExperienceBuffer(max_size=buffer_size)
        self.optimizer = optim.AdamW(
            self.triplet_encoder.parameters(),
            lr=learning_rate,
        )

        # EWC: Store important weights
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.optimal_weights: Dict[str, torch.Tensor] = {}

    def observe(
        self,
        what_images: torch.Tensor,
        action_images: torch.Tensor,
        result_images: torch.Tensor,
        triplet_label: Tuple[str, str, str],
        priority: float = 1.0,
    ) -> None:
        """
        Observe a new experience and add to buffer.
        
        Args:
            what_images: [B, C, H, W]
            action_images: [B, C, H, W]
            result_images: [B, C, H, W]
            triplet_label: (subject, relation, object) label
            priority: Experience priority for replay
        """
        experience = {
            "what": what_images.cpu(),
            "action": action_images.cpu(),
            "result": result_images.cpu(),
            "label": triplet_label,
            "timestamp": torch.tensor(len(self.experience_buffer)),
        }
        self.experience_buffer.add(experience, priority=priority)

    def learn_from_experience(
        self,
        batch_size: int = 16,
        num_steps: int = 10,
    ) -> Dict[str, float]:
        """
        Learn from past experiences via replay.
        
        Args:
            batch_size: Batch size for replay
            num_steps: Number of gradient steps
        
        Returns:
            Training metrics
        """
        if len(self.experience_buffer) < batch_size:
            return {"loss": 0.0}

        metrics = {"loss": 0.0, "ewc_loss": 0.0}

        for step in range(num_steps):
            # Sample from buffer
            experiences, weights, indices = self.experience_buffer.sample(
                batch_size, beta=0.4 + step * 0.05
            )

            if not experiences:
                break

            # Prepare batch
            what_batch = torch.stack([exp["what"] for exp in experiences])
            action_batch = torch.stack([exp["action"] for exp in experiences])
            result_batch = torch.stack([exp["result"] for exp in experiences])

            what_batch = what_batch.to(self.triplet_encoder.vision_encoder.projector[0].weight.device)
            action_batch = action_batch.to(self.triplet_encoder.vision_encoder.projector[0].weight.device)
            result_batch = result_batch.to(self.triplet_encoder.vision_encoder.projector[0].weight.device)

            # Forward pass
            fused_emb, components = self.triplet_encoder(
                what_batch, action_batch, result_batch
            )

            # Compute reconstruction loss
            what_recon = components[:, 0, :]
            action_recon = components[:, 1, :]
            result_recon = components[:, 2, :]

            recon_loss = (
                nn.functional.mse_loss(what_recon, fused_emb)
                + nn.functional.mse_loss(action_recon, fused_emb)
                + nn.functional.mse_loss(result_recon, fused_emb)
            ) / 3.0

            # EWC regularization
            ewc_loss = self._compute_ewc_loss()

            # Total loss
            total_loss = recon_loss + self.ewc_lambda * ewc_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.triplet_encoder.parameters(), max_norm=1.0
            )
            self.optimizer.step()

            # Update priorities based on loss
            priorities = [total_loss.item()] * len(indices)
            self.experience_buffer.update_priorities(indices, priorities)

            metrics["loss"] += recon_loss.item()
            metrics["ewc_loss"] += ewc_loss.item()

        # Average metrics
        for key in metrics:
            metrics[key] /= max(num_steps, 1)

        return metrics

    def _compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss to prevent forgetting."""
        if not self.fisher_information:
            return torch.tensor(0.0)

        ewc_loss = torch.tensor(0.0)
        device = next(self.triplet_encoder.parameters()).device

        for name, param in self.triplet_encoder.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name].to(device)
                optimal = self.optimal_weights[name].to(device)
                ewc_loss += (fisher * (param - optimal) ** 2).sum()

        return ewc_loss

    def consolidate_knowledge(self) -> None:
        """
        Consolidate current knowledge by computing Fisher information.
        Should be called after learning a task before moving to new data.
        """
        self.triplet_encoder.eval()
        self.fisher_information.clear()
        self.optimal_weights.clear()

        # Initialize Fisher information matrices
        for name, param in self.triplet_encoder.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param.data)
            self.optimal_weights[name] = param.data.clone()

        # Sample from buffer to estimate Fisher
        if len(self.experience_buffer) == 0:
            return

        batch_size = min(32, len(self.experience_buffer))
        experiences, _, _ = self.experience_buffer.sample(batch_size, beta=0.0)

        for exp in experiences:
            what = exp["what"].unsqueeze(0).to(self.optimal_weights[list(self.optimal_weights.keys())[0]].device)
            action = exp["action"].unsqueeze(0).to(self.optimal_weights[list(self.optimal_weights.keys())[0]].device)
            result = exp["result"].unsqueeze(0).to(self.optimal_weights[list(self.optimal_weights.keys())[0]].device)

            self.optimizer.zero_grad()
            fused_emb, _ = self.triplet_encoder(what, action, result)
            loss = fused_emb.norm()
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.triplet_encoder.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += param.grad.data ** 2

        # Normalize Fisher information
        n_samples = len(experiences)
        for name in self.fisher_information:
            self.fisher_information[name] /= n_samples

        self.triplet_encoder.train()

    def update_graph_knowledge(
        self,
        triplet: Tuple[str, str, str],
        embedding: torch.Tensor,
    ) -> None:
        """
        Update graph with new knowledge triplet.
        
        Args:
            triplet: (subject, relation, object)
            embedding: Associated embedding
        """
        subject, relation, obj = triplet
        self.graph_rag.ingest([triplet], [embedding])
        self.graph_rag.update_embeddings(subject, embedding)
        self.graph_rag.update_embeddings(obj, embedding)


class OnlineLearningLoop:
    """
    Main loop for continuous online learning from streaming data.
    """

    def __init__(
        self,
        learner: IncrementalLearner,
        consolidation_frequency: int = 100,
    ) -> None:
        self.learner = learner
        self.consolidation_frequency = consolidation_frequency
        self.step_count = 0

    def step(
        self,
        what_images: torch.Tensor,
        action_images: torch.Tensor,
        result_images: torch.Tensor,
        triplet_label: Tuple[str, str, str],
    ) -> Dict[str, float]:
        """
        Single step of online learning.
        
        Returns:
            Metrics from learning step
        """
        # Observe new experience
        self.learner.observe(
            what_images, action_images, result_images, triplet_label
        )

        # Learn from replay
        metrics = self.learner.learn_from_experience(batch_size=16, num_steps=1)

        # Periodic knowledge consolidation
        self.step_count += 1
        if self.step_count % self.consolidation_frequency == 0:
            self.learner.consolidate_knowledge()
            metrics["consolidated"] = 1.0

        # Update graph knowledge
        with torch.no_grad():
            fused_emb, _ = self.learner.triplet_encoder(
                what_images, action_images, result_images
            )
            self.learner.update_graph_knowledge(triplet_label, fused_emb.squeeze(0))

        return metrics
