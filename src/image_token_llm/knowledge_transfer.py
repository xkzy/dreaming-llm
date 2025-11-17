"""Knowledge transfer from Ollama open-source models."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ollama
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class OllamaDistillationPipeline:
    """
    Extract knowledge from Ollama models and transfer to our architecture.
    Supports distillation from llama2, mistral, phi, and other Ollama models.
    """

    def __init__(
        self,
        teacher_model: str = "llama2",
        temperature: float = 1.0,
        api_base: str = "http://localhost:11434",
    ) -> None:
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.api_base = api_base
        self._check_model_availability()

    def _check_model_availability(self) -> None:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.api_base}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available = [m["name"] for m in models]
                if self.teacher_model not in available:
                    print(
                        f"Warning: {self.teacher_model} not found. "
                        f"Available: {available}. "
                        f"Run: ollama pull {self.teacher_model}"
                    )
        except requests.RequestException:
            print("Warning: Ollama service not reachable at", self.api_base)

    def generate_reasoning_traces(
        self,
        prompts: List[str],
        num_samples: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Generate reasoning traces from teacher model.
        
        Args:
            prompts: Input prompts for reasoning tasks
            num_samples: Number of reasoning traces per prompt
        
        Returns:
            List of reasoning traces with logits/embeddings
        """
        traces = []

        for prompt in tqdm(prompts, desc="Generating traces"):
            for _ in range(num_samples):
                try:
                    response = ollama.generate(
                        model=self.teacher_model,
                        prompt=prompt,
                        options={
                            "temperature": self.temperature,
                            "num_predict": 256,
                        },
                    )

                    trace = {
                        "prompt": prompt,
                        "response": response.get("response", ""),
                        "model": self.teacher_model,
                    }
                    traces.append(trace)

                except Exception as e:
                    print(f"Error generating trace: {e}")
                    continue

        return traces

    def extract_knowledge_triplets(
        self,
        text: str,
    ) -> List[Tuple[str, str, str]]:
        """
        Use teacher model to extract (subject, relation, object) triplets.
        
        Args:
            text: Input text to extract knowledge from
        
        Returns:
            List of (subject, relation, object) triplets
        """
        extraction_prompt = f"""
Extract knowledge triplets from the following text in the format:
(subject, relation, object)

Text: {text}

Triplets:
"""
        try:
            response = ollama.generate(
                model=self.teacher_model,
                prompt=extraction_prompt,
                options={"temperature": 0.3},
            )

            # Parse response for triplets
            triplets = []
            response_text = response.get("response", "")
            for line in response_text.split("\n"):
                line = line.strip()
                if line.startswith("(") and line.endswith(")"):
                    parts = line[1:-1].split(",")
                    if len(parts) == 3:
                        triplets.append(tuple(p.strip() for p in parts))

            return triplets

        except Exception as e:
            print(f"Error extracting triplets: {e}")
            return []

    def distill_embeddings(
        self,
        texts: List[str],
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Extract embeddings from teacher model for distillation.
        
        Args:
            texts: Input texts
            batch_size: Processing batch size
        
        Returns:
            Embeddings tensor [N, embedding_dim]
        """
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Distilling"):
            batch = texts[i : i + batch_size]
            for text in batch:
                try:
                    response = ollama.embeddings(
                        model=self.teacher_model,
                        prompt=text,
                    )
                    emb = response.get("embedding", [])
                    if emb:
                        embeddings.append(torch.tensor(emb))
                    else:
                        # Fallback random embedding
                        embeddings.append(torch.randn(4096) * 0.01)

                except Exception as e:
                    print(f"Error getting embedding: {e}")
                    embeddings.append(torch.randn(4096) * 0.01)

        return torch.stack(embeddings)


class KnowledgeDistillationLoss(nn.Module):
    """
    Loss function for distilling knowledge from teacher to student model.
    Combines soft targets, hard targets, and feature matching.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        beta: float = 0.3,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for soft target loss
        self.beta = beta  # Weight for feature matching
        self.gamma = 1.0 - alpha - beta  # Weight for hard target loss

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        hard_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.
        
        Args:
            student_logits: [B, num_classes]
            teacher_logits: [B, num_classes]
            student_features: [B, feature_dim]
            teacher_features: [B, feature_dim]
            hard_targets: [B] optional ground truth labels
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Soft target distillation loss (KL divergence)
        student_soft = F.log_softmax(
            student_logits / self.temperature,
            dim=-1,
        )
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction="batchmean",
        ) * (self.temperature ** 2)

        # Feature matching loss (MSE)
        feature_loss = F.mse_loss(student_features, teacher_features)

        # Hard target loss (cross entropy)
        hard_loss = torch.tensor(0.0, device=student_logits.device)
        if hard_targets is not None:
            hard_loss = F.cross_entropy(student_logits, hard_targets)

        # Combine losses
        total_loss = (
            self.alpha * soft_loss
            + self.beta * feature_loss
            + self.gamma * hard_loss
        )

        loss_dict = {
            "soft_loss": soft_loss.item(),
            "feature_loss": feature_loss.item(),
            "hard_loss": hard_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict


class TeacherStudentAdapter(nn.Module):
    """
    Adapter to align teacher and student feature spaces.
    """

    def __init__(
        self,
        teacher_dim: int = 4096,
        student_dim: int = 512,
    ) -> None:
        super().__init__()
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim

        # Project teacher features to student dimension
        self.teacher_projector = nn.Sequential(
            nn.Linear(teacher_dim, student_dim * 2),
            nn.LayerNorm(student_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(student_dim * 2, student_dim),
            nn.LayerNorm(student_dim),
        )

    def forward(self, teacher_features: torch.Tensor) -> torch.Tensor:
        """Project teacher features to student space."""
        return self.teacher_projector(teacher_features)


def create_distillation_dataset(
    prompts: List[str],
    teacher_model: str = "llama2",
    save_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Create a dataset of reasoning traces from Ollama teacher model.
    
    Args:
        prompts: List of reasoning prompts
        teacher_model: Ollama model to use
        save_path: Optional path to save dataset
    
    Returns:
        Dataset of reasoning traces
    """
    pipeline = OllamaDistillationPipeline(teacher_model=teacher_model)
    dataset = pipeline.generate_reasoning_traces(prompts, num_samples=3)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved {len(dataset)} traces to {save_path}")

    return dataset
