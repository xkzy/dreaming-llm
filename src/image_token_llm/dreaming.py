"""Dreaming-based reasoning: Input tokenization, dream generation, and output decoding."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_token_llm.config import DreamingConfig, ImageTokenizerConfig  # type: ignore[import]


class InputTokenizer(nn.Module):
    """
    Converts any input (text, images) into sequences of image triplets (what, action, result).
    
    Text inputs: Uses a learned text→image projector to create synthetic image embeddings.
    Image inputs: Decomposes into (what, action, result) triplets via scene understanding.
    """
    
    def __init__(self, config: DreamingConfig, embedding_dim: int = 512):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        
        # Text-to-image projection: converts text embeddings → image embedding space
        self.text_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim * 3),  # 3x for (what, action, result)
        )
        
        # Image decomposition: splits images into triplet components
        self.image_decomposer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 3),  # 3x for (what, action, result)
        )
        
        # Learned role embeddings for (what, action, result)
        self.role_embeddings = nn.Parameter(torch.randn(3, embedding_dim))
        
    def forward(
        self,
        text_embedding: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert inputs to image triplets.
        
        Args:
            text_embedding: (B, embedding_dim) - text prompt embeddings
            image_embeddings: (B, N, embedding_dim) - image feature embeddings
            
        Returns:
            Tuple of (what, action, result) tensors, each (B, embedding_dim)
        """
        if text_embedding is not None:
            # Text → image triplets
            projected = self.text_projection(text_embedding)  # (B, embedding_dim * 3)
            B = text_embedding.shape[0]
            triplet = projected.view(B, 3, self.embedding_dim)  # (B, 3, embedding_dim)
            
        elif image_embeddings is not None:
            # Images → image triplets
            B, N, D = image_embeddings.shape
            # Average pool multiple images
            pooled = image_embeddings.mean(dim=1)  # (B, embedding_dim)
            decomposed = self.image_decomposer(pooled)  # (B, embedding_dim * 3)
            triplet = decomposed.view(B, 3, self.embedding_dim)  # (B, 3, embedding_dim)
            
        else:
            raise ValueError("Must provide either text_embedding or image_embeddings")
        
        # Add role embeddings
        triplet = triplet + self.role_embeddings.unsqueeze(0)  # (B, 3, embedding_dim)
        
        # Split into (what, action, result)
        what = triplet[:, 0, :]  # (B, embedding_dim)
        action = triplet[:, 1, :]  # (B, embedding_dim)
        result = triplet[:, 2, :]  # (B, embedding_dim)
        
        return what, action, result


class DreamSequence(nn.Module):
    """
    Generates a single dream sequence: a chain of image triplets representing a reasoning path.
    
    Each step transforms (what_t, action_t, result_t) → (what_{t+1}, action_{t+1}, result_{t+1})
    """
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Recurrent dream generator: predicts next triplet from current state
        self.dream_gru = nn.GRU(
            input_size=embedding_dim * 3,  # (what, action, result) concatenated
            hidden_size=embedding_dim * 2,
            num_layers=2,
            batch_first=True,
        )
        
        # Project GRU hidden state → next triplet
        self.triplet_projector = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 4, embedding_dim * 3),
        )
        
    def forward(
        self,
        initial_triplet: torch.Tensor,
        num_steps: int = 5,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Generate a dream sequence starting from initial triplet.
        
        Args:
            initial_triplet: (B, 3, embedding_dim) - (what, action, result)
            num_steps: Number of reasoning steps to dream
            
        Returns:
            List of (what, action, result) tuples, length=num_steps
        """
        B = initial_triplet.shape[0]
        device = initial_triplet.device
        
        # Initialize hidden state
        hidden = torch.zeros(2, B, self.embedding_dim * 2, device=device)
        
        # Flatten triplet for GRU input
        current = initial_triplet.view(B, 1, self.embedding_dim * 3)  # (B, 1, D*3)
        
        dream_sequence = []
        
        for _ in range(num_steps):
            # Generate next state
            output, hidden = self.dream_gru(current, hidden)  # output: (B, 1, D*2)
            
            # Project to next triplet
            next_triplet_flat = self.triplet_projector(output.squeeze(1))  # (B, D*3)
            next_triplet = next_triplet_flat.view(B, 3, self.embedding_dim)  # (B, 3, D)
            
            # Extract components
            what = next_triplet[:, 0, :]
            action = next_triplet[:, 1, :]
            result = next_triplet[:, 2, :]
            
            dream_sequence.append((what, action, result))
            
            # Update current for next iteration
            current = next_triplet_flat.unsqueeze(1)  # (B, 1, D*3)
        
        return dream_sequence


class MoEDreamGating(nn.Module):
    """Gating network for MoE dream expert selection."""
    
    def __init__(self, embedding_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_experts)
        )
    
    def forward(self, triplet: torch.Tensor) -> torch.Tensor:
        """
        Compute expert weights.
        
        Args:
            triplet: (B, 3, embedding_dim) - initial triplet
            
        Returns:
            weights: (B, num_experts) - softmax expert weights
        """
        B = triplet.shape[0]
        flat = triplet.view(B, -1)  # (B, 3*D)
        logits = self.gate(flat)  # (B, num_experts)
        return torch.softmax(logits, dim=-1)


class DreamGenerator(nn.Module):
    """
    Generates multiple parallel dream sequences with MoE experts.
    
    Each expert specializes in different reasoning:
    - Expert 0: Spatial reasoning (physical relationships)
    - Expert 1: Temporal reasoning (time-based sequences)
    - Expert 2: Causal reasoning (cause-effect chains)
    - Expert 3: Abstract reasoning (conceptual patterns)
    """
    
    def __init__(
        self,
        config: DreamingConfig,
        embedding_dim: int = 512,
        num_experts: int = 4
    ):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        self.num_dreams = config.num_dream_sequences
        self.dream_length = config.dream_length
        self.num_experts = num_experts
        
        # MoE: Multiple expert dream generators
        self.dream_experts = nn.ModuleList([
            DreamSequence(embedding_dim) for _ in range(num_experts)
        ])
        
        # Gating network selects which experts to use
        self.expert_gate = MoEDreamGating(embedding_dim, num_experts)
        
        # Dream diversification offsets
        self.dream_offsets = nn.Parameter(
            torch.randn(self.num_dreams, 3, embedding_dim) * 0.1
        )
        
    def forward(
        self,
        initial_what: torch.Tensor,
        initial_action: torch.Tensor,
        initial_result: torch.Tensor,
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Generate multiple parallel dream sequences using MoE experts.
        
        Args:
            initial_what, initial_action, initial_result:
                Each (B, embedding_dim)
            
        Returns:
            List of dream sequences:
            [num_dreams][dream_length][(B, D), (B, D), (B, D)]
        """
        B = initial_what.shape[0]
        
        # Stack initial triplet
        initial_triplet = torch.stack(
            [initial_what, initial_action, initial_result], dim=1
        )  # (B, 3, embedding_dim)
        
        # Compute expert weights via gating network
        expert_weights = self.expert_gate(initial_triplet)  # (B, num_experts)
        
        all_dreams = []
        
        for dream_idx in range(self.num_dreams):
            # Add learned offset for diversity
            offset = self.dream_offsets[dream_idx].unsqueeze(0)
            offset = offset.expand(B, -1, -1)
            diversified_start = initial_triplet + offset
            
            # MoE: Generate dreams from all experts, weight by gating
            expert_dreams = []
            for expert_idx in range(self.num_experts):
                expert_dream = self.dream_experts[expert_idx](
                    diversified_start,
                    num_steps=self.dream_length
                )
                expert_dreams.append(expert_dream)
            
            # Fuse expert outputs using gating weights (full MoE blending)
            # expert_dreams: List[num_experts][dream_length][(B, D), (B, D), (B, D)]
            # expert_weights: (B, num_experts)
            # We blend each step across experts using weights for each batch element
            blended_dream_seq = []
            for step in range(self.dream_length):
                # For each expert, get (what, action, result) at this step
                step_expert_outputs = [
                    expert_dreams[expert_idx][step]
                    for expert_idx in range(self.num_experts)
                ]
                # Each is (B, D), so stack: (num_experts, B, D)
                what_stack = torch.stack(
                    [out[0] for out in step_expert_outputs], dim=0
                )  # (num_experts, B, D)
                action_stack = torch.stack(
                    [out[1] for out in step_expert_outputs], dim=0
                )
                result_stack = torch.stack(
                    [out[2] for out in step_expert_outputs], dim=0
                )
                # Blend using expert_weights:
                # (B, num_experts) -> (B, 1, num_experts)
                weights = expert_weights.t().unsqueeze(-1)
                # (num_experts, B, 1)
                # Weighted sum over experts for each batch
                # (num_experts, B, D) * (num_experts, B, 1)
                # -> (num_experts, B, D)
                what_blend = (what_stack * weights).sum(dim=0)  # (B, D)
                action_blend = (action_stack * weights).sum(dim=0)
                result_blend = (result_stack * weights).sum(dim=0)
                blended_dream_seq.append(
                    (what_blend, action_blend, result_blend)
                )
            all_dreams.append(blended_dream_seq)
        
        return all_dreams


class OutputDecoder(nn.Module):
    """
    Decodes reasoning graph into output (text or images).
    
    Supports:
    - Text mode: Generate text tokens from reasoning
    - Image mode: Generate image embeddings/triplets
    - Mixed mode: Both text and images
    """
    
    def __init__(
        self,
        config: DreamingConfig,
        embedding_dim: int = 512,
        vocab_size: int = 4096
    ):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # Text decoder: reasoning embeddings → text tokens
        self.text_decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, vocab_size),
        )
        
        # Image decoder: reasoning embeddings → image triplet
        self.image_decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(
                embedding_dim * 2, embedding_dim * 3
            ),  # (what, action, result)
        )
        
    def forward(
        self,
        reasoning_embedding: torch.Tensor,
        mode: str = "text",
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dict
    ]:
        """
        Decode reasoning into output.
        
        Args:
            reasoning_embedding: (B, embedding_dim) - aggregated reasoning
            mode: "text", "image", or "both"
            
        Returns:
            If mode="text": (B, vocab_size) logits
            If mode="image":
                Tuple of (what, action, result), each (B, embedding_dim)
            If mode="both": Dict with both outputs
        """
        if mode == "text":
            return self.text_decoder(reasoning_embedding)
        
        elif mode == "image":
            image_output = self.image_decoder(reasoning_embedding)  # (B, D*3)
            B = reasoning_embedding.shape[0]
            triplet = image_output.view(B, 3, self.embedding_dim)
            
            what = triplet[:, 0, :]
            action = triplet[:, 1, :]
            result = triplet[:, 2, :]
            
            return what, action, result
        
        elif mode == "both":
            text_output = self.text_decoder(reasoning_embedding)
            image_output = self.image_decoder(reasoning_embedding)
            B = reasoning_embedding.shape[0]
            triplet = image_output.view(B, 3, self.embedding_dim)
            
            return {
                "text": text_output,
                "image": (triplet[:, 0, :], triplet[:, 1, :], triplet[:, 2, :])
            }
        
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Use 'text', 'image', or 'both'"
            )
    
    def generate_sequence(
        self,
        reasoning_embedding: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate text sequence autoregressively.
        
        Args:
            reasoning_embedding: (B, embedding_dim)
            max_length: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated token IDs (B, max_length)
        """
        B = reasoning_embedding.shape[0]
        device = reasoning_embedding.device
        
        # Start with BOS token (assuming ID 1)
        generated = torch.ones(B, 1, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Get logits for next token
            logits = self.text_decoder(reasoning_embedding)  # (B, vocab_size)
            
            # Sample next token
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences generated EOS (assuming ID 2)
            if (next_token == 2).all():
                break
        
        return generated
