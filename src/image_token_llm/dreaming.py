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
    Improvements: CLIP-style contrastive alignment, projection bottleneck, noise robustness.
    """
    
    def __init__(self, config: DreamingConfig, embedding_dim: int = 512):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        
        # Shared projection bottleneck for semantic alignment (512 -> 256 -> 512)
        bottleneck_dim = embedding_dim // 2
        self.text_bottleneck = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )
        self.image_bottleneck = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )
        
        # Text-to-image projection: converts text embeddings → image embedding space
        self.text_projection = nn.Sequential(
            nn.Linear(bottleneck_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim * 3),  # 3x for (what, action, result)
        )
        
        # Image decomposition: splits images into triplet components
        self.image_decomposer = nn.Sequential(
            nn.Linear(bottleneck_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 3),  # 3x for (what, action, result)
        )
        
        # Learned role embeddings for (what, action, result)
        self.role_embeddings = nn.Parameter(torch.randn(3, embedding_dim))
        
        # Noise robustness layers
        self.embedding_dropout = nn.Dropout(0.15)
        self.token_mask_prob = 0.1  # Random token masking probability
        
        # Contrastive alignment temperature (learnable)
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward(
        self,
        text_embedding: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert inputs to image triplets.
        
        Args:
            text_embedding: (B, embedding_dim) - text prompt embeddings
            image_embeddings: (B, N, embedding_dim) - image features
            
        Returns:
            Tuple of (what, action, result) tensors, each (B, embedding_dim)
        """
        if text_embedding is not None:
            # Apply noise augmentation during training
            if self.training:
                text_embedding = self.embedding_dropout(text_embedding)
                # Random token masking
                mask = torch.rand(text_embedding.shape[0], 1) > self.token_mask_prob
                mask = mask.to(text_embedding.device)
                text_embedding = text_embedding * mask
            
            # Text → bottleneck → triplets
            bottleneck = self.text_bottleneck(text_embedding)
            projected = self.text_projection(bottleneck)
            B = text_embedding.shape[0]
            triplet = projected.view(B, 3, self.embedding_dim)
            
        elif image_embeddings is not None:
            # Images → image triplets
            B, N, D = image_embeddings.shape
            # Average pool multiple images
            pooled = image_embeddings.mean(dim=1)  # (B, embedding_dim)
            
            # Apply noise augmentation during training
            if self.training:
                pooled = self.embedding_dropout(pooled)
            
            # Image → bottleneck → triplets
            bottleneck = self.image_bottleneck(pooled)
            decomposed = self.image_decomposer(bottleneck)
            triplet = decomposed.view(B, 3, self.embedding_dim)
            
        else:
            raise ValueError(
                "Must provide either text_embedding or image_embeddings"
            )
        
        # Add role embeddings
        triplet = triplet + self.role_embeddings.unsqueeze(0)
        
        # Split into (what, action, result)
        what = triplet[:, 0, :]  # (B, embedding_dim)
        action = triplet[:, 1, :]  # (B, embedding_dim)
        result = triplet[:, 2, :]  # (B, embedding_dim)
        
        return what, action, result
    
    def contrastive_loss(
        self,
        text_embedding: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CLIP-style contrastive alignment loss.
        
        Args:
            text_embedding: (B, embedding_dim)
            image_embeddings: (B, N, embedding_dim)
            
        Returns:
            Contrastive loss scalar
        """
        B = text_embedding.shape[0]
        
        # Project to shared bottleneck space
        text_bottleneck = self.text_bottleneck(text_embedding)  # (B, D/2)
        image_pooled = image_embeddings.mean(dim=1)  # (B, D)
        image_bottleneck = self.image_bottleneck(image_pooled)  # (B, D/2)
        
        # Normalize embeddings
        text_features = F.normalize(text_bottleneck, dim=-1)
        image_features = F.normalize(image_bottleneck, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(text_features, image_features.t()) / self.temperature
        
        # Symmetric cross-entropy loss
        labels = torch.arange(B, device=logits.device)
        loss_text_to_image = F.cross_entropy(logits, labels)
        loss_image_to_text = F.cross_entropy(logits.t(), labels)
        
        return (loss_text_to_image + loss_image_to_text) / 2.0


class TransformerExpertBlock(nn.Module):
    """
    Single Transformer encoder block for expert dream generation.
    Uses pre-norm architecture for stability.
    """
    
    def __init__(self, embedding_dim: int, num_heads: int = 8, ff_dim: int = 2048):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Pre-LayerNorm multi-head attention
        self.attn_norm = nn.LayerNorm(embedding_dim)
        self.self_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=0.1, batch_first=True
        )
        
        # Pre-LayerNorm feedforward
        self.ff_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(0.1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, embedding_dim)
        Returns:
            x: (B, seq_len, embedding_dim)
        """
        # Pre-norm self-attention with residual
        attn_out, _ = self.self_attn(
            self.attn_norm(x), self.attn_norm(x), self.attn_norm(x)
        )
        x = x + attn_out
        
        # Pre-norm feedforward with residual
        x = x + self.feedforward(self.ff_norm(x))
        
        return x


class DreamSequence(nn.Module):
    """
    Generates a single dream sequence using Transformer encoder blocks.
    
    Each step transforms (what_t, action_t, result_t) →
    (what_{t+1}, action_{t+1}, result_{t+1})
    Improved: Uses Transformer blocks instead of GRU for better capacity.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Positional encoding for sequence steps
        self.pos_embedding = nn.Parameter(torch.randn(1, 10, embedding_dim))
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerExpertBlock(
                embedding_dim, num_heads, ff_dim=embedding_dim * 4
            )
            for _ in range(num_layers)
        ])
        
        # Project triplet → next triplet
        self.triplet_projector = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim * 3),
        )
        
    def forward(
        self,
        initial_triplet: torch.Tensor,
        num_steps: int = 5,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Generate a dream sequence starting from initial triplet.
        
        Args:
            initial_triplet: (B, 3, embedding_dim)
            num_steps: Number of reasoning steps to dream
            
        Returns:
            List of (what, action, result) tuples, length=num_steps
        """
        B = initial_triplet.shape[0]
        device = initial_triplet.device
        
        # Average triplet components to get initial embedding
        # (B, 3, D) -> (B, D)
        current_emb = initial_triplet.mean(dim=1)
        
        # Build sequence by autoregressively generating steps
        sequence = [current_emb]
        
        for step in range(num_steps):
            # Stack current sequence: (B, seq_len, D)
            seq_tensor = torch.stack(sequence, dim=1)
            
            # Add positional embeddings
            seq_len = seq_tensor.shape[1]
            pos_emb = self.pos_embedding[:, :seq_len, :]
            seq_tensor = seq_tensor + pos_emb
            
            # Apply Transformer blocks
            for block in self.transformer_blocks:
                seq_tensor = block(seq_tensor)
            
            # Take last position and project to next triplet
            last_hidden = seq_tensor[:, -1, :]
            next_emb = self.triplet_projector(last_hidden)  # (B, D*3)
            next_emb = next_emb.view(B, 3, self.embedding_dim)
            next_emb = next_emb.mean(dim=1)  # (B, D)
            
            sequence.append(next_emb)
        
        # Convert sequence to triplets
        dream_sequence = []
        for step_emb in sequence[1:]:  # Skip initial
            # Project embedding to triplet components
            triplet_flat = self.triplet_projector(step_emb)  # (B, D*3)
            triplet = triplet_flat.view(B, 3, self.embedding_dim)
            
            what = triplet[:, 0, :]
            action = triplet[:, 1, :]
            result = triplet[:, 2, :]
            
            dream_sequence.append((what, action, result))
        
        return dream_sequence


class MoEDreamGating(nn.Module):
    """
    Noisy Top-K gating network for MoE dream expert selection.
    Implements Switch Transformer style routing with load balancing.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std
        
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, num_experts)
        )
        
        # Noise projection for tunable noise
        self.noise_gate = nn.Linear(embedding_dim * 3, num_experts)
        
        # Load balancing importance weights (learnable)
        self.expert_importance = nn.Parameter(torch.ones(num_experts))
    
    def forward(
        self,
        triplet: torch.Tensor,
        return_load_loss: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute expert weights with noisy top-k gating.
        
        Args:
            triplet: (B, 3, embedding_dim) - initial triplet
            return_load_loss: If True, also return load balancing loss
            
        Returns:
            weights: (B, num_experts) - sparse top-k expert weights
            load_loss (optional): Load balancing auxiliary loss
        """
        B = triplet.shape[0]
        flat = triplet.view(B, -1)  # (B, 3*D)
        
        # Clean gating logits
        clean_logits = self.gate(flat)  # (B, num_experts)
        
        # Add tunable noise during training
        if self.training:
            noise = torch.randn_like(clean_logits)
            noise_scale = F.softplus(self.noise_gate(flat))
            noisy_logits = clean_logits + noise * noise_scale * self.noise_std
        else:
            noisy_logits = clean_logits
        
        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(
            noisy_logits, self.top_k, dim=-1
        )
        
        # Sparse gating: only top-k experts get non-zero weight
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Create sparse weight matrix
        weights = torch.zeros_like(noisy_logits)
        weights.scatter_(1, top_k_indices, top_k_gates)
        
        if return_load_loss:
            # Load balancing loss: encourage uniform expert usage
            # Importance: fraction of batch routed to each expert
            importance = weights.sum(dim=0) / B
            
            # Load: average gate value for each expert
            load = F.softmax(clean_logits, dim=-1).mean(dim=0)
            
            # CV squared: coefficient of variation loss
            load_loss = self.num_experts * (importance * load).sum()
            
            return weights, load_loss
        
        return weights


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
        
        # Cross-expert attention for integration
        self.cross_expert_attn = nn.MultiheadAttention(
            embedding_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.cross_expert_norm = nn.LayerNorm(embedding_dim)
        
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
        
        # Compute expert weights via gating with load balancing
        expert_weights, load_loss = self.expert_gate(
            initial_triplet, return_load_loss=True
        )
        self.last_load_loss = load_loss.detach() if load_loss is not None else None  # Detach to avoid graph retention
        
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
