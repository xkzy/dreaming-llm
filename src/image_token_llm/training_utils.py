"""Training utilities: curriculum learning, ALiBi, context extension."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi) for extended context.
    Replaces absolute positional embeddings with relative position bias.
    """
    
    def __init__(self, num_heads: int, max_positions: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.max_positions = max_positions
        
        # Compute slopes for each head (geometric sequence)
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
    
    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """
        Compute per-head slopes for ALiBi.
        Uses geometric sequence: 2^(-8/n), 2^(-16/n), ...
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            return torch.tensor(
                get_slopes_power_of_2(num_heads), dtype=torch.float32
            )
        else:
            # Handle non-power-of-2 heads
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            # Interpolate for extra heads
            extra = num_heads - closest_power_of_2
            slopes += get_slopes_power_of_2(2 * closest_power_of_2)[::2][:extra]
            return torch.tensor(slopes, dtype=torch.float32)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Compute ALiBi bias matrix.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            bias: (1, num_heads, seq_len, seq_len) attention bias
        """
        # Relative position matrix: (seq_len, seq_len)
        # positions[i, j] = i - j (how far back j is from i)
        positions = torch.arange(seq_len, device=self.slopes.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Apply per-head slopes: (num_heads, seq_len, seq_len)
        bias = relative_positions.unsqueeze(0) * self.slopes.unsqueeze(-1).unsqueeze(-1)
        
        # Add batch dimension: (1, num_heads, seq_len, seq_len)
        return bias.unsqueeze(0)


class RoPEPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for improved long-range attention.
    Can be scaled with YaRN for extended context.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        max_positions: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Precompute frequency bands
        inv_freq = 1.0 / (
            base ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim)
        )
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotary position encoding.
        
        Args:
            x: (B, seq_len, embedding_dim)
            seq_len: Sequence length
            
        Returns:
            x_rotated: (B, seq_len, embedding_dim)
        """
        # Position indices with scaling
        positions = torch.arange(
            seq_len, device=x.device, dtype=torch.float32
        ) / self.scaling_factor
        
        # Compute frequencies: (seq_len, embedding_dim/2)
        freqs = torch.outer(positions, self.inv_freq)
        
        # Create rotation matrix
        cos = freqs.cos()  # (seq_len, embedding_dim/2)
        sin = freqs.sin()  # (seq_len, embedding_dim/2)
        
        # Split x into even and odd dimensions
        x1 = x[..., ::2]  # (B, seq_len, embedding_dim/2)
        x2 = x[..., 1::2]  # (B, seq_len, embedding_dim/2)
        
        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # Interleave back
        x_rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        x_rotated = x_rotated.flatten(-2)
        
        return x_rotated


class CurriculumLearningScheduler:
    """
    Curriculum learning scheduler for progressive task difficulty.
    
    Stages:
    1. Core reasoning (simple patterns)
    2. Multi-hop reasoning (2-3 hops)
    3. Causal reasoning (cause-effect chains)
    4. Multimodal reasoning (text + images)
    """
    
    def __init__(
        self,
        stages: List[str] = None,
        steps_per_stage: List[int] = None,
    ):
        if stages is None:
            stages = ['core', 'multi_hop', 'causal', 'multimodal']
        if steps_per_stage is None:
            steps_per_stage = [1000, 2000, 3000, 5000]
        
        assert len(stages) == len(steps_per_stage)
        
        self.stages = stages
        self.steps_per_stage = steps_per_stage
        self.current_stage_idx = 0
        self.current_step = 0
        self.stage_start_step = 0
    
    def step(self) -> str:
        """
        Advance one training step and return current stage.
        """
        self.current_step += 1
        
        # Check if we should advance to next stage
        steps_in_current = self.current_step - self.stage_start_step
        if (
            self.current_stage_idx < len(self.stages) - 1 and
            steps_in_current >= self.steps_per_stage[self.current_stage_idx]
        ):
            self.current_stage_idx += 1
            self.stage_start_step = self.current_step
        
        return self.get_current_stage()
    
    def get_current_stage(self) -> str:
        """Get current curriculum stage."""
        return self.stages[self.current_stage_idx]
    
    def get_stage_config(self) -> Dict:
        """
        Get training config for current stage.
        """
        stage = self.get_current_stage()
        
        configs = {
            'core': {
                'dream_length': 3,
                'num_dreams': 2,
                'graph_hops': 2,
                'use_images': False,
            },
            'multi_hop': {
                'dream_length': 5,
                'num_dreams': 4,
                'graph_hops': 3,
                'use_images': False,
            },
            'causal': {
                'dream_length': 7,
                'num_dreams': 4,
                'graph_hops': 4,
                'use_images': False,
            },
            'multimodal': {
                'dream_length': 5,
                'num_dreams': 4,
                'graph_hops': 3,
                'use_images': True,
            },
        }
        
        return configs.get(stage, configs['core'])
    
    def get_progress(self) -> float:
        """Get progress in current stage (0.0 to 1.0)."""
        steps_in_current = self.current_step - self.stage_start_step
        total_steps = self.steps_per_stage[self.current_stage_idx]
        return min(steps_in_current / total_steps, 1.0)


class SparseMaxActivation(nn.Module):
    """
    SparseMax activation: sparse alternative to softmax.
    Projects onto probability simplex with sparsity.
    """
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsemax activation.
        
        Args:
            x: Input logits (any shape)
            
        Returns:
            probs: Sparse probability distribution
        """
        # Sort input in descending order
        sorted_x, sorted_idx = torch.sort(x, dim=self.dim, descending=True)
        
        # Compute cumulative sum
        cumsum = torch.cumsum(sorted_x, dim=self.dim)
        
        # Find support size k
        k_array = torch.arange(
            1, x.shape[self.dim] + 1, device=x.device, dtype=x.dtype
        )
        
        # Expand k_array to match x shape
        k_shape = [1] * len(x.shape)
        k_shape[self.dim] = -1
        k_array = k_array.view(k_shape)
        
        # Check condition: sorted_x > (cumsum - 1) / k
        condition = sorted_x > (cumsum - 1) / k_array
        
        # Find largest k where condition holds
        k_max = condition.sum(dim=self.dim, keepdim=True)
        
        # Compute threshold tau
        tau = (
            torch.gather(cumsum, self.dim, k_max - 1) - 1
        ) / k_max.float()
        
        # Compute output
        output = torch.clamp(x - tau, min=0.0)
        
        return output


def create_extended_context_model(
    base_model: nn.Module,
    use_alibi: bool = True,
    use_rope: bool = False,
    context_length: int = 1024,
) -> nn.Module:
    """
    Wrap a model to support extended context windows.
    
    Args:
        base_model: Base transformer model
        use_alibi: Use ALiBi positional bias
        use_rope: Use RoPE positional encoding
        context_length: Target context length
        
    Returns:
        Extended context model
    """
    if use_alibi:
        # Add ALiBi to all attention layers
        for module in base_model.modules():
            if isinstance(module, nn.MultiheadAttention):
                # Wrap forward to add ALiBi bias
                original_forward = module.forward
                
                def forward_with_alibi(
                    query, key, value, *args, **kwargs
                ):
                    seq_len = query.shape[1]
                    num_heads = module.num_heads
                    
                    alibi = ALiBiPositionalBias(num_heads)
                    bias = alibi(seq_len).to(query.device)
                    
                    # Add bias to attention mask
                    if 'attn_mask' in kwargs and kwargs['attn_mask'] is not None:
                        kwargs['attn_mask'] = kwargs['attn_mask'] + bias
                    else:
                        kwargs['attn_mask'] = bias
                    
                    return original_forward(query, key, value, *args, **kwargs)
                
                module.forward = forward_with_alibi
    
    return base_model
