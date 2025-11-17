"""Utilities for converting multimodal inputs into image tokens."""

from __future__ import annotations

from typing import Iterable, List

import torch

from .config import ImageTokenizerConfig

from .vision_encoder import VisionEncoder

class ImageTokenizer:
    """
    Image tokenizer using a pretrained CLIP vision encoder for semantic tokens.
    This replaces the old patch-based approach with CLIP embeddings.
    """
    def __init__(
        self,
        config: ImageTokenizerConfig,
        backbone: str = "clip",
        device: str = "cpu",
    ) -> None:
        """
        Args:
            config: ImageTokenizerConfig
            backbone: "clip", "resnet", or "lite"
            device: "cpu", "cuda", or torch.device
        """
        self.config = config
        self.encoder = VisionEncoder(
            config, backbone=backbone, device=device
        )

    def tokenize(
        self,
        batches: Iterable[torch.Tensor],
    device: str = "cpu",
        batch_size: int = 32,
    ) -> List[torch.Tensor]:
        """
        Tokenize a batch of images into CLIP-based semantic tokens.
        Efficient for large batches and production.
        Args:
            batches: Iterable of [C, H, W] tensors
            device: Optional override device
            batch_size: Batch size for chunked processing
        Returns:
            List of [embedding_dim] tensors
        """
        images = torch.stack(
            [img if img.ndim == 3 else img.squeeze(0) for img in batches]
        )
        tokens = self.encoder(images, device=device, batch_size=batch_size)
        return [t for t in tokens]
