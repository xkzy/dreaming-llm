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
        self, config: ImageTokenizerConfig, backbone: str = "clip"
    ) -> None:
        self.config = config
        self.encoder = VisionEncoder(config, backbone=backbone)

    def tokenize(self, batches: Iterable[torch.Tensor]) -> List[torch.Tensor]:
        """
        Tokenize a batch of images into CLIP-based semantic tokens.
        Args:
            batches: Iterable of [C, H, W] tensors
        Returns:
            List of [embedding_dim] tensors
        """
        tokens: List[torch.Tensor] = []
        for image in batches:
            if image.ndim != 3:
                raise ValueError("Each image must be a 3D tensor [C, H, W].")
            image = image.unsqueeze(0)  # Add batch dim
            with torch.no_grad():
                token = self.encoder(image).squeeze(0)
            tokens.append(token)
        return tokens
