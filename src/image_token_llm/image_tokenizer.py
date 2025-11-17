"""Utilities for converting multimodal inputs into image tokens."""

from __future__ import annotations

from typing import Iterable, List

import torch

from .config import ImageTokenizerConfig


class ImageTokenizer:
    """Simple patch-based tokenizer placeholder."""

    def __init__(self, config: ImageTokenizerConfig) -> None:
        self.config = config

    def tokenize(self, batches: Iterable[torch.Tensor]) -> List[torch.Tensor]:
        tokens: List[torch.Tensor] = []
        for image in batches:
            if image.ndim != 3:
                raise ValueError("Each image must be a 3D tensor [C, H, W].")
            patches = image.unfold(
                1, self.config.patch_size, self.config.patch_size
            )
            patches = patches.unfold(
                2, self.config.patch_size, self.config.patch_size
            )
            tokens.append(patches.flatten(1))
        return tokens
