"""
Compatibility shim for legacy imports.

This module re-exports DreamingReasoningLLM so that code importing
ImageTokenReasoningLLM keeps working while the project fully transitions
to the dreaming-based architecture.
"""

from __future__ import annotations

import warnings
from typing import Optional

from .config import ExperimentConfig
from .dreaming_model import DreamingReasoningLLM


class ImageTokenReasoningLLM(DreamingReasoningLLM):
    """
    Deprecated alias for DreamingReasoningLLM.
    
    Please import DreamingReasoningLLM from image_token_llm.dreaming_model instead.
    This class exists only for backward compatibility with legacy code.
    """
    
    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        device: Optional[str] = None,
        embedding_dim: int = 512,
        vocab_size: int = 4096,
        enable_rl: bool = True,
    ) -> None:
        """Initialize with deprecation warning."""
        warnings.warn(
            "ImageTokenReasoningLLM is deprecated. Please import "
            "DreamingReasoningLLM from image_token_llm.dreaming_model instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        
        super().__init__(
            config=config,
            device=device,
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            enable_rl=enable_rl,
        )


__all__ = ["ImageTokenReasoningLLM", "DreamingReasoningLLM"]
