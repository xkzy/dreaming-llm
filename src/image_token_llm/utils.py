"""Utility helpers for configuration and device management."""

from __future__ import annotations

import os
from typing import Final

from .config import RuntimeConfig

GPT51_FLAG_ENV: Final[str] = "ENABLE_GPT51_CODEX_PREVIEW"


def resolve_device(runtime: RuntimeConfig) -> str:
    override = os.getenv("IMAGE_TOKEN_LLM_DEVICE")
    if override:
        return override
    return runtime.device


def is_gpt51_enabled(runtime: RuntimeConfig) -> bool:
    env_override = os.getenv(GPT51_FLAG_ENV)
    if env_override is not None:
        return env_override.lower() in {"1", "true", "yes"}
    return runtime.enable_gpt51_codex_preview
