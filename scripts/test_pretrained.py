#!/usr/bin/env python3
"""Test loading and using a pretrained model bundle.

Usage:
    python scripts/test_pretrained.py ./pretrained_llama3
"""

from __future__ import annotations

import argparse
import sys

from image_token_llm.model import ImageTokenReasoningLLM


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test pretrained model bundle"
    )
    parser.add_argument(
        "bundle_dir",
        type=str,
        help="Path to pretrained model bundle directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is machine learning?",
        help="Test prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Pretrained Model Bundle Test")
    print("=" * 70)
    print(f"Bundle: {args.bundle_dir}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Load model from bundle
    print("\nLoading model from bundle...")
    try:
        model = ImageTokenReasoningLLM.load_from_bundle(
            bundle_dir=args.bundle_dir,
            device=args.device,
            enable_rl=False,
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)

    # Generate text
    print(f"\nGenerating response to: '{args.prompt}'")
    print("-" * 70)
    try:
        output = model.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=0.8,
            stream=False,
        )
        print(f"Output: {output}")
        print("-" * 70)
        print("✓ Generation successful")
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        sys.exit(1)

    # Display metadata
    if model.last_metadata:
        print("\nGeneration metadata:")
        for key, value in model.last_metadata.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
