#!/usr/bin/env python3
"""Benchmark pretrained model performance.

Evaluates generation speed, quality metrics, and resource usage.

Usage:
    python scripts/benchmark_model.py ./pretrained_llama3
"""

from __future__ import annotations

import argparse
import time
from typing import List

from image_token_llm.model import ImageTokenReasoningLLM


def benchmark_prompts() -> List[str]:
    """Generate diverse benchmark prompts."""
    return [
        "What is machine learning?",
        "Explain the water cycle.",
        "How does a computer work?",
        "Describe the solar system.",
        "What is photosynthesis?",
        "Explain gravity in simple terms.",
        "How do airplanes fly?",
        "What causes seasons?",
        "Describe the human brain.",
        "How does electricity work?",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark pretrained model"
    )
    parser.add_argument(
        "bundle_dir",
        type=str,
        help="Path to pretrained model bundle directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Max tokens per generation",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per prompt",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" " * 20 + "Model Benchmark")
    print("=" * 70)
    print(f"Bundle: {args.bundle_dir}")
    print(f"Device: {args.device}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Runs per prompt: {args.runs}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    start_load = time.time()
    model = ImageTokenReasoningLLM.load_from_bundle(
        bundle_dir=args.bundle_dir,
        device=args.device,
        enable_rl=False,
    )
    load_time = time.time() - start_load
    print(f"✓ Model loaded in {load_time:.2f}s")

    # Get benchmark prompts
    prompts = benchmark_prompts()
    print(f"\nRunning benchmark on {len(prompts)} prompts...")

    # Benchmark
    total_time = 0.0
    total_tokens = 0
    results = []

    for i, prompt in enumerate(prompts, 1):
        prompt_times = []

        for run in range(args.runs):
            start = time.time()
            output = model.generate(
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=0.8,
                stream=False,
            )
            elapsed = time.time() - start

            prompt_times.append(elapsed)
            total_time += elapsed
            if isinstance(output, str):
                total_tokens += len(output)

        avg_time = sum(prompt_times) / len(prompt_times)
        results.append((prompt, avg_time))

        print(
            f"[{i}/{len(prompts)}] {prompt[:40]:<40} "
            f"{avg_time:.3f}s"
        )

    # Summary
    print("\n" + "=" * 70)
    print(" " * 25 + "Summary")
    print("=" * 70)
    print(f"Total prompts: {len(prompts)}")
    print(f"Total runs: {len(prompts) * args.runs}")
    print(f"Total time: {total_time:.2f}s")
    print(
        f"Average time per generation: "
        f"{total_time / (len(prompts) * args.runs):.3f}s"
    )
    print(f"Total characters generated: {total_tokens}")
    print(
        f"Characters per second: "
        f"{total_tokens / total_time:.1f}"
    )
    print("=" * 70)

    # Slowest/fastest
    results.sort(key=lambda x: x[1])
    print(
        f"\nFastest: {results[0][0][:50]} "
        f"({results[0][1]:.3f}s)"
    )
    print(
        f"Slowest: {results[-1][0][:50]} "
        f"({results[-1][1]:.3f}s)"
    )
    print()


if __name__ == "__main__":
    main()
