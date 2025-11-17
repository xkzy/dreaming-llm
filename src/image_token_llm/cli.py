"""Minimal CLI entry point for sandbox experiments."""

from __future__ import annotations

import argparse
from typing import List

import torch

from .orchestrator import ReasoningOrchestrator


def random_triplet(
    dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(3, dim, dim),
        torch.randn(3, dim, dim),
        torch.randn(3, dim, dim),
    )


def run_demo(branches: int) -> None:
    orchestrator = ReasoningOrchestrator()
    orchestrator.prime_graph(
        [
            ("vision", "infers", "action"),
            ("action", "yields", "result"),
        ]
    )

    image_triplets: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = [
        random_triplet() for _ in range(branches)
    ]
    report = orchestrator.infer(
        seeds=["vision"], image_triplets=image_triplets
    )
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=int, default=2)
    args = parser.parse_args()
    run_demo(branches=args.branches)


if __name__ == "__main__":
    main()
