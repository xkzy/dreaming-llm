#!/usr/bin/env python3
"""Create a pretrained model via distillation from an Ollama teacher.

This script initializes a fresh ImageTokenReasoningLLM, distills knowledge
from an Ollama-hosted teacher model (e.g., llama2, mistral, phi), and
exports the resulting weights as an Ollama-compatible bundle.

Usage:
    python scripts/create_pretrained_model.py \
        --teacher llama2 \
        --prompts 100 \
        --output ./pretrained \
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
from typing import List

import torch

from image_token_llm.config import ExperimentConfig
from image_token_llm.model import ImageTokenReasoningLLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def generate_training_prompts(num_prompts: int) -> List[str]:
    """Generate diverse prompts for distillation training."""
    categories = [
        # Factual knowledge
        "What is the capital of France?",
        "Explain the concept of gravity.",
        "Who wrote Romeo and Juliet?",
        # Reasoning
        "If it takes 5 machines 5 minutes to make 5 widgets, how long "
        "would it take 100 machines to make 100 widgets?",
        "A farmer has 17 sheep and all but 9 die. How many are left?",
        # Creative
        "Write a short poem about autumn.",
        "Describe a futuristic city.",
        # Multi-step
        "Plan a week-long vacation to Japan.",
        "How would you design a treehouse?",
        # Code/logic
        "Write a function to find prime numbers.",
        "Explain recursion with an example.",
        # Visual reasoning (adapted)
        "Describe what happens when you drop a ball.",
        "What are the steps to bake a cake?",
    ]

    prompts: List[str] = []
    for i in range(num_prompts):
        prompts.append(categories[i % len(categories)])

    logger.info(f"Generated {len(prompts)} training prompts")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create pretrained model via Ollama distillation"
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="llama2",
        help="Ollama teacher model name (e.g., llama2, mistral, phi)",
    )
    parser.add_argument(
        "--prompts",
        type=int,
        default=100,
        help="Number of training prompts to use",
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help="Number of teacher samples per prompt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./pretrained",
        help="Output directory for pretrained bundle",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--vision-backbone",
        type=str,
        default="lite",
        choices=["clip", "resnet", "lite"],
        help="Vision encoder backbone",
    )
    parser.add_argument(
        "--no-rl",
        action="store_true",
        help="Disable RL components (faster init)",
    )
    parser.add_argument(
        "--bundle-name",
        type=str,
        default="image-token-llm-pretrained",
        help="Name for exported bundle",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Image-Token LLM Pretrained Model Creator")
    logger.info("=" * 70)
    logger.info(f"Teacher model: {args.teacher}")
    logger.info(f"Training prompts: {args.prompts}")
    logger.info(f"Samples per prompt: {args.samples_per_prompt}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Vision backbone: {args.vision_backbone}")
    logger.info(f"RL enabled: {not args.no_rl}")
    logger.info(f"Output directory: {args.output}")
    logger.info("=" * 70)

    # Initialize fresh model
    logger.info("Initializing model...")
    config = ExperimentConfig()
    model = ImageTokenReasoningLLM(
        config=config,
        device=args.device,
        vision_backbone=args.vision_backbone,
        enable_rl=not args.no_rl,
    )
    logger.info("Model initialized successfully")

    # Generate training prompts
    logger.info("Generating training prompts...")
    prompts = generate_training_prompts(args.prompts)

    # Distill from teacher
    logger.info(f"Starting distillation from {args.teacher}...")
    logger.info("(This may take several minutes depending on prompt count)")
    try:
        metrics = model.distill_from_ollama(
            prompts=prompts,
            teacher_model=args.teacher,
            num_samples=args.samples_per_prompt,
        )
        logger.info("Distillation complete!")
        logger.info(f"  Distillation loss: {metrics['distillation_loss']:.4f}")
        logger.info(f"  Traces processed: {metrics['traces']}")
    except Exception as e:
        logger.error(f"Distillation failed: {e}")
        logger.error(
            "Make sure Ollama is installed and the teacher model is available:"
        )
        logger.error(f"  ollama pull {args.teacher}")
        raise

    # Export bundle
    logger.info(f"Exporting pretrained bundle to {args.output}...")
    output_path = model.export_ollama_bundle(
        output_dir=args.output,
        bundle_name=args.bundle_name,
    )
    logger.info(f"Bundle exported successfully to: {output_path}")

    # Summary
    logger.info("=" * 70)
    logger.info("Pretrained Model Creation Complete!")
    logger.info("=" * 70)
    logger.info(f"Bundle location: {output_path}")
    logger.info(f"Weights file: {args.bundle_name}_weights.pt")
    logger.info("Tokenizer: tokenizer.json")
    logger.info("Config: config.json")
    logger.info("")
    logger.info("To load this model:")
    logger.info("  from image_token_llm.model import ImageTokenReasoningLLM")
    logger.info(
        f'  model = ImageTokenReasoningLLM.load_from_bundle("{output_path}")'
    )
    logger.info('  output = model.generate("Your prompt here")')
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
