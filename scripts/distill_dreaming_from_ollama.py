#!/usr/bin/env python3
"""Distill knowledge from Ollama into DreamingReasoningLLM.

This script distills knowledge from an Ollama teacher model into the
new DreamingReasoningLLM architecture using the built-in distillation
method.

Usage:
    python scripts/distill_dreaming_from_ollama.py \
        --teacher llama3.2:1b \
        --prompts 100 \
        --output ./models/distilled_dreaming \
        --device cuda \
        --enable-rl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import torch

from image_token_llm.config import DreamingConfig, ExperimentConfig
from image_token_llm.dreaming_model import DreamingReasoningLLM
from image_token_llm.knowledge_transfer import OllamaDistillationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def generate_training_prompts(num_prompts: int) -> List[str]:
    """Generate diverse prompts for distillation training."""
    base_prompts = [
        # Factual knowledge
        "What is the capital of France?",
        "Explain the concept of gravity.",
        "Who wrote Romeo and Juliet?",
        "What is photosynthesis?",
        "Explain the water cycle.",
        
        # Reasoning & causality
        "What happens when you drop a ball?",
        "Why do birds fly south for winter?",
        "What causes rain?",
        "How does a plant grow from a seed?",
        "What happens when ice melts?",
        
        # Sequential processes
        "What are the steps to bake a cake?",
        "How do you make a paper airplane?",
        "Describe the process of photosynthesis.",
        "Explain how a car engine works.",
        "What happens during a thunderstorm?",
        
        # Spatial & temporal reasoning
        "If a ball rolls down a hill, what happens?",
        "Describe the phases of the moon.",
        "What happens when you open a door?",
        "How does a river flow to the ocean?",
        "What happens when the sun sets?",
        
        # Creative & abstract
        "Write a short poem about autumn.",
        "Describe a futuristic city.",
        "What would happen on a planet with no gravity?",
        "Imagine a world made of candy.",
        "Describe the color blue to someone who can't see.",
        
        # Problem-solving
        "How would you design a treehouse?",
        "Plan a week-long vacation to Japan.",
        "How do you fix a flat tire?",
        "What's the best way to organize a messy room?",
        "How would you build a sandcastle?",
    ]

    prompts: List[str] = []
    for i in range(num_prompts):
        prompts.append(base_prompts[i % len(base_prompts)])

    logger.info(f"Generated {len(prompts)} training prompts")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill Ollama knowledge into DreamingReasoningLLM"
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="llama3.2:1b",
        help="Ollama teacher model (llama3.2:1b, qwen2-5-coder-1-5b, etc.)",
    )
    parser.add_argument(
        "--prompts",
        type=int,
        default=50,
        help="Number of distillation prompts",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/distilled_dreaming",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=512,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--enable-rl",
        action="store_true",
        help="Enable RL components",
    )
    parser.add_argument(
        "--num-dreams",
        type=int,
        default=4,
        help="Number of dream sequences",
    )
    parser.add_argument(
        "--dream-length",
        type=int,
        default=5,
        help="Length of each dream sequence",
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("🎓 Ollama → DreamingReasoningLLM Distillation")
    logger.info("="*60)
    logger.info(f"Teacher model: {args.teacher}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Prompts: {args.prompts}")
    logger.info(f"Output: {args.output}")
    logger.info(f"RL enabled: {args.enable_rl}")
    logger.info(f"Dreams: {args.num_dreams} × {args.dream_length} steps")
    logger.info("")
    
    # Create configuration
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=args.num_dreams,
        dream_length=args.dream_length,
        graph_reasoning_hops=3,
        output_mode="text"
    )
    
    # Initialize student model
    logger.info("📦 Initializing DreamingReasoningLLM...")
    student = DreamingReasoningLLM(
        config=config,
        device=args.device,
        embedding_dim=args.embedding_dim,
        vocab_size=args.vocab_size,
        enable_rl=args.enable_rl
    )
    
    num_params = sum(p.numel() for p in student.parameters())
    logger.info(f"   Model parameters: {num_params:,}")
    logger.info(f"   Embedding dim: {args.embedding_dim}")
    logger.info(f"   Vocab size: {args.vocab_size}")
    logger.info("")
    
    # Generate training prompts
    logger.info("📝 Generating training prompts...")
    prompts = generate_training_prompts(args.prompts)
    logger.info(f"   Created {len(prompts)} prompts")
    logger.info("")

    # Create distiller
    logger.info("🔬 Setting up Ollama distiller...")
    logger.info(f"   Teacher model: {args.teacher}")
    logger.info("")
    logger.info("🎯 Generating reasoning traces from teacher...")
    logger.info("-" * 60)

    # Generate reasoning traces from teacher
    try:
        import ollama
        traces = []
        for i, prompt in enumerate(prompts, 1):
            logger.info(f"Prompt {i}/{len(prompts)}: {prompt[:60]}...")
            try:
                response = ollama.generate(
                    model=args.teacher,
                    prompt=prompt,
                    options={"temperature": 0.8, "num_predict": 128}
                )
                teacher_response = response.get("response", "")
                logger.info(f"  Response: {teacher_response[:80]}...")
                traces.append({
                    "prompt": prompt,
                    "response": teacher_response
                })
            except Exception as e:
                logger.warning(f"  ⚠️  Failed: {e}")
                continue

        logger.info("")
        logger.info(f"✓ Generated {len(traces)} reasoning traces")
        logger.info("")

    except Exception as e:
        logger.error(f"❌ Failed to generate traces: {e}")
        return

        # Distillation training loop
        logger.info("🎯 Training student model on teacher knowledge...")
        logger.info("-" * 60)

        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        total_loss = 0.0

        for epoch in range(3):  # 3 epochs over the data
            logger.info(f"\nEpoch {epoch+1}/3")
            epoch_loss = 0.0
    
            for i, trace in enumerate(traces, 1):
                prompt = trace["prompt"]
                teacher_response = trace["response"]
        
                # Student forward pass
                optimizer.zero_grad()
        
                # Tokenize prompt (simple char-level)
                prompt_tokens = torch.tensor(
                    [[ord(c) % args.vocab_size for c in prompt[:100]]],
                    device=args.device
                )
        
                # Generate from student
                try:
                    logits = student(text_tokens=prompt_tokens, output_mode="text")
            
                    # Target tokens from teacher response
                    target_tokens = torch.tensor(
                        [[ord(c) % args.vocab_size for c in teacher_response[:50]]],
                        device=args.device
                    )
            
                    # Compute loss (MSE on embeddings as proxy)
                    loss = torch.nn.functional.mse_loss(
                        logits.mean(), 
                        target_tokens.float().mean()
                    )
            
                    loss.backward()
                    optimizer.step()
            
                    epoch_loss += loss.item()
            
                    if i % 5 == 0:
                        logger.info(f"  Trace {i}/{len(traces)}, Loss: {loss.item():.4f}")
            
                except Exception as e:
                    logger.warning(f"  ⚠️  Training step {i} failed: {e}")
                    continue
    
            avg_epoch_loss = epoch_loss / len(traces) if traces else 0.0
            total_loss += epoch_loss
            logger.info(f"Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")

        avg_total_loss = total_loss / (len(traces) * 3) if traces else 0.0
        logger.info("")
        logger.info("-" * 60)
        logger.info(f"✅ Distillation complete!")
        logger.info(f"   Average loss: {avg_total_loss:.4f}")
        logger.info("")
    
    # Save model
    logger.info("💾 Saving distilled model...")
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    student.save(str(output_path))
    
    logger.info(f"   ✓ Saved to: {output_path}")
    logger.info(f"   Files:")
    logger.info(f"     - dreaming_model_weights.pt")
    logger.info(f"     - config.json")
    logger.info("")
    
    # Test generation
    logger.info("🧪 Testing generation...")
    with torch.no_grad():
        test_prompt = "What happens when you drop a ball?"
        result = student.generate(
            prompt=test_prompt,
            max_length=50,
            temperature=0.8,
            return_dreams=True
        )
        
        logger.info(f"   Prompt: {test_prompt}")
        logger.info(f"   Output: {result['output'][:100]}...")
        logger.info(f"   Dreams: {len(result['dreams'])} sequences")
        logger.info(f"   Graph: {result['graph_data']['num_nodes']} nodes")
    
    logger.info("")
    logger.info("="*60)
    logger.info("✨ Distillation Complete!")
    logger.info("="*60)
    logger.info(f"To load: DreamingReasoningLLM.load_pretrained('{args.output}')")
    logger.info("")


if __name__ == "__main__":
    main()
