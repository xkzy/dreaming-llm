#!/usr/bin/env python3
"""Large-scale distillation from Ollama with dataset caching.

This script performs large-scale knowledge distillation with support for:
- Dataset caching (save/load prompt-response pairs)
- Batch processing
- Resumable training
- Progress tracking

Usage:
    python scripts/distill_large_scale.py \
        --teacher llama3.2:1b \
        --num-pairs 100000 \
        --cache-dir ./datasets/distillation_cache \
        --output ./models/distilled_large_scale \
        --device cuda \
        --batch-size 32 \
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from image_token_llm.config import DreamingConfig, ExperimentConfig
from image_token_llm.dreaming_model import DreamingReasoningLLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def generate_diverse_prompts(num_prompts: int) -> List[str]:
    """Generate diverse prompts covering many domains."""
    
    # Base prompt templates
    templates = {
        "factual": [
            "What is {topic}?",
            "Explain {topic}.",
            "Describe {topic}.",
            "Tell me about {topic}.",
            "What do you know about {topic}?",
        ],
        "reasoning": [
            "What happens when {action}?",
            "Why does {phenomenon} occur?",
            "What causes {effect}?",
            "How does {process} work?",
            "What is the relationship between {A} and {B}?",
        ],
        "procedural": [
            "How do you {task}?",
            "What are the steps to {goal}?",
            "Explain the process of {activity}.",
            "Describe how to {objective}.",
            "What's the best way to {action}?",
        ],
        "creative": [
            "Write a short story about {topic}.",
            "Describe {scene} in detail.",
            "Imagine {scenario}.",
            "Create a poem about {subject}.",
            "What would {hypothetical} be like?",
        ],
        "analytical": [
            "Compare {A} and {B}.",
            "What are the advantages of {option}?",
            "Analyze {situation}.",
            "What are the implications of {event}?",
            "Evaluate {proposal}.",
        ],
    }
    
    # Topic pools
    topics = {
        "science": [
            "gravity", "photosynthesis", "evolution", "atoms",
            "electricity", "magnetism", "DNA", "cells"
        ],
        "history": [
            "World War II", "ancient Rome", "Renaissance",
            "Industrial Revolution", "moon landing"
        ],
        "geography": [
            "mountains", "oceans", "climate", "ecosystems",
            "continents", "rivers"
        ],
        "technology": [
            "computers", "internet", "AI", "smartphones",
            "robotics", "programming"
        ],
        "arts": [
            "painting", "music", "literature", "sculpture",
            "theater", "cinema"
        ],
        "everyday": [
            "cooking", "gardening", "sports", "traveling",
            "reading", "exercise"
        ],
    }
    
    prompts = []
    
    # Generate prompts
    for i in range(num_prompts):
        # Select template type
        template_type = list(templates.keys())[i % len(templates)]
        template = templates[template_type][i % len(templates[template_type])]
        
        # Select topic
        topic_category = list(topics.keys())[i % len(topics)]
        topic = topics[topic_category][i % len(topics[topic_category])]
        
        # Generate prompt
        if "{topic}" in template:
            prompt = template.format(topic=topic)
        elif "{action}" in template:
            actions = [
                "you heat water", "you mix colors",
                "plants get sunlight"
            ]
            prompt = template.format(action=actions[i % len(actions)])
        elif "{task}" in template:
            tasks = [
                "solve a math problem", "write an essay",
                "fix a bicycle"
            ]
            prompt = template.format(task=tasks[i % len(tasks)])
        else:
            prompt = template
        
        prompts.append(prompt)
    
    logger.info(f"Generated {len(prompts)} diverse prompts")
    return prompts


def load_cached_dataset(cache_path: Path) -> Optional[List[Dict]]:
    """Load cached dataset if exists."""
    if not cache_path.exists():
        return None
    
    logger.info(f"Loading cached dataset from {cache_path}...")
    with open(cache_path, "r") as f:
        data = json.load(f)
    
    logger.info(f"✓ Loaded {len(data)} cached pairs")
    return data


def save_dataset_cache(data: List[Dict], cache_path: Path) -> None:
    """Save dataset to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving dataset cache to {cache_path}...")
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"✓ Cached {len(data)} pairs")


def generate_dataset_from_ollama(
    teacher_model: str,
    num_pairs: int,
    cache_path: Path,
    resume: bool = True,
    save_interval: int = 100
) -> List[Dict]:
    """Generate or load dataset from Ollama teacher."""
    
    # Try to load cached dataset
    if resume:
        cached_data = load_cached_dataset(cache_path)
        if cached_data is not None:
            if len(cached_data) >= num_pairs:
                return cached_data[:num_pairs]
            else:
                logger.info(
                    f"Cache has {len(cached_data)} pairs, "
                    f"need {num_pairs - len(cached_data)} more"
                )
                start_idx = len(cached_data)
        else:
            cached_data = []
            start_idx = 0
    else:
        cached_data = []
        start_idx = 0
    
    # Generate prompts
    prompts = generate_diverse_prompts(num_pairs)
    
    # Generate responses from Ollama
    logger.info("="*60)
    logger.info(
        f"Generating {num_pairs - start_idx} responses from "
        f"{teacher_model}..."
    )
    logger.info("="*60)
    
    try:
        import ollama
    except ImportError:
        logger.error("❌ Ollama Python package not installed")
        logger.error("   Install with: pip install ollama")
        return cached_data
    
    dataset = list(cached_data)  # Start with cached data
    
    for i in tqdm(range(start_idx, num_pairs), desc="Generating"):
        prompt = prompts[i]
        
        try:
            response = ollama.generate(
                model=teacher_model,
                prompt=prompt,
                options={"temperature": 0.8, "num_predict": 128}
            )
            teacher_response = response.get("response", "")
            
            dataset.append({
                "prompt": prompt,
                "response": teacher_response,
                "index": i
            })
            
            # Save checkpoint at specified intervals
            if (i + 1) % save_interval == 0:
                save_dataset_cache(dataset, cache_path)
                pairs_saved = i + 1
                percent = (pairs_saved / num_pairs) * 100
                logger.info(
                    f"💾 Checkpoint: {pairs_saved:,}/{num_pairs:,} "
                    f"({percent:.1f}%) saved"
                )
        
        except Exception as e:
            logger.warning(f"Failed to generate pair {i}: {e}")
            continue
    
    # Final save
    save_dataset_cache(dataset, cache_path)
    
    logger.info("")
    logger.info(f"✓ Dataset complete: {len(dataset)} pairs")
    logger.info("")
    
    return dataset


def train_on_dataset(
    model: DreamingReasoningLLM,
    dataset: List[Dict],
    batch_size: int,
    epochs: int,
    device: str,
    vocab_size: int
) -> Dict:
    """Train model on cached dataset."""
    
    logger.info("="*60)
    logger.info("🎯 Training on dataset")
    logger.info("="*60)
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs: {epochs}")
    logger.info("")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        
        # Batch processing
        for batch_start in tqdm(
            range(0, len(dataset), batch_size),
            desc=f"Epoch {epoch+1}"
        ):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch = dataset[batch_start:batch_end]
            
            batch_loss = 0.0
            
            for item in batch:
                prompt = item["prompt"]
                teacher_response = item["response"]
                
                try:
                    optimizer.zero_grad()
                    
                    # Tokenize prompt
                    prompt_tokens = torch.tensor(
                        [[ord(c) % vocab_size for c in prompt[:100]]],
                        device=device
                    )
                    
                    # Generate from student
                    logits = model(
                        text_tokens=prompt_tokens,
                        output_mode="text"
                    )
                    
                    # Target tokens
                    target_tokens = torch.tensor(
                        [[ord(c) % vocab_size
                          for c in teacher_response[:50]]],
                        device=device
                    )
                    
                    # Compute loss
                    loss = torch.nn.functional.mse_loss(
                        logits.mean(),
                        target_tokens.float().mean()
                    )
                    
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss += loss.item()
                
                except Exception as e:
                    logger.debug(f"Training step failed: {e}")
                    continue
            
            avg_batch_loss = batch_loss / len(batch) if batch else 0.0
            epoch_loss += batch_loss
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / len(dataset) if dataset else 0.0
        total_loss += epoch_loss
        
        logger.info(f"  Avg loss: {avg_epoch_loss:.4f}")
        logger.info("")
    
    avg_total_loss = total_loss / (len(dataset) * epochs) if dataset else 0.0
    
    return {
        "total_loss": total_loss,
        "avg_loss": avg_total_loss,
        "num_batches": num_batches,
        "num_epochs": epochs
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Large-scale Ollama distillation with caching"
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="llama3.2:1b",
        help="Ollama teacher model"
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=100000,
        help="Number of prompt-response pairs to generate"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./datasets/distillation_cache",
        help="Directory for caching datasets"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/distilled_large_scale",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=512,
        help="Embedding dimension"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8192,
        help="Vocabulary size"
    )
    parser.add_argument(
        "--enable-rl",
        action="store_true",
        help="Enable RL components"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from cached dataset"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Save cache every N pairs (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"{args.teacher.replace(':', '_')}_{args.num_pairs}.json"
    
    logger.info("="*60)
    logger.info("🚀 LARGE-SCALE DISTILLATION")
    logger.info("="*60)
    logger.info(f"Teacher: {args.teacher}")
    logger.info(f"Pairs: {args.num_pairs:,}")
    logger.info(f"Cache: {cache_file}")
    logger.info(f"Save interval: Every {args.save_interval} pairs")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("")
    
    # Step 1: Generate or load dataset
    dataset = generate_dataset_from_ollama(
        teacher_model=args.teacher,
        num_pairs=args.num_pairs,
        cache_path=cache_file,
        resume=not args.no_resume,
        save_interval=args.save_interval
    )
    
    if len(dataset) < args.num_pairs:
        logger.warning(
            f"⚠️  Only generated {len(dataset)} pairs "
            f"(requested {args.num_pairs})"
        )
    
    # Step 2: Initialize student model
    logger.info("📦 Initializing DreamingReasoningLLM...")
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=4,
        dream_length=5,
        graph_reasoning_hops=3,
        output_mode="text"
    )
    
    model = DreamingReasoningLLM(
        config=config,
        device=args.device,
        embedding_dim=args.embedding_dim,
        vocab_size=args.vocab_size,
        enable_rl=args.enable_rl
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {num_params:,}")
    logger.info(f"  Embedding dim: {args.embedding_dim}")
    logger.info(f"  Vocab size: {args.vocab_size}")
    logger.info("")
    
    # Step 3: Train on dataset
    metrics = train_on_dataset(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        vocab_size=args.vocab_size
    )
    
    logger.info("="*60)
    logger.info("✅ Training Complete")
    logger.info("="*60)
    logger.info(f"Total loss: {metrics['total_loss']:.4f}")
    logger.info(f"Avg loss: {metrics['avg_loss']:.6f}")
    logger.info(f"Batches: {metrics['num_batches']}")
    logger.info("")
    
    # Step 4: Save model
    logger.info("💾 Saving model...")
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save(str(output_path))
    
    logger.info(f"  ✓ Saved to: {output_path}")
    logger.info(f"  Files:")
    logger.info(f"    - dreaming_model_weights.pt")
    logger.info(f"    - config.json")
    logger.info("")
    
    # Step 5: Test generation
    logger.info("🧪 Testing generation...")
    with torch.no_grad():
        test_prompt = "What happens when you drop a ball?"
        result = model.generate(
            prompt=test_prompt,
            max_length=50,
            temperature=0.8,
            return_dreams=True
        )
        
        logger.info(f"  Prompt: {test_prompt}")
        logger.info(f"  Output: {result['output'][:100]}...")
        logger.info(f"  Dreams: {len(result['dreams'])} sequences")
        logger.info(f"  Graph: {result['graph_data']['num_nodes']} nodes")
    
    logger.info("")
    logger.info("="*60)
    logger.info("✨ DISTILLATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"Dataset: {len(dataset)} pairs cached at {cache_file}")
    logger.info(f"Model: {output_path}")
    logger.info(f"To load: DreamingReasoningLLM.load_pretrained('{args.output}')")
    logger.info("")


if __name__ == "__main__":
    main()
