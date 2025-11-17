#!/usr/bin/env python3
"""Simple Ollama distillation for DreamingReasoningLLM."""

import argparse
import logging
from pathlib import Path
import torch
import ollama

from image_token_llm.dreaming_model import DreamingReasoningLLM

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="llama3.2:1b")
    parser.add_argument("--prompts", type=int, default=10)
    parser.add_argument("--output", default="./models/distilled")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Distilling from Ollama")
    logger.info("=" * 60)
    
    # Init model
    logger.info("\nInitializing student...")
    student = DreamingReasoningLLM(
        embedding_dim=256,
        vocab_size=512,
        enable_rl=True,
        device=args.device
    )
    logger.info(f"Parameters: {sum(p.numel() for p in student.parameters()):,}")
    
    # Prompts
    prompts = [
        "What happens when you drop a ball?",
        "Why does ice melt?",
        "Explain gravity.",
        "What is rain?",
        "How do plants grow?",
        "Why is the sky blue?",
        "What makes fire hot?",
        "How does a car move?",
        "Why do we sleep?",
        "What is electricity?",
    ][:args.prompts]
    
    # Query teacher
    logger.info(f"\nQuerying teacher: {args.teacher}")
    traces = []
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"[{i}/{len(prompts)}] {prompt}")
        try:
            resp = ollama.generate(
                model=args.teacher,
                prompt=prompt,
                options={"temperature": 0.8, "num_predict": 100}
            )
            answer = resp.get("response", "")
            logger.info(f"-> {answer[:70]}...")
            traces.append({"prompt": prompt, "response": answer})
        except Exception as e:
            logger.warning(f"Failed: {e}")
    
    logger.info(f"\nCollected {len(traces)} traces")
    
    # Train
    logger.info("\nTraining student...")
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    
    for epoch in range(2):
        logger.info(f"\nEpoch {epoch+1}/2")
        for i, trace in enumerate(traces):
            optimizer.zero_grad()
            
            # Tokenize
            prompt_tok = torch.tensor(
                [[ord(c) % 512 for c in trace["prompt"][:50]]],
                device=args.device
            )
            target_tok = torch.tensor(
                [[ord(c) % 512 for c in trace["response"][:30]]],
                device=args.device
            )
            
            # Forward
            logits = student(text_tokens=prompt_tok, output_mode="text")
            loss = torch.nn.functional.mse_loss(
                logits.mean(), target_tok.float().mean()
            )
            
            loss.backward()
            optimizer.step()
            
            if (i+1) % 3 == 0:
                logger.info(f"  Batch {i+1}: loss={loss.item():.4f}")
    
    # Save
    logger.info("\nSaving model...")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    student.save(args.output)
    logger.info(f"Saved to: {args.output}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
