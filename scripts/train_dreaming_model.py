#!/usr/bin/env python
"""Simple training demo for DreamingReasoningLLM with synthetic data."""

import argparse
import torch
from pathlib import Path

from image_token_llm.dreaming_model import DreamingReasoningLLM
from image_token_llm.config import ExperimentConfig, DreamingConfig


def create_synthetic_batch(batch_size: int, vocab_size: int, device: str):
    """Create synthetic training batch."""
    # Random text tokens (B, seq_len)
    seq_len = 20
    text_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Random target tokens
    target_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    return text_tokens.to(device), target_tokens.to(device)


def train_step(model, text_tokens, target_tokens, optimizer):
    """Single training step."""
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(text_tokens=text_tokens, output_mode="text")
    
    # Compute loss (cross-entropy)
    B, vocab_size = logits.shape
    target_flat = target_tokens[:, :1].reshape(-1)  # Just first token
    loss = torch.nn.functional.cross_entropy(logits, target_flat)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="Train DreamingReasoningLLM")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output", default="./models/dreaming_trained", 
                        help="Output directory")
    parser.add_argument("--enable-rl", action="store_true", 
                        help="Enable RL components")
    args = parser.parse_args()
    
    print(f"🚀 Starting training on {args.device}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   RL enabled: {args.enable_rl}")
    
    # Create model
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=2,  # Smaller for faster training
        dream_length=3,
        graph_reasoning_hops=2,
        output_mode="text"
    )
    
    model = DreamingReasoningLLM(
        config=config,
        device=args.device,
        embedding_dim=256,  # Smaller for demo
        vocab_size=1024,
        enable_rl=args.enable_rl
    )
    
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters())} params")
    
    # Optimizer (exclude RL components to avoid issues)
    trainable_params = []
    for name, param in model.named_parameters():
        if "rl_" not in name and "policy" not in name and "reward" not in name:
            trainable_params.append(param)
    
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    
    print(f"✓ Optimizer created: {len(trainable_params)} param groups")
    
    # Training loop
    print("\n📚 Training...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 10  # Small number for demo
        
        for batch_idx in range(num_batches):
            text_tokens, target_tokens = create_synthetic_batch(
                args.batch_size, model.vocab_size, args.device
            )
            
            loss = train_step(model, text_tokens, target_tokens, optimizer)
            epoch_loss += loss
            
            if batch_idx % 5 == 0:
                print(f"   Epoch {epoch+1}/{args.epochs}, "
                      f"Batch {batch_idx}/{num_batches}, "
                      f"Loss: {loss:.4f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"✓ Epoch {epoch+1} complete: Avg Loss = {avg_loss:.4f}")
    
    # Save model
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    
    print(f"\n💾 Model saved to: {output_path}")
    print(f"   Weights: dreaming_model_weights.pt")
    print(f"   Config: config.json")
    
    # Test generation
    print("\n🧪 Testing generation...")
    with torch.no_grad():
        output = model.generate(
            prompt="Test prompt",
            max_length=20,
            temperature=0.8
        )
        print(f"   Generated: {output[:50]}...")
    
    print("\n✅ Training complete!")
    print(f"   To load: DreamingReasoningLLM.load_pretrained('{output_path}')")


if __name__ == "__main__":
    main()
