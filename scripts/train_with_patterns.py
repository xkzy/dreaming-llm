#!/usr/bin/env python
"""Improved training with structured synthetic data that has learnable patterns."""

import argparse
import torch
from pathlib import Path
import random

from image_token_llm.dreaming_model import DreamingReasoningLLM
from image_token_llm.config import ExperimentConfig, DreamingConfig


def create_pattern_batch(batch_size: int, vocab_size: int, device: str, pattern_type='sequence'):
    """Create synthetic batch with learnable patterns."""
    seq_len = 20
    text_tokens = []
    target_tokens = []
    
    for _ in range(batch_size):
        if pattern_type == 'sequence':
            # Pattern: ascending sequences
            start = random.randint(0, vocab_size - seq_len - 5)
            tokens = list(range(start, start + seq_len))
            target = (tokens[0] + seq_len) % vocab_size  # Predict next in sequence
            
        elif pattern_type == 'repeat':
            # Pattern: repeated tokens
            base = random.randint(0, vocab_size - 1)
            tokens = [base] * seq_len
            target = base
            
        elif pattern_type == 'alternating':
            # Pattern: alternating between two values
            a = random.randint(0, vocab_size // 2)
            b = random.randint(vocab_size // 2, vocab_size - 1)
            tokens = [a if i % 2 == 0 else b for i in range(seq_len)]
            target = a if seq_len % 2 == 0 else b
            
        elif pattern_type == 'sum':
            # Pattern: sum of first two tokens (mod vocab_size)
            tokens = [random.randint(0, vocab_size - 1) for _ in range(seq_len)]
            target = (tokens[0] + tokens[1]) % vocab_size
            
        else:  # 'mixed'
            # Mix of patterns
            choice = random.choice(['sequence', 'repeat', 'alternating', 'sum'])
            return create_pattern_batch(batch_size, vocab_size, device, choice)
        
        text_tokens.append(tokens)
        target_tokens.append([target] + tokens[:-1])  # Shift by 1
    
    text_tokens = torch.tensor(text_tokens, dtype=torch.long)
    target_tokens = torch.tensor(target_tokens, dtype=torch.long)
    
    return text_tokens.to(device), target_tokens.to(device)


def train_step(model, text_tokens, target_tokens, optimizer, use_contrastive=False):
    """Single training step with optional contrastive loss."""
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(text_tokens=text_tokens, output_mode="text")
    
    # Compute main loss (cross-entropy)
    B, vocab_size = logits.shape
    target_flat = target_tokens[:, 0].reshape(-1)
    main_loss = torch.nn.functional.cross_entropy(logits, target_flat)
    
    total_loss = main_loss
    
    # Optional: Add contrastive loss if enabled
    if use_contrastive and hasattr(model, 'input_tokenizer'):
        try:
            # Get text embeddings
            text_emb = model.text_embedder(text_tokens).mean(dim=1)
            
            # Compute contrastive loss
            contrastive_loss = model.input_tokenizer.contrastive_loss(
                text_embedding=text_emb,
                image_embeddings=text_emb  # Use same for simplicity
            )
            
            total_loss = main_loss + 0.1 * contrastive_loss
        except Exception:
            pass  # Skip if contrastive loss fails
    
    # Backward pass
    total_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return total_loss.item()


def main():
    parser = argparse.ArgumentParser(description="Train DreamingReasoningLLM with patterns")
    parser.add_argument("--device", default="cuda", help="Device (cpu/cuda)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--output", default="./models/trained_patterns", 
                        help="Output directory")
    parser.add_argument("--pattern", default="mixed", 
                        choices=['sequence', 'repeat', 'alternating', 'sum', 'mixed'],
                        help="Pattern type to learn")
    parser.add_argument("--use-contrastive", action="store_true",
                        help="Use contrastive loss")
    args = parser.parse_args()
    
    print(f"🚀 Starting training on {args.device}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Pattern type: {args.pattern}")
    print(f"   Contrastive loss: {args.use_contrastive}")
    
    # Create model with smaller dimensions for faster training
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=2,
        dream_length=3,
        graph_reasoning_hops=2,
        output_mode="text"
    )
    
    model = DreamingReasoningLLM(
        config=config,
        device=args.device,
        embedding_dim=256,
        vocab_size=512,  # Smaller vocab for patterns
        enable_rl=False
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params:,} params")
    print()
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.lr * 0.1
    )
    
    print(f"✓ Optimizer created with AdamW + Cosine schedule")
    
    # Training loop
    print("\n📚 Training...")
    best_loss = float('inf')
    num_batches = 50  # More batches per epoch
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        for batch_idx in range(num_batches):
            text_tokens, target_tokens = create_pattern_batch(
                args.batch_size, 
                model.vocab_size, 
                args.device,
                args.pattern
            )
            
            loss = train_step(
                model, 
                text_tokens, 
                target_tokens, 
                optimizer,
                args.use_contrastive
            )
            epoch_loss += loss
            
            if batch_idx % 25 == 0:
                print(f"   Epoch {epoch+1}/{args.epochs}, "
                      f"Batch {batch_idx}/{num_batches}, "
                      f"Loss: {loss:.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        avg_loss = epoch_loss / num_batches
        
        # Update learning rate
        scheduler.step()
        
        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            marker = "🌟 NEW BEST"
        else:
            marker = ""
        
        print(f"✓ Epoch {epoch+1} complete: Avg Loss = {avg_loss:.4f} {marker}")
        
        # Early stopping if loss gets very low
        if avg_loss < 0.1:
            print(f"\n🎯 Converged! Loss < 0.1")
            break
    
    # Save model
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    
    print(f"\n💾 Model saved to: {output_path}")
    print("\n" + "="*70)
    print("Training Summary:")
    print(f"   Final Loss: {avg_loss:.4f}")
    print(f"   Best Loss: {best_loss:.4f}")
    print(f"   Model: {args.output}")
    
    # Test the model
    print("\n🧪 Testing pattern recognition...")
    model.eval()
    with torch.no_grad():
        # Test on same pattern type
        test_tokens, test_targets = create_pattern_batch(
            4, model.vocab_size, args.device, args.pattern
        )
        logits = model(text_tokens=test_tokens, output_mode="text")
        predictions = logits.argmax(dim=-1)
        
        accuracy = (predictions == test_targets[:, 0]).float().mean().item()
        print(f"   Pattern accuracy: {accuracy*100:.1f}%")
    
    print("\n✅ Training complete!")
    print(f"   To load: DreamingReasoningLLM.load_pretrained('{output_path}')")


if __name__ == "__main__":
    main()
