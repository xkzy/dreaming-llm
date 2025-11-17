#!/usr/bin/env python
"""Train DreamingReasoningLLM on OpenAssistant dataset."""

import argparse
import torch
import json
from pathlib import Path
from tqdm import tqdm
import random

from image_token_llm.dreaming_model import DreamingReasoningLLM
from image_token_llm.config import ExperimentConfig, DreamingConfig


def load_openassistant_data(data_path, max_samples=None):
    """Load OpenAssistant conversations."""
    conversations = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            data = json.loads(line)
            
            # Filter for high quality English prompts and assistants
            if (data.get('role') in ['prompter', 'assistant'] and
                data.get('lang') == 'en' and
                data.get('text') and
                len(data['text']) > 10):
                
                conversations.append({
                    'text': data['text'],
                    'role': data['role']
                })
    
    print(f"Loaded {len(conversations)} conversations")
    return conversations


def create_training_pairs(conversations, tokenizer_func, 
                         max_length=128, device='cuda'):
    """Create input-output pairs from conversations."""
    pairs = []
    
    for i in range(len(conversations) - 1):
        curr = conversations[i]
        next_msg = conversations[i + 1]
        
        # Create pairs: prompter -> assistant
        if curr['role'] == 'prompter' and next_msg['role'] == 'assistant':
            input_text = curr['text'][:max_length]
            output_text = next_msg['text'][:max_length]
            
            # Simple character-level tokenization
            input_tokens = tokenizer_func(input_text)
            output_tokens = tokenizer_func(output_text)
            
            if len(input_tokens) > 5 and len(output_tokens) > 5:
                pairs.append((input_tokens, output_tokens))
    
    print(f"Created {len(pairs)} training pairs")
    return pairs


def simple_tokenize(text, vocab_size=2048, max_len=64):
    """Simple character tokenization."""
    # Convert to char IDs
    tokens = [min(ord(c), vocab_size-1) for c in text]
    
    # Pad or truncate
    if len(tokens) < max_len:
        tokens = tokens + [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    
    return tokens


def create_batch(pairs, batch_size, device):
    """Create a training batch."""
    batch_pairs = random.sample(pairs, min(batch_size, len(pairs)))
    
    input_tokens = torch.tensor(
        [p[0] for p in batch_pairs], dtype=torch.long
    ).to(device)
    
    target_tokens = torch.tensor(
        [p[1] for p in batch_pairs], dtype=torch.long
    ).to(device)
    
    return input_tokens, target_tokens


def train_epoch(model, pairs, optimizer, scheduler, args):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = len(pairs) // args.batch_size
    
    progress = tqdm(range(num_batches), desc="Training")
    
    for _ in progress:
        # Create batch
        input_tokens, target_tokens = create_batch(
            pairs, args.batch_size, args.device
        )
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(text_tokens=input_tokens, output_mode="text")
        
        # Compute loss
        B, vocab_size = logits.shape
        target_flat = target_tokens[:, 0]
        loss = torch.nn.functional.cross_entropy(logits, target_flat)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        progress.set_postfix({'loss': f'{loss.item():.4f}'})
    
    if scheduler:
        scheduler.step()
    
    return epoch_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/openassistant/OpenAssistant_oasst1_train.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="./models/trained_oa")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=2048)
    args = parser.parse_args()
    
    print(f"🚀 Training on OpenAssistant Dataset")
    print(f"   Device: {args.device}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Max samples: {args.max_samples}")
    print()
    
    # Load data
    print("📚 Loading dataset...")
    conversations = load_openassistant_data(
        args.data, 
        max_samples=args.max_samples
    )
    
    # Create training pairs
    print("🔧 Creating training pairs...")
    tokenizer = lambda text: simple_tokenize(
        text, 
        vocab_size=args.vocab_size,
        max_len=64
    )
    pairs = create_training_pairs(
        conversations, 
        tokenizer,
        device=args.device
    )
    
    if len(pairs) == 0:
        print("❌ No training pairs created!")
        return
    
    # Create model
    print("🤖 Creating model...")
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=4,
        dream_length=4,
        graph_reasoning_hops=3,
        output_mode="text"
    )
    
    model = DreamingReasoningLLM(
        config=config,
        device=args.device,
        embedding_dim=args.embedding_dim,
        vocab_size=args.vocab_size,
        enable_rl=False
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: {total_params:,} parameters")
    print()
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.1
    )
    
    # Training loop
    print("🎯 Training...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        avg_loss = train_epoch(model, pairs, optimizer, scheduler, args)
        
        marker = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            marker = "🌟 NEW BEST"
            
            # Save best model
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            model.save(str(output_path))
        
        print(f"✓ Epoch {epoch+1}: Avg Loss = {avg_loss:.4f} {marker}")
        print(f"   LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Final save
    print(f"\n💾 Final model saved to: {args.output}")
    print("\n" + "="*70)
    print("Training Summary:")
    print(f"   Final Loss: {avg_loss:.4f}")
    print(f"   Best Loss: {best_loss:.4f}")
    print(f"   Training pairs: {len(pairs)}")
    print(f"   Model: {total_params:,} parameters")
    
    # Test generation
    print("\n🧪 Testing generation...")
    model.eval()
    test_prompts = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "How do neural networks work?"
    ]
    
    with torch.no_grad():
        for prompt in test_prompts:
            output = model.generate(
                prompt=prompt,
                max_length=50,
                temperature=0.8
            )
            print(f"   Q: {prompt[:40]}...")
            print(f"   A: {output[:60]}...")
            print()
    
    print("✅ Training complete!")
    print(f"   Load with: DreamingReasoningLLM.load_pretrained('{args.output}')")


if __name__ == "__main__":
    main()
