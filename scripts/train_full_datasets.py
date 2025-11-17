#!/usr/bin/env python
"""Train DreamingReasoningLLM on combined datasets."""

import argparse
import torch
import json
from pathlib import Path
from tqdm import tqdm
import random

from image_token_llm.dreaming_model import DreamingReasoningLLM
from image_token_llm.config import ExperimentConfig, DreamingConfig


def load_openassistant(path):
    """Load OpenAssistant data."""
    convos = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if (data.get('role') in ['prompter', 'assistant'] and
                data.get('lang') == 'en' and
                data.get('text') and len(data['text']) > 10):
                convos.append({
                    'text': data['text'],
                    'role': data['role']
                })
    return convos


def load_anthropic_hh(path):
    """Load Anthropic HH-RLHF data."""
    convos = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            chosen = data.get('chosen', '')
            if chosen and len(chosen) > 20:
                # Parse conversation format
                parts = chosen.split('\n\nAssistant: ')
                if len(parts) == 2:
                    human_text = parts[0].replace('Human: ', '').strip()
                    assistant_text = parts[1].strip()
                    if human_text and assistant_text:
                        convos.append({'text': human_text, 'role': 'prompter'})
                        convos.append({'text': assistant_text, 'role': 'assistant'})
    return convos


def load_dolly(path):
    """Load Databricks Dolly data."""
    convos = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            instruction = data.get('instruction', '')
            response = data.get('response', '')
            context = data.get('context', '')
            
            if instruction and response:
                # Combine instruction with context if available
                prompt = f"{instruction}\n{context}".strip() if context else instruction
                if len(prompt) > 10 and len(response) > 10:
                    convos.append({'text': prompt, 'role': 'prompter'})
                    convos.append({'text': response, 'role': 'assistant'})
    return convos


def load_all_datasets(data_dir):
    """Load all available datasets."""
    data_dir = Path(data_dir)
    all_convos = []
    
    # OpenAssistant (full dataset, no limit - 84k examples)
    for oa_dir in ["openassistant_full", "openassistant"]:
        oa_path = data_dir / f"{oa_dir}/OpenAssistant_oasst1_train.jsonl"
        if oa_path.exists():
            print(f"Loading OpenAssistant from {oa_dir}...")
            convos = load_openassistant(oa_path)
            all_convos.extend(convos)
            print(f"  ✓ {len(convos)} messages")
            break
    
    # Anthropic HH-RLHF (check both limited and full)
    for hh_dir in ["anthropic_hh_full", "anthropic_hh"]:
        hh_path = data_dir / f"{hh_dir}/Anthropic_hh-rlhf_train.jsonl"
        if hh_path.exists():
            print(f"Loading Anthropic HH-RLHF from {hh_dir}...")
            convos = load_anthropic_hh(hh_path)
            all_convos.extend(convos)
            print(f"  ✓ {len(convos)} messages")
            break
    
    # Databricks Dolly (full 15k)
    dolly_path = data_dir / "dolly/databricks_databricks-dolly-15k_train.jsonl"
    if dolly_path.exists():
        print("Loading Databricks Dolly (full)...")
        convos = load_dolly(dolly_path)
        all_convos.extend(convos)
        print(f"  ✓ {len(convos)} messages")
    
    return all_convos


def create_training_pairs(conversations, tokenizer_func, 
                         max_length=128):
    """Create input-output pairs."""
    pairs = []
    
    for i in range(len(conversations) - 1):
        curr = conversations[i]
        next_msg = conversations[i + 1]
        
        if curr['role'] == 'prompter' and next_msg['role'] == 'assistant':
            input_text = curr['text'][:max_length]
            output_text = next_msg['text'][:max_length]
            
            input_tokens = tokenizer_func(input_text)
            output_tokens = tokenizer_func(output_text)
            
            if len(input_tokens) > 5 and len(output_tokens) > 5:
                pairs.append((input_tokens, output_tokens))
    
    return pairs


def simple_tokenize(text, vocab_size=2048, max_len=64):
    """Character tokenization."""
    tokens = [min(ord(c), vocab_size-1) for c in text]
    
    if len(tokens) < max_len:
        tokens = tokens + [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    
    return tokens


def create_batch(pairs, batch_size, device):
    """Create training batch."""
    batch_pairs = random.sample(pairs, min(batch_size, len(pairs)))
    
    input_tokens = torch.tensor(
        [p[0] for p in batch_pairs], dtype=torch.long
    ).to(device)
    
    target_tokens = torch.tensor(
        [p[1] for p in batch_pairs], dtype=torch.long
    ).to(device)
    
    return input_tokens, target_tokens


def train_epoch(model, pairs, optimizer, scheduler, args):
    """Train one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = len(pairs) // args.batch_size
    
    progress = tqdm(range(num_batches), desc="Training")
    
    for _ in progress:
        input_tokens, target_tokens = create_batch(
            pairs, args.batch_size, args.device
        )
        
        optimizer.zero_grad()
        
        # Generate sequence predictions
        B, seq_len = target_tokens.shape
        total_loss = 0.0
        
        # Teacher forcing: predict each token given previous context
        for t in range(min(10, seq_len)):  # Predict first 10 tokens
            logits = model(text_tokens=input_tokens, output_mode="text")
            target_t = target_tokens[:, t]
            loss_t = torch.nn.functional.cross_entropy(logits, target_t)
            total_loss += loss_t
        
        loss = total_loss / min(10, seq_len)
        
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
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="./models/trained_combined")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=2048)
    args = parser.parse_args()
    
    print(f"🚀 Training on Combined Datasets")
    print(f"   Device: {args.device}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print()
    
    # Load all datasets
    print("📚 Loading all datasets...")
    conversations = load_all_datasets(args.data_dir)
    print(f"\n✓ Total: {len(conversations)} messages from all datasets\n")
    
    # Create training pairs
    print("🔧 Creating training pairs...")
    tokenizer = lambda text: simple_tokenize(
        text, 
        vocab_size=args.vocab_size,
        max_len=64
    )
    pairs = create_training_pairs(conversations, tokenizer)
    print(f"✓ Created {len(pairs)} training pairs\n")
    
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
    print(f"✓ Model: {total_params:,} parameters\n")
    
    # Optimizer
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
            
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            model.save(str(output_path))
        
        print(f"✓ Epoch {epoch+1}: Loss = {avg_loss:.4f} {marker}")
        print(f"   LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Final summary
    print(f"\n💾 Model saved to: {args.output}")
    print("\n" + "="*70)
    print("Training Summary:")
    print(f"   Final Loss: {avg_loss:.4f}")
    print(f"   Best Loss: {best_loss:.4f}")
    print(f"   Training pairs: {len(pairs)}")
    print(f"   Total messages: {len(conversations)}")
    print(f"   Model parameters: {total_params:,}")
    
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


if __name__ == "__main__":
    main()
