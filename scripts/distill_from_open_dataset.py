#!/usr/bin/env python3
"""
Distill DreamingReasoningLLM from open datasets (COCO captions, LAION, OpenAssistant, etc).

import torch

Usage:
    python scripts/distill_from_open_dataset.py --dataset coco_captions --data-dir ./data/coco --output ./models/distilled_from_coco --device cuda
    python scripts/distill_from_open_dataset.py --dataset openassistant --data-dir ./data/openassistant --output ./models/distilled_from_oa --device cuda

Supported datasets:
- coco_captions: MS COCO image captions (JSON)
- openassistant: OpenAssistant conversations (JSON)
- laion: LAION image-text pairs (TSV/JSON)

"""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

import torch
from src.image_token_llm.config import ExperimentConfig
from src.image_token_llm.dreaming_model import DreamingReasoningLLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_coco_captions(data_dir: Path, limit: int = None) -> List[Dict]:
    """Load COCO captions as prompt-response pairs."""
    ann_path = data_dir / "annotations" / "captions_train2017.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"COCO captions not found: {ann_path}")
    with open(ann_path) as f:
        data = json.load(f)
    id2img = {img['id']: img['file_name'] for img in data['images']}
    pairs = []
    for ann in data['annotations']:
        img_file = id2img[ann['image_id']]
        prompt = f"Describe the image: {img_file}"
        response = ann['caption']
        pairs.append({"prompt": prompt, "response": response, "image": img_file})
        if limit and len(pairs) >= limit:
            break
    return pairs

def load_openassistant(data_dir: Path, limit: int = None) -> List[Dict]:
    """Load OpenAssistant conversations as prompt-response pairs."""
    # Parse OpenAssistant message trees from JSONL
    files = list(Path(data_dir).glob("*.jsonl*"))
    pairs = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            # Build a mapping from message_id to message object
            messages = [json.loads(line) for line in f]
            id2msg = {m['message_id']: m for m in messages}
            # Find all root messages (parent_id is None)
            roots = [m for m in messages if m.get('parent_id') is None]
            for root in roots:
                # Traverse the tree in order, collect prompt-response pairs
                stack = [(root, [])]  # (current_message, history)
                while stack:
                    msg, history = stack.pop()
                    if msg['role'] == 'prompter':
                        # Find assistant children
                        children = [m for m in messages if m.get('parent_id') == msg['message_id'] and m['role'] == 'assistant']
                        for child in children:
                            pairs.append({
                                "prompt": msg['text'],
                                "response": child['text']
                            })
                            if limit and len(pairs) >= limit:
                                return pairs
                            # Continue traversal for multi-turn
                            stack.append((child, history + [msg['text'], child['text']]))
    return pairs

def load_laion(data_dir: Path, limit: int = None) -> List[Dict]:
    """Load LAION image-text pairs (TSV or JSON)."""
    import csv
    files = list(Path(data_dir).glob("*.tsv"))
    pairs = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 2:
                    continue
                img_url, caption = row[0], row[1]
                pairs.append({"prompt": f"Describe the image: {img_url}", "response": caption, "image": img_url})
                if limit and len(pairs) >= limit:
                    return pairs
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Distill DreamingReasoningLLM from open datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["coco_captions", "openassistant", "laion"], help="Dataset type")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Output model directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--limit", type=int, help="Limit number of pairs (for testing)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--hf-tokenizer-name", type=str, default="bert-base-uncased", help="HuggingFace tokenizer name")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.dataset == "coco_captions":
        pairs = load_coco_captions(data_dir, args.limit)
    elif args.dataset == "openassistant":
        pairs = load_openassistant(data_dir, args.limit)
    elif args.dataset == "laion":
        pairs = load_laion(data_dir, args.limit)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info(f"Loaded {len(pairs)} prompt-response pairs from {args.dataset}")

    # Initialize model
    config = ExperimentConfig()
    config.hf_tokenizer_name = args.hf_tokenizer_name
    model = DreamingReasoningLLM(config, device=args.device)

    # Supervised training loop
    import torch.nn as nn
    import torch.optim as optim
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    # Use ignore_index for padding tokens
    # Ensure pad_token_id is an int, fallback to 0 if not found
    pad_token_id = 0
    if hasattr(model.tokenizer, 'pad_token_id'):
        pad_token_id_val = model.tokenizer.pad_token_id
        if isinstance(pad_token_id_val, int):
            pad_token_id = pad_token_id_val
        elif hasattr(pad_token_id_val, 'item'):
            try:
                item_attr = pad_token_id_val.item
                val = item_attr() if callable(item_attr) else item_attr
                if isinstance(val, (int, float)):
                    pad_token_id = int(val)
                else:
                    pad_token_id = 0
            except Exception:
                pad_token_id = 0
        else:
            pad_token_id = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0.0
        for i in range(0, len(pairs), args.batch_size):
            batch = pairs[i:i+args.batch_size]
            prompts = [ex["prompt"] for ex in batch]
            targets = [ex["response"] for ex in batch]
            # Tokenize prompts and targets
            input_ids = [model._tokenize_prompt(p) for p in prompts]
            target_ids = [model._tokenize_prompt(t) for t in targets]
            # Pad to max length in batch
            max_in = max(x.shape[1] for x in input_ids)
            max_out = max(x.shape[1] for x in target_ids)
            input_ids = [
                nn.functional.pad(x, (0, max_in - x.shape[1]))
                for x in input_ids
            ]
            target_ids = [
                nn.functional.pad(x, (0, max_out - x.shape[1]))
                for x in target_ids
            ]
            input_ids = torch.cat(input_ids, dim=0)
            target_ids = torch.cat(target_ids, dim=0)
            input_ids = input_ids.to(model.device)
            target_ids = target_ids.to(model.device)
            # Forward pass
            logits = model.forward(text_tokens=input_ids, output_mode="text")
            # If output is a tuple, take the first element
            if isinstance(logits, tuple):
                logits = logits[0]
            # logits: (B, seq_len, vocab_size) or (B, vocab_size)
            if logits.dim() == 2:
                logits = logits.unsqueeze(1)
            # Ensure logits and targets are same seq_len
            seq_len = min(logits.shape[1], target_ids.shape[1])
            logits = logits[:, :seq_len, :]
            target_ids = target_ids[:, :seq_len]
            # Cross-entropy expects (B*seq, vocab), targets (B*seq)
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i // args.batch_size) % 10 == 0:
                logger.info(
                    f"  Batch {i//args.batch_size+1}: loss={loss.item():.4f}"
                )
        avg_loss = total_loss / ((len(pairs) // args.batch_size) + 1)
        logger.info(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Save model
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    logger.info(f"✓ Model saved to {args.output}")

if __name__ == "__main__":
    main()
