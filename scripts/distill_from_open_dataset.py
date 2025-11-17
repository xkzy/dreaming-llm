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
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["coco_captions", "openassistant", "laion", "all"],
        help="Dataset type (or 'all' to load all available datasets)",
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Output model directory")
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
        help="Model embedding dimension (default: 512)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help="Vocabulary size for text decoder (default: 4096)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for optimizer (default: 3e-4)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed precision training (uses torch.cuda.amp)",
    )
    parser.add_argument(
        "--accumulate-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps to simulate larger batch sizes",
    )
    parser.add_argument(
        "--use-deepspeed",
        action="store_true",
        help="Enable DeepSpeed/ZeRO optimizer offload (requires deepspeed package)",
    )
    parser.add_argument(
        "--deepspeed-zero-stage",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="ZeRO optimization stage (default: 2)",
    )
    parser.add_argument(
        "--deepspeed-offload",
        type=str,
        default=None,
        choices=[None, "cpu", "nvme"],
        help="Offload optimizer/params to 'cpu' or 'nvme' when using ZeRO (optional)",
    )
    parser.add_argument("--limit", type=int, help="Limit number of pairs (for testing)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--hf-tokenizer-name", type=str, default="bert-base-uncased", help="HuggingFace tokenizer name")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    def load_all(root_dir: Path, limit: int = None):
        """Attempt to load all supported datasets from conventional subdirs.

        Looks for `coco`, `openassistant`, `laion` subdirectories under
        `root_dir` and loads whatever is present.
        """
        all_pairs = []
        # COCO
        coco_dir = root_dir / "coco"
        if coco_dir.exists():
            try:
                coco_pairs = load_coco_captions(coco_dir, limit)
                all_pairs.extend(coco_pairs)
            except Exception:
                logger.warning("Failed to load COCO captions from %s", coco_dir)
        # OpenAssistant
        oa_dir = root_dir / "openassistant"
        if oa_dir.exists():
            try:
                oa_pairs = load_openassistant(oa_dir, limit)
                all_pairs.extend(oa_pairs)
            except Exception:
                logger.warning("Failed to load OpenAssistant from %s", oa_dir)
        # LAION
        laion_dir = root_dir / "laion"
        if laion_dir.exists():
            try:
                laion_pairs = load_laion(laion_dir, limit)
                all_pairs.extend(laion_pairs)
            except Exception:
                logger.warning("Failed to load LAION from %s", laion_dir)

        return all_pairs

    if args.dataset == "coco_captions":
        pairs = load_coco_captions(data_dir, args.limit)
    elif args.dataset == "openassistant":
        pairs = load_openassistant(data_dir, args.limit)
    elif args.dataset == "laion":
        pairs = load_laion(data_dir, args.limit)
    elif args.dataset == "all":
        pairs = load_all(Path(args.data_dir), args.limit)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info(f"Loaded {len(pairs)} prompt-response pairs from {args.dataset}")

    # Initialize model
    config = ExperimentConfig()
    config.hf_tokenizer_name = args.hf_tokenizer_name
    # Create the model with requested size
    model = DreamingReasoningLLM(
        config,
        device=args.device,
        embedding_dim=args.embedding_dim,
        vocab_size=args.vocab_size,
    )

    # Create train/validation split
    import random

    random.shuffle(pairs)
    val_ratio = 0.1
    if len(pairs) >= 10:
        val_count = max(1, int(len(pairs) * val_ratio))
    else:
        val_count = 0
    if val_count > 0:
        val_pairs = pairs[:val_count]
        train_pairs = pairs[val_count:]
    else:
        train_pairs = pairs
        val_pairs = []

    logger.info(f"Using {len(train_pairs)} train pairs and {len(val_pairs)} val pairs")

    # Supervised training loop
    import torch.nn as nn
    import torch.optim as optim
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # DeepSpeed initialization (optional)
    ds_engine = None
    use_deepspeed = args.use_deepspeed
    if use_deepspeed:
        try:
            import deepspeed

            # Build a minimal DeepSpeed config
            ds_config = {
                "train_micro_batch_size_per_gpu": args.batch_size,
                "gradient_accumulation_steps": args.accumulate_steps,
                "zero_optimization": {"stage": args.deepspeed_zero_stage},
                "fp16": {"enabled": bool(args.fp16)},
            }
            # Optional offload
            if args.deepspeed_offload == "cpu":
                ds_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}
                ds_config["zero_optimization"]["offload_param"] = {"device": "cpu"}
            elif args.deepspeed_offload == "nvme":
                ds_config["zero_optimization"]["offload_optimizer"] = {"device": "nvme"}
                ds_config["zero_optimization"]["offload_param"] = {"device": "nvme"}

            # Initialize DeepSpeed engine; pass model parameters for optimizer creation
            engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                model_parameters=model.parameters(),
                config_params=ds_config,
            )
            ds_engine = engine
            # When using DeepSpeed, it manages fp16 and accumulation; disable local AMP
            use_amp = False
            scaler = None
            accumulate_steps = 1
            logger.info("Initialized DeepSpeed engine with ZeRO stage %d", args.deepspeed_zero_stage)
        except Exception as e:
            logger.warning("DeepSpeed requested but failed to import/initialize: %s", e)
            logger.warning("Falling back to native training path")
            use_deepspeed = False
            ds_engine = None
            use_amp = args.fp16 and args.device.startswith("cuda")
            scaler = torch.cuda.amp.GradScaler() if use_amp else None
            accumulate_steps = max(1, args.accumulate_steps)
    else:
        use_amp = args.fp16 and args.device.startswith("cuda")
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        accumulate_steps = max(1, args.accumulate_steps)
    # Use ignore_index for padding tokens
    # Ensure pad_token_id is an int, fallback to 0 if not found
    pad_token_id = 0
    try:
        pad_token_id_val = model.tokenizer.pad_token_id  # type: ignore[attr-defined]
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
    except Exception:
        # Model may not expose a tokenizer attribute; default to 0
        pad_token_id = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    metrics = {"epochs": args.epochs, "history": []}
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0.0
        for i in range(0, len(train_pairs), args.batch_size):
            batch = train_pairs[i:i+args.batch_size]
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
            # Use AMP and gradient accumulation if requested
            # If using DeepSpeed engine, let it handle backward/step
            if ds_engine is not None:
                # DeepSpeed expects raw loss (not scaled by accumulation)
                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                )
                loss_value = loss.item()
                ds_engine.backward(loss)
                # DeepSpeed will handle gradient accumulation internally
                ds_engine.step()
                total_loss += loss_value
            else:
                if use_amp:
                    with torch.cuda.amp.autocast():
                        loss = loss_fn(
                            logits.reshape(-1, logits.size(-1)),
                            target_ids.reshape(-1),
                        )
                    loss_value = loss.item()
                    loss = loss / accumulate_steps
                    scaler.scale(loss).backward()
                    if ((i // args.batch_size) + 1) % accumulate_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss = loss_fn(
                        logits.reshape(-1, logits.size(-1)),
                        target_ids.reshape(-1),
                    )
                    loss_value = loss.item()
                    loss = loss / accumulate_steps
                    loss.backward()
                    if ((i // args.batch_size) + 1) % accumulate_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                total_loss += loss_value
            if (i // args.batch_size) % 10 == 0:
                logger.info(f"  Batch {i//args.batch_size+1}: loss={loss_value:.4f}")
        avg_loss = total_loss / ((len(train_pairs) // args.batch_size) + 1)
        logger.info(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        epoch_metrics = {"epoch": epoch + 1, "train_loss": avg_loss}

        # Evaluate on validation set (if any)
        if len(val_pairs) > 0:
            model.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                import math
                for j in range(0, len(val_pairs), args.batch_size):
                    vbatch = val_pairs[j:j+args.batch_size]
                    vprompts = [ex["prompt"] for ex in vbatch]
                    vtargets = [ex["response"] for ex in vbatch]
                    vin_ids = [model._tokenize_prompt(p) for p in vprompts]
                    vt_ids = [model._tokenize_prompt(t) for t in vtargets]
                    max_in = max(x.shape[1] for x in vin_ids)
                    max_out = max(x.shape[1] for x in vt_ids)
                    vin_ids = [nn.functional.pad(x, (0, max_in - x.shape[1])) for x in vin_ids]
                    vt_ids = [nn.functional.pad(x, (0, max_out - x.shape[1])) for x in vt_ids]
                    vin_ids = torch.cat(vin_ids, dim=0).to(model.device)
                    vt_ids = torch.cat(vt_ids, dim=0).to(model.device)
                    vlogits = model.forward(text_tokens=vin_ids, output_mode="text")
                    if isinstance(vlogits, tuple):
                        vlogits = vlogits[0]
                    if vlogits.dim() == 2:
                        vlogits = vlogits.unsqueeze(1)
                    seq_len = min(vlogits.shape[1], vt_ids.shape[1])
                    vlogits = vlogits[:, :seq_len, :]
                    vt_ids = vt_ids[:, :seq_len]
                    vloss = loss_fn(vlogits.reshape(-1, vlogits.size(-1)), vt_ids.reshape(-1))
                    total_val_loss += vloss.item()
                val_steps = (len(val_pairs) // args.batch_size) + 1
                avg_val_loss = total_val_loss / val_steps
                try:
                    ppl = float(math.exp(avg_val_loss))
                except OverflowError:
                    ppl = float('inf')
                logger.info(f"Epoch {epoch+1} validation loss: {avg_val_loss:.4f}, ppl: {ppl:.2f}")
                epoch_metrics["val_loss"] = avg_val_loss
                epoch_metrics["val_ppl"] = ppl
            model.train()

        metrics["history"].append(epoch_metrics)

    # Save model
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    # Save metrics
    try:
        with open(Path(args.output) / "metrics.json", "w") as mf:
            json.dump(metrics, mf, indent=2)
    except Exception:
        logger.warning("Failed to save metrics.json")

    logger.info(f"✓ Model saved to {args.output}")

if __name__ == "__main__":
    main()
