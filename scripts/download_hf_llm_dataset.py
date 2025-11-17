#!/usr/bin/env python3
"""
Download a Hugging Face LLM dataset (OpenAssistant, ShareGPT, Dolly, etc) for distillation.

Usage:
    python scripts/download_hf_llm_dataset.py --dataset OpenAssistant/oasst1 --output ./data/openassistant
    python scripts/download_hf_llm_dataset.py --dataset databricks/databricks-dolly-15k --output ./data/dolly
    python scripts/download_hf_llm_dataset.py --dataset teknium/ShareGPT-Vicuna-unfiltered --output ./data/sharegpt

"""
import argparse
from pathlib import Path
from datasets import load_dataset
import shutil
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Hugging Face LLM dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Hugging Face dataset name (e.g. OpenAssistant/oasst1)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    args = parser.parse_args()

    print(f"Downloading {args.dataset} split={args.split} ...")
    ds = load_dataset(args.dataset, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.dataset.replace('/', '_')}_{args.split}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in ds:
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")
    print(f"✓ Saved {len(ds)} examples to {out_file}")
