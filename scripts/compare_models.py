#!/usr/bin/env python3
"""
Compare multiple pretrained model bundles side-by-side.

Usage:
    python scripts/compare_models.py model1/ model2/ --prompts 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image_token_llm.model import ImageTokenReasoningLLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


TEST_PROMPTS = [
    "What is artificial intelligence?",
    "Explain photosynthesis in simple terms",
    "How do computers process information?",
    "What causes seasons on Earth?",
    "Describe the water cycle",
    "What is DNA?",
    "How does the internet work?",
    "Explain gravity",
    "What are the primary colors?",
    "How do vaccines work?",
]


def load_model_info(bundle_dir: str) -> Dict[str, Any]:
    """Load model configuration and metadata."""
    bundle_path = Path(bundle_dir)
    
    info = {
        "bundle_dir": str(bundle_path),
        "bundle_name": bundle_path.name,
    }
    
    # Load config if available
    config_file = bundle_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
            info["vocab_size"] = config.get("tokenizer", {}).get(
                "vocab_size", "unknown"
            )
            info["max_seq_len"] = config.get("text_decoder", {}).get(
                "max_seq_len", "unknown"
            )
    
    # Check if weights file exists
    weights_files = list(bundle_path.glob("*_weights.pt"))
    if weights_files:
        weights_file = weights_files[0]
        size_mb = weights_file.stat().st_size / (1024 * 1024)
        info["weights_size_mb"] = f"{size_mb:.2f}"
    
    return info


def compare_models(
    bundle_dirs: List[str],
    num_prompts: int = 5,
    device: str = "cpu",
) -> None:
    """Compare multiple models on test prompts."""
    
    logger.info("=" * 70)
    logger.info("Model Comparison Tool")
    logger.info("=" * 70)
    
    # Load all models
    models = []
    model_infos = []
    
    for bundle_dir in bundle_dirs:
        logger.info(f"Loading model from {bundle_dir}...")
        
        try:
            model = ImageTokenReasoningLLM.load_from_bundle(
                bundle_dir=bundle_dir,
                device=device,
                enable_rl=False,
            )
            info = load_model_info(bundle_dir)
            
            models.append(model)
            model_infos.append(info)
            
            logger.info(f"  ✓ Loaded: {info['bundle_name']}")
            logger.info(f"    Vocab size: {info.get('vocab_size', 'N/A')}")
            logger.info(f"    Max seq len: {info.get('max_seq_len', 'N/A')}")
            logger.info(
                f"    Weights: {info.get('weights_size_mb', 'N/A')} MB"
            )
            
        except Exception as e:
            logger.error(f"  ✗ Failed to load {bundle_dir}: {e}")
            sys.exit(1)
    
    # Select test prompts
    test_prompts = TEST_PROMPTS[:num_prompts]
    
    logger.info("=" * 70)
    logger.info(f"Testing on {len(test_prompts)} prompts")
    logger.info("=" * 70)
    
    # Compare on each prompt
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\nPrompt {i}/{len(test_prompts)}: {prompt}")
        logger.info("-" * 70)
        
        prompt_results = {"prompt": prompt, "outputs": []}
        
        for model, info in zip(models, model_infos):
            try:
                output = model.generate(
                    prompt=prompt,
                    max_new_tokens=100,
                    temperature=0.7,
                    stream=False,
                )
                
                # Truncate for display
                display_output = output[:150]
                if len(output) > 150:
                    display_output += "..."
                
                logger.info(f"[{info['bundle_name']}]")
                logger.info(f"  {display_output}")
                
                prompt_results["outputs"].append({
                    "model": info["bundle_name"],
                    "output": output,
                    "length": len(output),
                })
                
            except Exception as e:
                logger.error(f"  Error generating: {e}")
                prompt_results["outputs"].append({
                    "model": info["bundle_name"],
                    "output": f"ERROR: {e}",
                    "length": 0,
                })
        
        results.append(prompt_results)
    
    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    
    for info in model_infos:
        model_name = info["bundle_name"]
        
        # Calculate average output length
        lengths = [
            r["length"]
            for result in results
            for r in result["outputs"]
            if r["model"] == model_name and r["length"] > 0
        ]
        
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        
        logger.info(f"\n{model_name}:")
        logger.info(f"  Average output length: {avg_length:.1f} chars")
        logger.info(f"  Successful generations: {len(lengths)}/{num_prompts}")
        logger.info(
            f"  Bundle size: {info.get('weights_size_mb', 'N/A')} MB"
        )
    
    # Save detailed results
    output_file = Path("model_comparison_results.json")
    with open(output_file, "w") as f:
        json.dump(
            {
                "models": model_infos,
                "prompts": results,
            },
            f,
            indent=2,
        )
    
    logger.info(f"\nDetailed results saved to: {output_file}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple pretrained model bundles"
    )
    parser.add_argument(
        "bundles",
        nargs="+",
        help="Paths to model bundle directories to compare",
    )
    parser.add_argument(
        "--prompts",
        type=int,
        default=5,
        help="Number of test prompts to use (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use: 'cpu' or 'cuda' (default: cpu)",
    )
    
    args = parser.parse_args()
    
    if len(args.bundles) < 2:
        parser.error("At least 2 model bundles are required for comparison")
    
    # Validate bundle directories
    for bundle_dir in args.bundles:
        if not Path(bundle_dir).exists():
            parser.error(f"Bundle directory not found: {bundle_dir}")
    
    compare_models(
        bundle_dirs=args.bundles,
        num_prompts=args.prompts,
        device=args.device,
    )


if __name__ == "__main__":
    main()
