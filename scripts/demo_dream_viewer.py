#!/usr/bin/env python3
"""Demo: Watch what the model is dreaming in real-time."""

from __future__ import annotations

import argparse

import torch

from image_token_llm.config import DreamingConfig, ExperimentConfig
from image_token_llm.dreaming_model import DreamingReasoningLLM


def main():
    parser = argparse.ArgumentParser(
        description="Watch model dreams in real-time"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What happens when you drop a ball?",
        help="Prompt for generation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to pretrained model (optional)"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch dreams in real-time during generation"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export dreams to file (json or txt)"
    )
    parser.add_argument(
        "--visualize",
        type=str,
        default=None,
        help="Save dream evolution heatmap to this path"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔮 DREAM VIEWER DEMO")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Device: {args.device}")
    print(f"Watch mode: {args.watch}")
    print()
    
    # Create model with dream viewer enabled
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=4,
        dream_length=5,
        graph_reasoning_hops=3,
        output_mode="text"
    )
    
    print("📦 Initializing model with dream viewer...")
    model = DreamingReasoningLLM(
        config=config,
        device=args.device,
        embedding_dim=256,
        vocab_size=1024,
        enable_rl=False,
        enable_dream_viewer=True  # Enable dream viewing!
    )
    
    # Load pretrained weights if provided
    if args.model_path:
        print(f"📥 Loading model from {args.model_path}...")
        model = DreamingReasoningLLM.load_pretrained(
            args.model_path,
            device=args.device
        )
        # Re-enable dream viewer
        if model.dream_viewer is None:
            from image_token_llm.dream_viewer import DreamViewer
            model.dream_viewer = DreamViewer(enable_recording=True)
            model.enable_dream_viewer = True
    
    print()
    print("🎯 Generating with dream visualization...")
    print("-" * 60)
    print()
    
    # Generate with dream watching
    result = model.generate(
        prompt=args.prompt,
        max_length=50,
        temperature=0.8,
        return_dreams=True,
        watch_dreams=args.watch  # Live dream streaming!
    )
    
    print()
    print("=" * 60)
    print("📝 GENERATION RESULT")
    print("=" * 60)
    print(f"Output: {result['output'][:200]}...")
    print(f"Dreams: {len(result['dreams'])} sequences")
    print(f"Graph: {result['graph_data']['num_nodes']} nodes")
    print()
    
    # Export dreams if requested
    if args.export:
        ext = args.export.split(".")[-1]
        model.export_recorded_dreams(
            args.export,
            format=ext if ext in ["json", "txt"] else "json"
        )
    
    # Visualize dream evolution if requested
    if args.visualize:
        print("📊 Creating dream evolution heatmap...")
        model.visualize_dream_evolution(save_path=args.visualize)
    
    # Get dream summary
    if model.dream_viewer:
        summary = model.dream_viewer.get_dream_summary()
        print("=" * 60)
        print("📊 DREAM SUMMARY")
        print("=" * 60)
        print(f"Total dreams: {summary['total_dreams']}")
        print(f"Total steps:  {summary['total_steps']}")
        if summary['total_dreams'] > 0:
            print(f"Avg mean:     {summary['avg_mean']:.4f}")
            print(f"Avg std:      {summary['avg_std']:.4f}")
            print(f"Mean range:   {summary['mean_range']}")
            print(f"Std range:    {summary['std_range']}")
        print()
    
    print("=" * 60)
    print("✨ Demo Complete!")
    print("=" * 60)
    print()
    print("Try these commands:")
    print("  # Watch dreams live:")
    print(f"  python {__file__} --watch")
    print()
    print("  # Export dreams to JSON:")
    print(f"  python {__file__} --export dreams.json")
    print()
    print("  # Create visualization:")
    print(f"  python {__file__} --visualize dreams.png")
    print()
    print("  # Use pretrained model:")
    print(f"  python {__file__} --model-path ./models/distilled_large_scale")
    print()


if __name__ == "__main__":
    main()
