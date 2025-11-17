#!/usr/bin/env python3
"""Quick demo of the Dreaming-Based Reasoning LLM."""

import torch
from image_token_llm.dreaming_model import DreamingReasoningLLM
from image_token_llm.config import ExperimentConfig, DreamingConfig


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def demo_basic_generation():
    """Demo 1: Basic text generation."""
    print_header("Demo 1: Basic Text Generation")
    
    # Create model
    print("Creating model...")
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=4,
        dream_length=5,
        output_mode="text"
    )
    
    model = DreamingReasoningLLM(config=config, device="cpu")
    print("✓ Model created")
    
    # Generate response
    prompt = "What happens when you drop a ball?"
    print(f"\nPrompt: {prompt}")
    print("\nThinking... (generating 4 dreams × 5 steps = 20 reasoning nodes)")
    
    output = model.generate(
        prompt=prompt,
        max_length=50,
        temperature=0.8,
        output_mode="text"
    )
    
    print(f"\nOutput: {output}")
    print(f"\nMetadata: {model.last_metadata}")


def demo_with_visualization():
    """Demo 2: Generation with dream visualization."""
    print_header("Demo 2: Visualizing the Thinking Process")
    
    model = DreamingReasoningLLM(device="cpu")
    
    prompt = "How does a plant grow?"
    print(f"Prompt: {prompt}")
    print("\nGenerating with dream visualization...")
    
    result = model.generate(
        prompt=prompt,
        max_length=30,
        return_dreams=True
    )
    
    print(f"\nOutput: {result['output']}")
    
    print(f"\n📊 Thinking Process Analysis:")
    print(f"  Dreams explored: {len(result['dreams'])} parallel paths")
    print(f"  Steps per dream: {len(result['dreams'][0])} reasoning steps")
    
    graph_data = result['graph_data']
    print(f"\n  Graph Structure:")
    print(f"    Nodes (triplets): {graph_data['num_nodes']}")
    print(f"    Edges (connections): {graph_data['num_edges']}")
    print(f"    Temporal edges: {len(graph_data['temporal_edges'])}")
    print(f"    Causal edges: {len(graph_data['causal_edges'])}")
    
    print(f"\n  Reasoning embedding shape: {result['reasoning_embedding'].shape}")


def demo_image_output():
    """Demo 3: Image generation."""
    print_header("Demo 3: Generating Image Triplets")
    
    config = ExperimentConfig()
    config.dreaming.output_mode = "image"
    
    model = DreamingReasoningLLM(config=config, device="cpu")
    
    prompt = "A bird flying over water"
    print(f"Prompt: {prompt}")
    print("\nGenerating image triplets...")
    
    what, action, result = model.generate(
        prompt=prompt,
        output_mode="image"
    )
    
    print(f"\n✓ Generated Image Triplet:")
    print(f"  what (scene state):   {what.shape}")
    print(f"  action (transformation): {action.shape}")
    print(f"  result (outcome):     {result.shape}")
    
    print(f"\nThese embeddings can be decoded to actual images")
    print(f"using a diffusion model or image generator.")


def demo_mixed_output():
    """Demo 4: Both text and images."""
    print_header("Demo 4: Mixed Output (Text + Images)")
    
    config = ExperimentConfig()
    config.dreaming.output_mode = "both"
    
    model = DreamingReasoningLLM(config=config, device="cpu")
    
    prompt = "Describe and visualize rain"
    print(f"Prompt: {prompt}")
    print("\nGenerating both text description and visual representation...")
    
    result = model.generate(
        prompt=prompt,
        output_mode="both"
    )
    
    print(f"\n📝 Text Output:")
    print(f"  {result['text']}")
    
    what, action, result_img = result['image']
    print(f"\n🖼️  Image Triplet:")
    print(f"  what:   {what.shape}")
    print(f"  action: {action.shape}")
    print(f"  result: {result_img.shape}")


def demo_architecture_info():
    """Demo 5: Show architecture information."""
    print_header("Demo 5: Architecture Information")
    
    model = DreamingReasoningLLM(device="cpu")
    
    print("Architecture Components:")
    print(f"  ✓ InputTokenizer:    {model.input_tokenizer.__class__.__name__}")
    print(f"  ✓ DreamGenerator:    {model.dream_generator.__class__.__name__}")
    print(f"  ✓ GraphReasoner:     {model.graph_reasoner.__class__.__name__}")
    print(f"  ✓ OutputDecoder:     {model.output_decoder.__class__.__name__}")
    
    print(f"\nConfiguration:")
    print(f"  Embedding dimension: {model.embedding_dim}")
    print(f"  Vocabulary size:     {model.vocab_size}")
    print(f"  Device:              {model.device}")
    
    config = model.config.dreaming
    print(f"\nDreaming Settings:")
    print(f"  Dream sequences:     {config.num_dream_sequences}")
    print(f"  Dream length:        {config.dream_length}")
    print(f"  Reasoning hops:      {config.graph_reasoning_hops}")
    print(f"  Output mode:         {config.output_mode}")
    print(f"  Visualization:       {config.enable_visualization}")
    
    print(f"\nTotal Parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {total_params:,} parameters")


def main():
    """Run all demos."""
    print("\n" + "🌟"*35)
    print("     DREAMING-BASED REASONING LLM - INTERACTIVE DEMO")
    print("🌟"*35)
    
    print("\nThis demo showcases the new architecture where:")
    print("  1. All inputs (text/images) → image triplets (what, action, result)")
    print("  2. Thinking happens via parallel 'dream' sequences")
    print("  3. Dreams are connected via graph reasoning")
    print("  4. Output can be text, images, or both")
    
    demos = [
        ("Basic Text Generation", demo_basic_generation),
        ("Visualizing Thinking Process", demo_with_visualization),
        ("Image Generation", demo_image_output),
        ("Mixed Output", demo_mixed_output),
        ("Architecture Info", demo_architecture_info),
    ]
    
    print("\n" + "-"*70)
    input("Press Enter to start the demos...")
    
    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
            
            if i < len(demos):
                print("\n" + "-"*70)
                input(f"Press Enter to continue to Demo {i+1}...")
        except Exception as e:
            print(f"\n❌ Error in demo: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "🌟"*35)
    print("     ALL DEMOS COMPLETED!")
    print("🌟"*35)
    
    print("\n📚 For more information:")
    print("  - Architecture diagram: docs/dreaming_architecture.svg")
    print("  - Detailed guide: docs/DREAMING_ARCHITECTURE.md")
    print("  - Quick start: DREAMING_README.md")
    print("  - Examples: examples/dreaming_examples.py")
    print("  - Tests: tests/test_dreaming_model.py")


if __name__ == "__main__":
    main()
