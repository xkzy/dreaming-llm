"""Example usage of the Dreaming-based Reasoning LLM."""

import torch
from image_token_llm.dreaming_model import DreamingReasoningLLM
from image_token_llm.config import ExperimentConfig, DreamingConfig

def example_text_to_text():
    """Example: Text prompt → dreaming reasoning → text output."""
    print("\n" + "="*60)
    print("Example 1: Text → Dreaming → Text")
    print("="*60)
    
    # Create model
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=4,
        dream_length=5,
        graph_reasoning_hops=3,
        output_mode="text",
        enable_visualization=True
    )
    
    model = DreamingReasoningLLM(
        config=config,
        device="cpu",
        embedding_dim=512,
        vocab_size=4096
    )
    
    # Generate response
    prompt = "What happens when you open a door?"
    print(f"\nPrompt: {prompt}")
    
    result = model.generate(
        prompt=prompt,
        max_length=50,
        temperature=0.8,
        output_mode="text",
        return_dreams=True
    )
    
    print(f"\nOutput: {result['output']}")
    print(f"\nDreaming Process:")
    print(f"  - Generated {len(result['dreams'])} dream sequences")
    print(f"  - Each with {len(result['dreams'][0])} reasoning steps")
    
    graph_data = result['graph_data']
    print(f"\nGraph Reasoning:")
    print(f"  - Total nodes: {graph_data['num_nodes']}")
    print(f"  - Total edges: {graph_data['num_edges']}")
    print(f"  - Temporal edges: {len(graph_data['temporal_edges'])}")
    print(f"  - Causal edges: {len(graph_data['causal_edges'])}")


def example_image_to_text():
    """Example: Images → dreaming reasoning → text description."""
    print("\n" + "="*60)
    print("Example 2: Images → Dreaming → Text")
    print("="*60)
    
    # Create model
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=3,
        dream_length=4,
        output_mode="text"
    )
    
    model = DreamingReasoningLLM(config=config, device="cpu")
    
    # Simulate input images (B=1, N=3 images, C=3, H=W=224)
    images = torch.randn(1, 3, 3, 224, 224)
    
    print(f"\nInput: {images.shape[1]} images")
    
    result = model.generate(
        images=images,
        max_length=50,
        output_mode="text",
        return_dreams=True
    )
    
    print(f"\nGenerated Description: {result['output']}")
    print(f"\nDreaming Analysis:")
    print(f"  - Explored {len(result['dreams'])} reasoning paths")
    print(f"  - Each path: {len(result['dreams'][0])} steps")


def example_text_to_image():
    """Example: Text prompt → dreaming → image triplet output."""
    print("\n" + "="*60)
    print("Example 3: Text → Dreaming → Image Triplets")
    print("="*60)
    
    # Create model configured for image output
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=4,
        dream_length=5,
        output_mode="image"
    )
    
    model = DreamingReasoningLLM(config=config, device="cpu")
    
    prompt = "A person walking through a forest"
    print(f"\nPrompt: {prompt}")
    
    result = model.generate(
        prompt=prompt,
        output_mode="image",
        return_dreams=True
    )
    
    what, action, result_img = result['output']
    print(f"\nGenerated Image Triplet:")
    print(f"  - what (scene): {what.shape}")
    print(f"  - action (transformation): {action.shape}")
    print(f"  - result (outcome): {result_img.shape}")
    
    print(f"\nDream sequences explored {len(result['dreams'])} paths")


def example_mixed_output():
    """Example: Text → dreaming → both text and images."""
    print("\n" + "="*60)
    print("Example 4: Text → Dreaming → Text + Images")
    print("="*60)
    
    # Create model
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=3,
        dream_length=4,
        output_mode="both"
    )
    
    model = DreamingReasoningLLM(config=config, device="cpu")
    
    prompt = "Describe and visualize rain"
    print(f"\nPrompt: {prompt}")
    
    result = model.generate(
        prompt=prompt,
        output_mode="both",
        return_dreams=True
    )
    
    print(f"\nText Output: {result['text']}")
    
    what, action, result_img = result['image']
    print(f"\nImage Triplet Output:")
    print(f"  - what: {what.shape}")
    print(f"  - action: {action.shape}")
    print(f"  - result: {result_img.shape}")


def example_visualize_thinking():
    """Example: Visualize the thinking/dreaming process."""
    print("\n" + "="*60)
    print("Example 5: Visualizing the Thinking Process")
    print("="*60)
    
    # Create model
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=4,
        dream_length=5,
        enable_visualization=True
    )
    
    model = DreamingReasoningLLM(config=config, device="cpu")
    
    prompt = "What causes a river to flow?"
    print(f"\nPrompt: {prompt}")
    
    # Get thinking visualization
    viz_data = model.visualize_thinking(prompt=prompt)
    
    print(f"\nThinking Process Visualization:")
    print(f"  - Dream sequences: {len(viz_data['dreams'])}")
    print(f"  - Steps per dream: {len(viz_data['dreams'][0])}")
    
    graph_data = viz_data['graph_data']
    print(f"\n  Graph Structure:")
    print(f"    - Nodes (triplets): {graph_data['num_nodes']}")
    print(f"    - Edges (connections): {graph_data['num_edges']}")
    print(f"    - Dreams: {graph_data['num_dreams']}")
    print(f"    - Length: {graph_data['dream_length']}")
    
    print(f"\n  Output: {viz_data['output']}")


def example_save_and_load():
    """Example: Save and load a trained model."""
    print("\n" + "="*60)
    print("Example 6: Save and Load Model")
    print("="*60)
    
    # Create and configure model
    config = ExperimentConfig()
    model = DreamingReasoningLLM(config=config, device="cpu")
    
    # Save model
    save_path = "./saved_dreaming_model"
    model.save_pretrained(save_path)
    print(f"\nModel saved to: {save_path}")
    
    # Load model
    loaded_model = DreamingReasoningLLM.load_pretrained(
        save_path,
        device="cpu"
    )
    print(f"Model loaded from: {save_path}")
    
    # Test loaded model
    result = loaded_model.generate(
        prompt="Test prompt",
        max_length=20,
        output_mode="text"
    )
    print(f"\nGenerated with loaded model: {result}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" "*15 + "Dreaming-Based Reasoning LLM Examples")
    print("="*70)
    
    examples = [
        ("Text → Dreaming → Text", example_text_to_text),
        ("Images → Dreaming → Text", example_image_to_text),
        ("Text → Dreaming → Images", example_text_to_image),
        ("Mixed Output (Text + Images)", example_mixed_output),
        ("Visualize Thinking Process", example_visualize_thinking),
        ("Save and Load Model", example_save_and_load),
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()
