#!/usr/bin/env python
"""Simple training script for testing."""

import torch
from image_token_llm.dreaming_model import DreamingReasoningLLM
from image_token_llm.config import ExperimentConfig, DreamingConfig

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model (NO RL to avoid issues)
    config = ExperimentConfig()
    config.dreaming = DreamingConfig(
        num_dream_sequences=2,
        dream_length=3,
        graph_reasoning_hops=2,
        output_mode="text"
    )
    
    model = DreamingReasoningLLM(
        config=config,
        device=device,
        embedding_dim=256,
        vocab_size=1024,
        enable_rl=False  # Disable RL
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {total_params:,} parameters")
    
    # Create optimizer for ALL model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    print("\nTraining...")
    for epoch in range(5):
        for batch_idx in range(10):
            # Create synthetic batch
            text_tokens = torch.randint(0, 1024, (4, 20), device=device)
            target_tokens = torch.randint(0, 1024, (4, 20), device=device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(text_tokens=text_tokens, output_mode="text")
            
            # Loss
            target_flat = target_tokens[:, 0].reshape(-1)
            loss = torch.nn.functional.cross_entropy(logits, target_flat)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}/5, Batch {batch_idx}/10, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} complete")
    
    # Save
    model.save("./models/simple_trained")
    print("\nModel saved to ./models/simple_trained")
    
    # Test generation
    print("\nTesting generation...")
    output = model.generate(prompt="Test", max_length=20)
    print(f"Generated: {output[:50]}...")
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
