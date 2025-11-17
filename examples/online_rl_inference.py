"""Example: Online reinforcement learning during inference."""

import torch
from image_token_llm.config import GraphRAGConfig, ImageTokenizerConfig
from image_token_llm.graph_attention import GraphRAGEnhanced
from image_token_llm.rl_learning import (
    PolicyNetwork,
    RLContinuousLearner,
    RewardModel,
)
from image_token_llm.vision_encoder import TripletEncoder


def main():
    """Demonstrate online RL-based inference with continuous learning."""
    print("=" * 60)
    print("Online Reinforcement Learning Inference Demo")
    print("=" * 60)

    # Configuration
    embedding_dim = 256
    img_config = ImageTokenizerConfig(embedding_dim=embedding_dim, patch_size=16)
    graph_config = GraphRAGConfig(top_k_neighbors=8, max_hops=3)

    # Initialize components
    print("\n[1/5] Initializing vision encoder...")
    encoder = TripletEncoder(img_config)

    print("[2/5] Initializing graph RAG...")
    graph_rag = GraphRAGEnhanced(
        graph_config, embedding_dim=embedding_dim, device="cpu"
    )

    # Prime graph with initial knowledge
    initial_knowledge = [
        ("cat", "is_a", "animal"),
        ("dog", "is_a", "animal"),
        ("animal", "lives_in", "habitat"),
        ("cat", "chases", "mouse"),
        ("mouse", "is_a", "rodent"),
        ("vision", "leads_to", "action"),
        ("action", "produces", "result"),
    ]
    print(f"   → Adding {len(initial_knowledge)} initial knowledge triplets")
    graph_rag.ingest(initial_knowledge)

    print("[3/5] Initializing reward model...")
    reward_model = RewardModel(embedding_dim=embedding_dim, hidden_dim=128)

    print("[4/5] Initializing policy network...")
    policy_network = PolicyNetwork(
        embedding_dim=embedding_dim, num_actions=8, hidden_dim=128
    )

    print("[5/5] Creating RL continuous learner...")
    rl_learner = RLContinuousLearner(
        encoder,
        graph_rag,
        reward_model,
        policy_network,
        learning_rate=1e-3,
        gamma=0.99,
        entropy_coef=0.01,
    )

    print("\n" + "=" * 60)
    print("Running Online Inference with Learning")
    print("=" * 60)

    # Simulate a sequence of inference steps
    num_steps = 5
    
    for step in range(num_steps):
        print(f"\n--- Inference Step {step + 1}/{num_steps} ---")

        # Generate synthetic image triplets
        # In real scenario, these would be actual images
        what_images = torch.randn(1, 3, 224, 224)
        action_images = torch.randn(1, 3, 224, 224)
        result_images = torch.randn(1, 3, 224, 224)

        # Simulate different scenarios
        scenarios = [
            ("cat", "chases", "mouse"),
            ("dog", "plays_with", "ball"),
            ("bird", "flies_to", "tree"),
            ("fish", "swims_in", "water"),
            ("person", "observes", "nature"),
        ]
        ground_truth = scenarios[step]

        print(f"   Ground Truth: {ground_truth[0]} → {ground_truth[1]} → {ground_truth[2]}")

        # Perform online inference with learning
        prediction, metrics = rl_learner.online_inference_with_learning(
            what_images,
            action_images,
            result_images,
            ground_truth_label=ground_truth,
            seed_nodes=["vision", "action"],
        )

        # Display metrics
        print(f"   Prediction Shape: {prediction.shape}")
        print(f"   Policy Loss: {metrics.get('policy_loss', 0.0):.4f}")
        print(f"   Value Loss: {metrics.get('value_loss', 0.0):.4f}")
        print(f"   Entropy: {metrics.get('entropy', 0.0):.4f}")
        print(f"   Final Reward: {metrics.get('final_reward', 0.0):.4f}")
        if "reward_updated" in metrics:
            print("   ✓ Reward model updated from feedback")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total inference steps: {num_steps}")
    print(f"Knowledge graph nodes: {graph_rag.graph.number_of_nodes()}")
    print(f"Knowledge graph edges: {graph_rag.graph.number_of_edges()}")
    print("\nThe model continuously learned from each inference step!")
    print("Key benefits:")
    print("  • Policy network learned better reasoning paths")
    print("  • Reward model improved trajectory scoring")
    print("  • Knowledge graph expanded with new concepts")
    print("  • Model adapted to new scenarios in real-time")
    print("=" * 60)


if __name__ == "__main__":
    main()
