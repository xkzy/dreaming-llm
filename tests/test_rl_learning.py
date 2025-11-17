"""Lightweight tests for RL-based continuous learning."""

import pytest
import torch

from image_token_llm.config import GraphRAGConfig, ImageTokenizerConfig
from image_token_llm.graph_attention import GraphRAGEnhanced
from image_token_llm.rl_learning import PolicyNetwork, RewardModel, RLContinuousLearner
from image_token_llm.vision_encoder import TripletEncoder


@pytest.fixture
def small_config():
    """Small config for fast testing (CLIP ViT-B/32: 512-dim)."""
    return {
        "embedding_dim": 512,
        "hidden_dim": 32,
        "num_actions": 4,
    }


class TestRewardModel:
    """Test reward model components."""

    def test_reward_model_forward(self, small_config):
        """Test reward prediction."""
        model = RewardModel(
            embedding_dim=small_config["embedding_dim"],
            hidden_dim=small_config["hidden_dim"],
        )

        # Create dummy embeddings
        what_emb = torch.randn(2, small_config["embedding_dim"])
        action_emb = torch.randn(2, small_config["embedding_dim"])
        result_emb = torch.randn(2, small_config["embedding_dim"])

        reward, confidence = model(what_emb, action_emb, result_emb)

        assert reward.shape == (2, 1)
        assert confidence.shape == (2, 1)
        assert (reward >= 0).all() and (reward <= 1).all()


class TestPolicyNetwork:
    """Test policy network components."""

    def test_policy_forward(self, small_config):
        """Test policy forward pass."""
        policy = PolicyNetwork(
            embedding_dim=small_config["embedding_dim"],
            num_actions=small_config["num_actions"],
            hidden_dim=small_config["hidden_dim"],
        )

        state_emb = torch.randn(2, small_config["embedding_dim"])
        neighbor_embs = torch.randn(
            2, small_config["num_actions"], small_config["embedding_dim"]
        )
        mask = torch.ones(2, small_config["num_actions"])

        action_probs, value = policy(state_emb, neighbor_embs, mask)

        assert action_probs.shape == (2, small_config["num_actions"])
        assert value.shape == (2, 1)
        assert torch.allclose(
            action_probs.sum(dim=-1), torch.ones(2), atol=1e-5
        )

    def test_policy_sample_action(self, small_config):
        """Test action sampling."""
        policy = PolicyNetwork(
            embedding_dim=small_config["embedding_dim"],
            num_actions=small_config["num_actions"],
            hidden_dim=small_config["hidden_dim"],
        )

        state_emb = torch.randn(1, small_config["embedding_dim"])
        neighbor_embs = torch.randn(
            1, small_config["num_actions"], small_config["embedding_dim"]
        )

        action, log_prob = policy.sample_action(state_emb, neighbor_embs)

        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert 0 <= action.item() < small_config["num_actions"]


class TestRLContinuousLearner:
    """Test RL-based continuous learner."""

    @pytest.fixture
    def rl_learner(self, small_config):
        """Create RL learner for testing."""
        img_config = ImageTokenizerConfig(
            embedding_dim=small_config["embedding_dim"], patch_size=16
        )
        graph_config = GraphRAGConfig(top_k_neighbors=3, max_hops=1)

        encoder = TripletEncoder(img_config, backbone="clip")
        graph = GraphRAGEnhanced(
    graph_config,
    embedding_dim=small_config["embedding_dim"],
    device="cpu",
        )
        reward_model = RewardModel(
            embedding_dim=small_config["embedding_dim"],
            hidden_dim=small_config["hidden_dim"],
        )
        policy_network = PolicyNetwork(
            embedding_dim=small_config["embedding_dim"],
            num_actions=small_config["num_actions"],
            hidden_dim=small_config["hidden_dim"],
        )

        learner = RLContinuousLearner(
            encoder, graph, reward_model, policy_network, learning_rate=1e-3
        )

        # Prime graph with some nodes
        graph.ingest(
            [
                ("vision", "leads_to", "action"),
                ("action", "produces", "perception"),
            ]
        )

        return learner

    def test_simulate_reasoning(self, rl_learner):
        """Test reasoning trajectory simulation."""
        # Use 224x224 images to match encoder
        what = torch.randn(1, 3, 224, 224)
        action = torch.randn(1, 3, 224, 224)
        result = torch.randn(1, 3, 224, 224)

        trajectory, final_reward = rl_learner.simulate_reasoning(
            what, action, result, seed_nodes=["vision"], max_steps=2
        )

        # May be empty if no neighbors found
        assert isinstance(trajectory, list)
        assert isinstance(final_reward, torch.Tensor)

    def test_learn_from_trajectory(self, rl_learner, small_config):
        """Test learning from trajectory."""
        # Create minimal trajectory
        emb_dim = small_config["embedding_dim"]
        num_actions = small_config["num_actions"]
        trajectory = [
            {
                "state": torch.randn(1, emb_dim),
                "action": torch.tensor([0]),
                "log_prob": torch.tensor([-0.5]),
                "reward": torch.tensor([[0.8]]),
                "confidence": torch.tensor([[0.9]]),
                "neighbors": torch.randn(1, num_actions, emb_dim),
                "mask": torch.ones(1, num_actions),
            }
        ]
        final_reward = torch.tensor(0.8)

        metrics = rl_learner.learn_from_trajectory(trajectory, final_reward)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "final_reward" in metrics

    def test_online_inference_minimal(self, rl_learner):
        """Test online inference with learning (minimal)."""
        what = torch.randn(1, 3, 224, 224)
        action = torch.randn(1, 3, 224, 224)
        result = torch.randn(1, 3, 224, 224)

        prediction, metrics = rl_learner.online_inference_with_learning(
            what, action, result,
            ground_truth_label=("test", "action", "result"),
            seed_nodes=["vision"]
        )

        assert prediction.shape[0] == 1
        assert isinstance(metrics, dict)
