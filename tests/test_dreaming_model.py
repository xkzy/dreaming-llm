"""Tests for the Dreaming-based Reasoning LLM."""

import pytest
import torch

from image_token_llm.config import DreamingConfig, ExperimentConfig
from image_token_llm.dreaming import (
    DreamGenerator,
    DreamSequence,
    InputTokenizer,
    OutputDecoder,
)
from image_token_llm.dream_graph_reasoner import DreamGraphReasoner
from image_token_llm.dreaming_model import DreamingReasoningLLM


@pytest.fixture
def dreaming_config():
    """Create test configuration."""
    return DreamingConfig(
        num_dream_sequences=2,
        dream_length=3,
        graph_reasoning_hops=2,
        output_mode="text",
        enable_visualization=True
    )


@pytest.fixture
def embedding_dim():
    """Standard embedding dimension."""
    return 512


class TestInputTokenizer:
    """Tests for InputTokenizer."""
    
    def test_text_to_triplet(self, dreaming_config, embedding_dim):
        """Test text → image triplet conversion."""
        tokenizer = InputTokenizer(dreaming_config, embedding_dim)
        
        # Create dummy text embedding
        B = 2
        text_emb = torch.randn(B, embedding_dim)
        
        # Convert to triplet
        what, action, result = tokenizer(text_embedding=text_emb)
        
        assert what.shape == (B, embedding_dim)
        assert action.shape == (B, embedding_dim)
        assert result.shape == (B, embedding_dim)
    
    def test_image_to_triplet(self, dreaming_config, embedding_dim):
        """Test images → image triplet conversion."""
        tokenizer = InputTokenizer(dreaming_config, embedding_dim)
        
        # Create dummy image embeddings (B=2, N=3 images)
        B, N = 2, 3
        image_embs = torch.randn(B, N, embedding_dim)
        
        # Convert to triplet
        what, action, result = tokenizer(image_embeddings=image_embs)
        
        assert what.shape == (B, embedding_dim)
        assert action.shape == (B, embedding_dim)
        assert result.shape == (B, embedding_dim)
    
    def test_requires_input(self, dreaming_config, embedding_dim):
        """Test that tokenizer requires either text or images."""
        tokenizer = InputTokenizer(dreaming_config, embedding_dim)
        
        with pytest.raises(ValueError, match="Must provide"):
            tokenizer()


class TestDreamSequence:
    """Tests for DreamSequence."""
    
    def test_dream_generation(self, embedding_dim):
        """Test generating a dream sequence."""
        dream_seq = DreamSequence(embedding_dim)
        
        # Create initial triplet
        B = 2
        initial = torch.randn(B, 3, embedding_dim)
        
        # Generate dream
        num_steps = 4
        sequence = dream_seq(initial, num_steps=num_steps)
        
        assert len(sequence) == num_steps
        for what, action, result in sequence:
            assert what.shape == (B, embedding_dim)
            assert action.shape == (B, embedding_dim)
            assert result.shape == (B, embedding_dim)


class TestDreamGenerator:
    """Tests for DreamGenerator."""
    
    def test_multiple_dreams(self, dreaming_config, embedding_dim):
        """Test generating multiple dream sequences."""
        generator = DreamGenerator(dreaming_config, embedding_dim)
        
        B = 2
        what = torch.randn(B, embedding_dim)
        action = torch.randn(B, embedding_dim)
        result = torch.randn(B, embedding_dim)
        
        # Generate dreams
        all_dreams = generator(what, action, result)
        
        assert len(all_dreams) == dreaming_config.num_dream_sequences
        for dream in all_dreams:
            assert len(dream) == dreaming_config.dream_length
            for triplet in dream:
                assert len(triplet) == 3  # (what, action, result)


class TestDreamGraphReasoner:
    """Tests for DreamGraphReasoner."""
    
    def test_graph_reasoning(self, dreaming_config, embedding_dim):
        """Test graph-based reasoning over dreams."""
        reasoner = DreamGraphReasoner(dreaming_config, embedding_dim)
        
        # Create mock dream sequences
        B = 2
        num_dreams = 2
        dream_length = 3
        
        dreams = []
        for _ in range(num_dreams):
            dream = []
            for _ in range(dream_length):
                what = torch.randn(B, embedding_dim)
                action = torch.randn(B, embedding_dim)
                result = torch.randn(B, embedding_dim)
                dream.append((what, action, result))
            dreams.append(dream)
        
        # Apply reasoning
        reasoning_emb = reasoner(dreams)
        
        assert reasoning_emb.shape == (B, embedding_dim)
    
    def test_graph_visualization_data(self, dreaming_config, embedding_dim):
        """Test extracting graph data for visualization."""
        reasoner = DreamGraphReasoner(dreaming_config, embedding_dim)
        
        # Create mock dreams
        B = 1
        dreams = []
        for _ in range(2):
            dream = []
            for _ in range(3):
                triplet = (
                    torch.randn(B, embedding_dim),
                    torch.randn(B, embedding_dim),
                    torch.randn(B, embedding_dim)
                )
                dream.append(triplet)
            dreams.append(dream)
        
        # Get visualization data
        viz_data = reasoner.get_graph_visualization_data(dreams)
        
        assert "num_nodes" in viz_data
        assert "num_edges" in viz_data
        assert "temporal_edges" in viz_data
        assert "causal_edges" in viz_data
        assert viz_data["num_nodes"] == 6  # 2 dreams × 3 steps


class TestOutputDecoder:
    """Tests for OutputDecoder."""
    
    def test_text_mode(self, dreaming_config, embedding_dim):
        """Test text output decoding."""
        decoder = OutputDecoder(dreaming_config, embedding_dim)
        
        B = 2
        reasoning = torch.randn(B, embedding_dim)
        
        logits = decoder(reasoning, mode="text")
        
        assert logits.shape == (B, decoder.vocab_size)
    
    def test_image_mode(self, dreaming_config, embedding_dim):
        """Test image output decoding."""
        decoder = OutputDecoder(dreaming_config, embedding_dim)
        
        B = 2
        reasoning = torch.randn(B, embedding_dim)
        
        what, action, result = decoder(reasoning, mode="image")
        
        assert what.shape == (B, embedding_dim)
        assert action.shape == (B, embedding_dim)
        assert result.shape == (B, embedding_dim)
    
    def test_both_mode(self, dreaming_config, embedding_dim):
        """Test mixed text+image output."""
        decoder = OutputDecoder(dreaming_config, embedding_dim)
        
        B = 2
        reasoning = torch.randn(B, embedding_dim)
        
        output = decoder(reasoning, mode="both")
        
        assert isinstance(output, dict)
        assert "text" in output
        assert "image" in output
        assert output["text"].shape == (B, decoder.vocab_size)
    
    def test_generate_sequence(self, dreaming_config, embedding_dim):
        """Test autoregressive text generation."""
        decoder = OutputDecoder(dreaming_config, embedding_dim)
        
        B = 2
        reasoning = torch.randn(B, embedding_dim)
        
        tokens = decoder.generate_sequence(
            reasoning,
            max_length=10,
            temperature=1.0
        )
        
        assert tokens.shape == (B, 10)


class TestDreamingReasoningLLM:
    """Tests for the full DreamingReasoningLLM model."""
    
    def test_model_initialization(self):
        """Test model creates all components."""
        config = ExperimentConfig()
        model = DreamingReasoningLLM(config=config, device="cpu")
        
        assert hasattr(model, "input_tokenizer")
        assert hasattr(model, "dream_generator")
        assert hasattr(model, "graph_reasoner")
        assert hasattr(model, "output_decoder")
    
    def test_model_with_rl(self):
        """Test model initializes with RL components."""
        config = ExperimentConfig()
        model = DreamingReasoningLLM(
            config=config,
            device="cpu",
            enable_rl=True
        )
        
        assert hasattr(model, "policy_network")
        assert hasattr(model, "reward_model")
        assert hasattr(model, "policy_optimizer")
        assert hasattr(model, "reward_optimizer")
        assert hasattr(model, "trajectory_buffer")
        assert model.enable_rl is True
    
    def test_model_without_rl(self):
        """Test model works without RL."""
        model = DreamingReasoningLLM(device="cpu", enable_rl=False)
        
        assert not hasattr(model, "policy_network")
        assert model.enable_rl is False
    
    def test_forward_text_input(self):
        """Test forward pass with text input."""
        model = DreamingReasoningLLM(device="cpu")
        
        B, seq_len = 2, 10
        text_tokens = torch.randint(0, 100, (B, seq_len))
        
        output = model(text_tokens=text_tokens, output_mode="text")
        
        assert output.shape == (B, model.vocab_size)
    
    def test_forward_image_input(self):
        """Test forward pass with image input."""
        model = DreamingReasoningLLM(device="cpu")
        
        B, N = 2, 3
        image_embs = torch.randn(B, N, model.embedding_dim)
        
        output = model(image_embeddings=image_embs, output_mode="text")
        
        assert output.shape == (B, model.vocab_size)
    
    def test_generate_text(self):
        """Test text generation."""
        model = DreamingReasoningLLM(device="cpu")
        
        output = model.generate(
            prompt="Test prompt",
            max_length=20,
            output_mode="text"
        )
        
        assert isinstance(output, str)
    
    def test_generate_with_dreams(self):
        """Test generation with dream visualization."""
        model = DreamingReasoningLLM(device="cpu")
        
        result = model.generate(
            prompt="Test",
            return_dreams=True,
            output_mode="text"
        )
        
        assert isinstance(result, dict)
        assert "output" in result
        assert "dreams" in result
        assert "graph_data" in result
    
    def test_visualize_thinking(self):
        """Test thinking visualization."""
        model = DreamingReasoningLLM(device="cpu")
        
        viz_data = model.visualize_thinking(prompt="Test")
        
        assert "dreams" in viz_data
        assert "graph_data" in viz_data
        assert "output" in viz_data
    
    def test_save_and_load(self, tmp_path):
        """Test model save and load."""
        model = DreamingReasoningLLM(device="cpu")
        
        # Save
        save_path = tmp_path / "test_model"
        model.save_pretrained(str(save_path))
        
        assert (save_path / "dreaming_model_weights.pt").exists()
        assert (save_path / "config.json").exists()
        
        # Load
        loaded = DreamingReasoningLLM.load_pretrained(
            str(save_path),
            device="cpu"
        )
        
        assert isinstance(loaded, DreamingReasoningLLM)


class TestRLIntegration:
    """Tests for RL integration."""
    
    def test_rl_training_step(self):
        """Test single RL training step."""
        model = DreamingReasoningLLM(device="cpu", enable_rl=True)
        
        metrics = model.train_step_rl(
            prompt="Test prompt",
            reward_signal=1.0
        )
        
        assert "reward" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert metrics["reward"] == 1.0
    
    def test_learn_from_feedback(self):
        """Test continuous learning from feedback."""
        model = DreamingReasoningLLM(device="cpu", enable_rl=True)
        
        examples = [
            {"prompt": "Test 1", "reward": 0.8},
            {"prompt": "Test 2", "reward": 0.9},
        ]
        
        metrics = model.learn_from_feedback(examples, num_epochs=2)
        
        assert "avg_reward" in metrics
        assert "avg_policy_loss" in metrics
        assert "total_reward" in metrics


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_text_to_text_pipeline(self):
        """Test complete text → dreaming → text pipeline."""
        config = ExperimentConfig()
        config.dreaming = DreamingConfig(
            num_dream_sequences=2,
            dream_length=3,
            output_mode="text"
        )
        
        model = DreamingReasoningLLM(
            config=config, device="cpu", enable_rl=False
        )
        
        output = model.generate(
            prompt="What is AI?",
            max_length=30,
            return_dreams=True
        )
        
        assert isinstance(output["output"], str)
        assert len(output["dreams"]) == 2
        assert len(output["dreams"][0]) == 3
    
    def test_image_to_text_pipeline(self):
        """Test image → dreaming → text pipeline."""
        model = DreamingReasoningLLM(device="cpu")
        
        # Simulated images
        images = torch.randn(1, 2, 3, 224, 224)
        
        output = model.generate(images=images, output_mode="text")
        
        assert isinstance(output, str)
    
    def test_text_to_image_pipeline(self):
        """Test text → dreaming → image pipeline."""
        model = DreamingReasoningLLM(device="cpu")
        
        what, action, result = model.generate(
            prompt="A bird",
            output_mode="image"
        )
        
        assert what.shape[-1] == model.embedding_dim
        assert action.shape[-1] == model.embedding_dim
        assert result.shape[-1] == model.embedding_dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
