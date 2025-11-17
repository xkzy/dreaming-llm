"""Tests for multi-modal components."""

import torch

from image_token_llm.config import GraphRAGConfig, ImageTokenizerConfig
from image_token_llm.continuous_learning import (
    ExperienceBuffer, IncrementalLearner
)
from image_token_llm.graph_attention import (
    GraphAttentionLayer, GraphRAGEnhanced
)
from image_token_llm.vision_encoder import TripletEncoder, VisionEncoder


class TestVisionEncoder:
    """Test vision encoding components."""

    def test_vision_encoder_forward(self):
        config = ImageTokenizerConfig(embedding_dim=512, patch_size=16)
        encoder = VisionEncoder(config, backbone="clip", freeze_backbone=True)

        images = torch.randn(2, 3, 224, 224)
        embeddings = encoder(images)

        assert embeddings.shape == (2, 512)

    def test_triplet_encoder_forward(self):
        config = ImageTokenizerConfig(embedding_dim=512, patch_size=16)
        encoder = TripletEncoder(config, backbone="clip")

        what = torch.randn(2, 3, 224, 224)
        action = torch.randn(2, 3, 224, 224)
        result = torch.randn(2, 3, 224, 224)

        fused, components = encoder(what, action, result)

        assert fused.shape == (2, 512)
        assert components.shape == (2, 3, 512)


class TestGraphAttention:
    """Test graph attention mechanisms."""

    def test_graph_attention_layer(self):
        layer = GraphAttentionLayer(in_dim=128, out_dim=128, num_heads=4)

        node_features = torch.randn(1, 10, 128)
        adjacency = torch.randint(0, 2, (1, 10, 10)).float()

        output = layer(node_features, adjacency)

        assert output.shape == (1, 10, 128)

    def test_graph_rag_enhanced_ingest(self):
        config = GraphRAGConfig(top_k_neighbors=5, max_hops=2)
        graph = GraphRAGEnhanced(config, embedding_dim=256, device="cpu")

        triplets = [
            ("person", "holds", "object"),
            ("object", "is", "red"),
            ("person", "stands_in", "room"),
        ]
        embeddings = [torch.randn(256) for _ in triplets]

        graph.ingest(triplets, embeddings)

        assert len(graph.graph.nodes) >= 3
        assert len(graph.graph.edges) >= 3

    def test_graph_subgraph_extraction(self):
        config = GraphRAGConfig(top_k_neighbors=5, max_hops=2)
        graph = GraphRAGEnhanced(config, embedding_dim=256, device="cpu")

        triplets = [
            ("A", "relates", "B"),
            ("B", "connects", "C"),
            ("C", "links", "D")
        ]
        graph.ingest(triplets)

        subgraph, nodes = graph.get_subgraph(["A"], max_nodes=10)

        assert len(nodes) > 0
        assert "A" in nodes


class TestContinuousLearning:
    """Test continuous learning components."""

    def test_experience_buffer_add_sample(self):
        buffer = ExperienceBuffer(max_size=100)

        for i in range(10):
            experience = {"data": torch.randn(10), "label": i}
            buffer.add(experience, priority=1.0)

        assert len(buffer) == 10

        experiences, weights, indices = buffer.sample(batch_size=5, beta=0.4)

        assert len(experiences) == 5
        assert weights.shape[0] == 5
        assert len(indices) == 5

    def test_incremental_learner_observe(self):
        img_config = ImageTokenizerConfig(embedding_dim=512, patch_size=16)
        graph_config = GraphRAGConfig(top_k_neighbors=5, max_hops=2)

        encoder = TripletEncoder(img_config, backbone="clip")
        graph = GraphRAGEnhanced(graph_config, embedding_dim=512, device="cpu")
        learner = IncrementalLearner(encoder, graph, learning_rate=1e-3)

        what = torch.randn(1, 3, 224, 224)
        action = torch.randn(1, 3, 224, 224)
        result = torch.randn(1, 3, 224, 224)
        label = ("entity1", "action", "entity2")

        learner.observe(what, action, result, label, priority=1.0)

        assert len(learner.experience_buffer) == 1

    def test_learn_from_experience(self):
        img_config = ImageTokenizerConfig(embedding_dim=512, patch_size=16)
        graph_config = GraphRAGConfig(top_k_neighbors=5, max_hops=2)

        encoder = TripletEncoder(img_config, backbone="clip")
        graph = GraphRAGEnhanced(graph_config, embedding_dim=512, device="cpu")
        learner = IncrementalLearner(encoder, graph, learning_rate=1e-3)

        # Add multiple experiences
        for i in range(20):
            what = torch.randn(1, 3, 224, 224)
            action = torch.randn(1, 3, 224, 224)
            result = torch.randn(1, 3, 224, 224)
            label = (f"entity{i}", "action", f"entity{i+1}")
            learner.observe(what, action, result, label)

        # Learn from experiences
        metrics = learner.learn_from_experience(batch_size=8, num_steps=5)

        assert "loss" in metrics
        assert metrics["loss"] >= 0


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_end_to_end_training_step(self):
        """Test a complete training iteration."""
        img_config = ImageTokenizerConfig(embedding_dim=512, patch_size=16)
        graph_config = GraphRAGConfig(top_k_neighbors=5, max_hops=2)

        encoder = TripletEncoder(img_config, backbone="clip")
        _ = GraphRAGEnhanced(graph_config, embedding_dim=512, device="cpu")

        # Create batch
        what = torch.randn(4, 3, 224, 224)
        action = torch.randn(4, 3, 224, 224)
        result = torch.randn(4, 3, 224, 224)

        # Forward pass
        fused, components = encoder(what, action, result)

        # Compute loss
        recon_loss = torch.nn.functional.mse_loss(components[:, 0], fused)

        assert recon_loss.item() >= 0
        assert fused.shape == (4, 512)

    def test_graph_reasoning_pipeline(self):
        """Test graph reasoning with embeddings."""
        config = GraphRAGConfig(top_k_neighbors=5, max_hops=2)
        graph = GraphRAGEnhanced(config, embedding_dim=128, device="cpu")

        # Build small knowledge graph
        triplets = [
            ("cat", "is_a", "animal"),
            ("dog", "is_a", "animal"),
            ("animal", "lives_in", "habitat"),
            ("cat", "has", "fur"),
        ]
        embeddings = [torch.randn(128) for _ in triplets]
        graph.ingest(triplets, embeddings)

        # Query graph
        query_emb = torch.randn(128)
        scores = graph.reason_over_graph(["cat"], query_embedding=query_emb)

        assert len(scores) > 0
        assert "cat" in scores
