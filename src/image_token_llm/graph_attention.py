"""Graph attention networks for relational reasoning and traversal."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GraphRAGConfig

Triplet = Tuple[str, str, str]


class GraphAttentionLayer(nn.Module):
    """
    Single graph attention layer for node feature aggregation.
    Implements multi-head attention over graph neighborhoods.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        assert out_dim % num_heads == 0

        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)
        self.W_o = nn.Linear(out_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [B, N, in_dim]
            adjacency: [B, N, N] - adjacency matrix
            mask: [B, N, N] - optional attention mask
        
        Returns:
            updated_features: [B, N, out_dim]
        """
        B, N, _ = node_features.shape

        Q = self.W_q(node_features).view(B, N, self.num_heads, self.head_dim)
        K = self.W_k(node_features).view(B, N, self.num_heads, self.head_dim)
        V = self.W_v(node_features).view(B, N, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)  # [B, H, N, D]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Apply graph structure mask (adjacency)
        adjacency_mask = adjacency.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores.masked_fill(adjacency_mask == 0, float("-inf"))

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Aggregate
        out = torch.matmul(attn, V)  # [B, H, N, D]
        out = out.transpose(1, 2).contiguous().view(B, N, self.out_dim)

        out = self.W_o(out)
        out = self.layer_norm(out + node_features if self.in_dim == self.out_dim else out)

        return out


class GraphReasoningNetwork(nn.Module):
    """
    Multi-layer graph attention network for relational reasoning.
    """

    def __init__(
        self,
        config: GraphRAGConfig,
        node_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.config = config
        self.node_dim = node_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            GraphAttentionLayer(
                in_dim=node_dim,
                out_dim=node_dim,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        self.ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(node_dim * 4, node_dim),
            nn.LayerNorm(node_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [B, N, node_dim]
            adjacency: [B, N, N]
        
        Returns:
            updated_features: [B, N, node_dim]
        """
        x = node_features
        for layer in self.layers:
            x = layer(x, adjacency)

        x = self.ffn(x) + x
        return x


class GraphRAGEnhanced:
    """
    Enhanced Graph RAG with attention-based traversal and embedding support.
    """

    def __init__(
        self,
        config: GraphRAGConfig,
        embedding_dim: int = 512,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.embedding_dim = embedding_dim
        self.device = device
        self.graph = nx.MultiDiGraph()
        self.node_embeddings: Dict[str, torch.Tensor] = {}
        self.edge_embeddings: Dict[Tuple[str, str, str], torch.Tensor] = {}

        # Initialize graph reasoning network
        self.reasoning_net = GraphReasoningNetwork(
            config=config,
            node_dim=embedding_dim,
        ).to(device)

    def ingest(
        self,
        triplets: List[Triplet],
        embeddings: Optional[List[torch.Tensor]] = None,
    ) -> None:
        """
        Add triplets to the graph with optional embeddings.
        
        Args:
            triplets: List of (subject, action, result) tuples
            embeddings: Optional embeddings for each triplet
        """
        for i, (subject, action, result) in enumerate(triplets):
            self.graph.add_node(subject, type="entity")
            self.graph.add_node(result, type="entity")
            edge_key = self.graph.add_edge(subject, result, action=action)

            # Store embeddings if provided
            if embeddings is not None and i < len(embeddings):
                emb = embeddings[i].to(self.device)
                self.node_embeddings[subject] = emb
                self.node_embeddings[result] = emb
                self.edge_embeddings[(subject, result, action)] = emb

    def get_subgraph(
        self,
        seed_nodes: List[str],
        max_nodes: int = 50,
    ) -> Tuple[nx.DiGraph, List[str]]:
        """
        Extract a subgraph around seed nodes via multi-hop traversal.
        
        Returns:
            subgraph: NetworkX subgraph
            node_list: Ordered list of nodes in subgraph
        """
        visited: Set[str] = set()
        frontier = set(seed_nodes)
        node_list = []

        for hop in range(self.config.max_hops):
            if len(visited) >= max_nodes:
                break

            next_frontier = set()
            for node in frontier:
                if node not in self.graph or node in visited:
                    continue

                visited.add(node)
                node_list.append(node)

                # Get neighbors
                neighbors = list(self.graph.successors(node))
                neighbors += list(self.graph.predecessors(node))
                top_neighbors = neighbors[: self.config.top_k_neighbors]
                next_frontier.update(top_neighbors)

            frontier = next_frontier
            if not frontier:
                break

        subgraph = self.graph.subgraph(node_list).copy()
        return subgraph, node_list

    def reason_over_graph(
        self,
        seed_nodes: List[str],
        query_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform graph reasoning starting from seed nodes.
        
        Args:
            seed_nodes: Starting nodes for traversal
            query_embedding: Optional query context [embedding_dim]
        
        Returns:
            node_scores: Dictionary mapping nodes to relevance scores
        """
        subgraph, node_list = self.get_subgraph(seed_nodes)

        if not node_list:
            return {}

        # Build node feature matrix
        node_features = []
        for node in node_list:
            if node in self.node_embeddings:
                node_features.append(self.node_embeddings[node])
            else:
                # Random initialization for unseen nodes
                node_features.append(
                    torch.randn(self.embedding_dim, device=self.device) * 0.01
                )

        node_features = torch.stack(node_features).unsqueeze(0)  # [1, N, D]

        # Build adjacency matrix
        N = len(node_list)
        adjacency = torch.zeros((1, N, N), device=self.device)
        node_to_idx = {node: i for i, node in enumerate(node_list)}

        for u, v in subgraph.edges():
            if u in node_to_idx and v in node_to_idx:
                i, j = node_to_idx[u], node_to_idx[v]
                adjacency[0, i, j] = 1
                adjacency[0, j, i] = 1  # Undirected for reasoning

        # Run graph reasoning
        with torch.no_grad():
            updated_features = self.reasoning_net(node_features, adjacency)

        # Compute relevance scores
        node_scores = {}
        if query_embedding is not None:
            query_emb = query_embedding.to(self.device)
            for i, node in enumerate(node_list):
                score = F.cosine_similarity(
                    updated_features[0, i],
                    query_emb,
                    dim=0,
                )
                node_scores[node] = score
        else:
            # Default: use centrality-based scoring
            for i, node in enumerate(node_list):
                node_scores[node] = updated_features[0, i].norm()

        return node_scores

    def update_embeddings(self, node: str, embedding: torch.Tensor) -> None:
        """Update or add embedding for a node."""
        self.node_embeddings[node] = embedding.to(self.device)

    def neighborhood(self, node: str) -> List[str]:
        """Legacy method for backward compatibility."""
        if node not in self.graph:
            return []
        neighbors = set([node])
        frontier = {node}
        for _ in range(self.config.max_hops):
            next_frontier = set()
            for current in frontier:
                succ = list(self.graph.successors(current))
                neighbors.update(succ[: self.config.top_k_neighbors])
                next_frontier.update(succ)
            frontier = next_frontier
            if not frontier:
                break
        return list(neighbors)
