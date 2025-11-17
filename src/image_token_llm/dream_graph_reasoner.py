"""Graph-based reasoning over dream sequences."""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from image_token_llm.config import DreamingConfig  # type: ignore


class DreamGraphReasoner(nn.Module):
    def save_important_knowledge_to_graphml(
        self,
        dream_sequences: List[List[Tuple[torch.Tensor, ...]]],
        filepath: str,
        importance_fn=None,
        max_nodes: int = 10000,
    ):
        """
        Save a subgraph of important nodes/edges to GraphML for persistent knowledge.
        Args:
            dream_sequences: [num_dreams][dream_length][(B,D),(B,D),(B,D)]
            filepath: Path to save GraphML
            importance_fn: function(node, data) -> float, returns importance score (higher=more important)
            max_nodes: Maximum number of nodes to save (prevents memory explosion)
        """
        _, graph = self._build_graph(dream_sequences)
        # Score all nodes
        if importance_fn is None:
            # Default: all nodes equally important
            node_scores = [(n, 1.0) for n in graph.nodes()]
        else:
            node_scores = [(n, importance_fn(n, d)) for n, d in graph.nodes(data=True)]
        # Sort by importance
        node_scores.sort(key=lambda x: x[1], reverse=True)
        # Select top-K
        important_nodes = set(n for n, _ in node_scores[:max_nodes])
        # Induce subgraph
        subgraph = graph.subgraph(important_nodes).copy()
        nx.write_graphml(subgraph, filepath)
        print(f"[DreamGraphReasoner] Saved {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges to {filepath}")
    """
    Connects dream sequences into a reasoning graph.
    
    Nodes: Image triplets from all dream sequences
    Edges: Causal/temporal relationships between triplets
    Reasoning: Multi-hop graph attention to aggregate insights
    """
    
    def __init__(self, config: DreamingConfig, embedding_dim: int = 512):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        self.num_hops = config.graph_reasoning_hops
        
        # Graph attention: query, key, value projections
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Multi-hop aggregation
        self.hop_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            for _ in range(self.num_hops)
        ])
        
        # Final reasoning aggregation
        self.final_aggregator = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        
    def _build_graph(
        self,
        dream_sequences: List[List[Tuple[torch.Tensor, ...]]],
    ) -> Tuple[torch.Tensor, nx.DiGraph]:
        """
        Build reasoning graph from dream sequences.
        
        Args:
            dream_sequences: [num_dreams][dream_length][(B,D),(B,D),(B,D)]
            
        Returns:
            - node_embeddings: (B, num_nodes, embedding_dim)
            - graph: NetworkX graph with node indices
        """
        num_dreams = len(dream_sequences)
        dream_length = len(dream_sequences[0])
        B = dream_sequences[0][0][0].shape[0]
        device = dream_sequences[0][0][0].device
        
        # Collect all triplet embeddings as nodes
        nodes = []
        node_idx = 0
        graph = nx.DiGraph()
        
        # Map (dream_idx, step_idx) -> node_idx
        node_map = {}
        
        for dream_idx in range(num_dreams):
            for step_idx in range(dream_length):
                what, action, result = dream_sequences[dream_idx][step_idx]
                
                # Average triplet components to create node embedding
                node_emb = (what + action + result) / 3.0  # (B, D)
                nodes.append(node_emb)
                
                graph.add_node(node_idx)
                node_map[(dream_idx, step_idx)] = node_idx
                node_idx += 1
        
        # Stack all nodes
        node_embeddings = torch.stack(nodes, dim=1)  # (B, num_nodes, D)
        
        # Add edges: temporal within sequence, causal between sequences
        for dream_idx in range(num_dreams):
            for step_idx in range(dream_length - 1):
                # Temporal edge within dream
                src = node_map[(dream_idx, step_idx)]
                dst = node_map[(dream_idx, step_idx + 1)]
                graph.add_edge(src, dst, edge_type="temporal")
            
            # Causal edges to other dreams at same step
            for other_dream in range(num_dreams):
                if other_dream != dream_idx:
                    for step_idx in range(dream_length):
                        src = node_map[(dream_idx, step_idx)]
                        dst = node_map[(other_dream, step_idx)]
                        graph.add_edge(src, dst, edge_type="causal")
        
        return node_embeddings, graph
    
    def _graph_attention(
        self,
        node_embeddings: torch.Tensor,
        graph: nx.DiGraph,
    ) -> torch.Tensor:
        """
        Apply graph attention over nodes.
        
        Args:
            node_embeddings: (B, num_nodes, embedding_dim)
            graph: NetworkX graph
            
        Returns:
            Updated node embeddings (B, num_nodes, embedding_dim)
        """
        B, num_nodes, D = node_embeddings.shape
        device = node_embeddings.device
        
        # Project to Q, K, V
        Q = self.query_proj(node_embeddings)  # (B, num_nodes, D)
        K = self.key_proj(node_embeddings)  # (B, num_nodes, D)
        V = self.value_proj(node_embeddings)  # (B, num_nodes, D)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(1, 2))  # (B, num_nodes, num_nodes)
        scores = scores / (D ** 0.5)
        
        # Mask attention based on graph edges
        edge_mask = torch.zeros(
            num_nodes, num_nodes, device=device
        )  # (num_nodes, num_nodes)
        
        for src, dst in graph.edges():
            edge_mask[src, dst] = 1.0
        
        # Apply mask (set non-edges to -inf)
        masked_scores = scores.masked_fill(
            edge_mask.unsqueeze(0) == 0, float('-inf')
        )
        
        # Softmax attention weights
        attn_weights = F.softmax(masked_scores, dim=-1)  # (B, num_nodes, num_nodes)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # Handle isolated nodes
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # (B, num_nodes, D)
        
        return attended
    
    def forward(
        self,
        dream_sequences: List[List[Tuple[torch.Tensor, ...]]],
    ) -> torch.Tensor:
        """
        Reason over dream sequences via graph attention.
        
        Args:
            dream_sequences: [num_dreams][dream_length][(B,D),(B,D),(B,D)]
            
        Returns:
            reasoning_embedding: (B, embedding_dim) - aggregated reasoning
        """
        # Build reasoning graph
        node_embeddings, graph = self._build_graph(dream_sequences)
        # (B, num_nodes, D)
        
        # Multi-hop graph attention
        current_embeddings = node_embeddings
        
        for hop_idx in range(self.num_hops):
            # Apply graph attention
            attended = self._graph_attention(current_embeddings, graph)
            
            # Apply MLP transformation
            transformed = self.hop_layers[hop_idx](attended)
            
            # Residual connection
            current_embeddings = current_embeddings + transformed
        
        # Aggregate all nodes into final reasoning
        aggregated = current_embeddings.mean(dim=1)  # (B, D)
        
        # Final transformation
        reasoning_embedding = self.final_aggregator(aggregated)  # (B, D)
        
        return reasoning_embedding
    
    def get_graph_visualization_data(
        self,
        dream_sequences: List[List[Tuple[torch.Tensor, ...]]],
    ) -> Dict:
        """
        Extract graph structure for visualization.
        
        Returns:
            Dictionary with nodes, edges, and positions for plotting
        """
        _, graph = self._build_graph(dream_sequences)
        
        num_dreams = len(dream_sequences)
        dream_length = len(dream_sequences[0])
        
        # Compute positions (grid layout)
        pos = {}
        for dream_idx in range(num_dreams):
            for step_idx in range(dream_length):
                node_idx = dream_idx * dream_length + step_idx
                pos[node_idx] = (step_idx, -dream_idx)  # Grid layout
        
        # Extract edge info
        temporal_edges = []
        causal_edges = []
        
        for src, dst, data in graph.edges(data=True):
            edge_type = data.get("edge_type", "unknown")
            if edge_type == "temporal":
                temporal_edges.append((src, dst))
            elif edge_type == "causal":
                causal_edges.append((src, dst))
        
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "positions": pos,
            "temporal_edges": temporal_edges,
            "causal_edges": causal_edges,
            "num_dreams": num_dreams,
            "dream_length": dream_length,
        }
