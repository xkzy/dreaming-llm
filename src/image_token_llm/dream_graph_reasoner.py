"""Graph-based reasoning over dream sequences."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from image_token_llm.config import DreamingConfig  # type: ignore


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer layer with edge-aware attention.
    Implements relational attention with edge type embeddings.
    """
    
    def __init__(self, embedding_dim: int, num_heads: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        assert (
            self.head_dim * num_heads == embedding_dim
        ), "embedding_dim must be divisible by num_heads"
        
        # Pre-norm architecture for stability
        self.node_norm = nn.LayerNorm(embedding_dim)
        self.edge_norm = nn.LayerNorm(embedding_dim)
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Edge-aware bias for attention scores
        self.edge_bias = nn.Linear(embedding_dim, num_heads)
        
        # Feedforward network with gated activation
        self.ff_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(0.1),
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        adjacency_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (B, N, D) - Node embeddings
            edge_features: (B, N, N, D) - Edge embeddings (optional)
            adjacency_mask: (B, N, N) - Adjacency matrix (1=edge, 0=no edge)
            
        Returns:
            updated_nodes: (B, N, D) - Updated node embeddings
        """
        B, N, D = node_features.shape
        
        # Pre-norm
        normed_nodes = self.node_norm(node_features)
        
        # Compute Q, K, V
        Q = self.q_proj(normed_nodes)  # (B, N, D)
        K = self.k_proj(normed_nodes)  # (B, N, D)
        V = self.v_proj(normed_nodes)  # (B, N, D)
        
        # Reshape for multi-head attention
        # (B, N, D) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # scores: (B, num_heads, N, N)
        
        # Add edge-aware bias if edge features provided
        if edge_features is not None:
            normed_edges = self.edge_norm(edge_features)  # (B, N, N, D)
            edge_bias = self.edge_bias(normed_edges)  # (B, N, N, num_heads)
            edge_bias = edge_bias.permute(0, 3, 1, 2)  # (B, num_heads, N, N)
            scores = scores + edge_bias
        
        # Apply adjacency mask
        if adjacency_mask is not None:
            # (B, N, N) -> (B, 1, N, N) for broadcasting
            mask = adjacency_mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and apply attention
        attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, N, N)
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, N, head_dim)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, N, D)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection
        node_features = node_features + attn_output
        
        # Feedforward with residual
        node_features = node_features + self.feedforward(
            self.ff_norm(node_features)
        )
        
        return node_features


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
        
        # Learned edge predictor network
        self.edge_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, 4),  # 4 edge types
        )
        
        # Edge type embeddings (temporal, spatial, causal, abstract)
        self.edge_embeddings = nn.Parameter(torch.randn(4, embedding_dim))
        
        # Graph Transformer layers with edge-aware attention
        self.gt_layers = nn.ModuleList([
            GraphTransformerLayer(embedding_dim, num_heads=8)
            for _ in range(self.num_hops)
        ])
        
        # Node memory states for recurrent updates
        self.node_memory = nn.GRUCell(embedding_dim, embedding_dim)
        
        # Final reasoning aggregation with gated activation
        self.final_aggregator = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
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
        Reason over dream sequences via Graph Transformer.
        
        Args:
            dream_sequences: [num_dreams][dream_length][(B,D),(B,D),(B,D)]
            
        Returns:
            reasoning_embedding: (B, embedding_dim) - aggregated reasoning
        """
        # Build reasoning graph
        node_embeddings, graph = self._build_graph(dream_sequences)
        # (B, num_nodes, D)
        
        B, num_nodes, D = node_embeddings.shape
        device = node_embeddings.device
        
        # Build adjacency matrix
        adjacency = torch.zeros(B, num_nodes, num_nodes, device=device)
        edge_features = torch.zeros(B, num_nodes, num_nodes, D, device=device)
        
        # Predict edge types and create edge features
        for i in range(num_nodes):
            for j in range(num_nodes):
                if graph.has_edge(i, j):
                    # Concatenate source and target node embeddings
                    edge_input = torch.cat([
                        node_embeddings[:, i, :],
                        node_embeddings[:, j, :]
                    ], dim=-1)  # (B, 2*D)
                    
                    # Predict edge type
                    edge_logits = self.edge_predictor(edge_input)  # (B, 4)
                    edge_type_probs = F.softmax(edge_logits, dim=-1)
                    
                    # Weighted sum of edge type embeddings
                    edge_emb = torch.matmul(
                        edge_type_probs, self.edge_embeddings
                    )  # (B, D)
                    
                    edge_features[:, i, j, :] = edge_emb
                    adjacency[:, i, j] = 1.0
        
        # Initialize node memory states (always create fresh, detached tensor)
        if not hasattr(self, '_node_memory_states') or self._node_memory_states.shape[0] != B:
            self._node_memory_states = torch.zeros(
                B, num_nodes, D, device=device
            )
        else:
            # Detach from previous computation graph
            self._node_memory_states = self._node_memory_states.detach()
        
        # Multi-hop graph reasoning with memory
        current_embeddings = node_embeddings
        
        for hop_idx in range(self.num_hops):
            # Apply Graph Transformer layer
            updated_embeddings = self.gt_layers[hop_idx](
                current_embeddings, edge_features, adjacency
            )
            
            # Update node memory with GRU
            # Flatten: (B, num_nodes, D) -> (B*num_nodes, D)
            flat_updated = updated_embeddings.view(-1, D)
            flat_memory = self._node_memory_states.detach().view(-1, D)
            
            # Apply GRU cell
            new_memory = self.node_memory(flat_updated, flat_memory)
            
            # Reshape back
            self._node_memory_states = new_memory.view(B, num_nodes, D)
            current_embeddings = self._node_memory_states
        
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
