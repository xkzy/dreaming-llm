"""Graph-based retrieval augmented generation utilities."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import networkx as nx

from .config import GraphRAGConfig

Triplet = Tuple[str, str, str]


class GraphRAG:
    def __init__(self, config: GraphRAGConfig) -> None:
        self.config = config
        self.graph = nx.MultiDiGraph()

    def ingest(self, triplets: Iterable[Triplet]) -> None:
        for subject, action, result in triplets:
            self.graph.add_node(subject, type="entity")
            self.graph.add_node(result, type="entity")
            self.graph.add_edge(subject, result, action=action)

    def neighborhood(self, node: str) -> List[str]:
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
