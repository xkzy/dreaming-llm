"""Parallel situation simulation utilities."""

from __future__ import annotations

import random
from typing import List, Sequence

from .config import SimulatorConfig
from .graph_rag import GraphRAG


class Scenario:
    def __init__(self, path: List[str], score: float = 0.0) -> None:
        self.path = path
        self.score = score


class MultiScenarioSimulator:
    def __init__(self, config: SimulatorConfig, graph: GraphRAG) -> None:
        self.config = config
        self.graph = graph

    def rollout(self, seeds: Sequence[str]) -> List[Scenario]:
        simulations: List[Scenario] = []
        for seed in seeds:
            path = [seed]
            current = seed
            for _ in range(self.config.max_depth):
                neighborhood = self.graph.neighborhood(current)
                if not neighborhood:
                    break
                current = random.choice(neighborhood)
                path.append(current)
            simulations.append(Scenario(path=path))
        return simulations
