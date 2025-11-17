"""Simple evaluation heuristics for simulated reasoning chains."""

from __future__ import annotations

from typing import List

from .config import EvaluatorConfig
from .simulation import Scenario


class ScenarioEvaluator:
    def __init__(self, config: EvaluatorConfig) -> None:
        self.config = config

    def score(self, scenarios: List[Scenario]) -> List[Scenario]:
        ranked = []
        for scenario in scenarios:
            penalty = self.config.penalty_alpha * len(scenario.path)
            scenario.score = max(0.0, 1.0 - penalty)
            ranked.append(scenario)
        ranked.sort(key=lambda s: s.score, reverse=True)
        return ranked
