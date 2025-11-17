"""High-level orchestration for image-token reasoning."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch

from .config import ExperimentConfig
from .evaluator import ScenarioEvaluator
from .graph_rag import GraphRAG
from .image_tokenizer import ImageTokenizer
from .simulation import MultiScenarioSimulator
from .utils import is_gpt51_enabled


TripletImage = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ReasoningOrchestrator:
    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self.config = config or ExperimentConfig()
        self.tokenizer = ImageTokenizer(self.config.image_tokenizer)
        self.graph_rag = GraphRAG(self.config.graph_rag)
        self.simulator = MultiScenarioSimulator(
            self.config.simulator, self.graph_rag
        )
        self.evaluator = ScenarioEvaluator(self.config.evaluator)

    def prime_graph(self, triplets: Iterable[tuple[str, str, str]]) -> None:
        self.graph_rag.ingest(triplets)

    def register_image_triplets(
        self, triplets: Sequence[TripletImage]
    ) -> list[tuple[str, str, str]]:
        encoded: list[tuple[str, str, str]] = []
        for triplet in triplets:
            tokens = self.tokenizer.tokenize(triplet)
            hashes = tuple(self._token_signature(tok) for tok in tokens)
            encoded.append(hashes)  # type: ignore[arg-type]
        self.graph_rag.ingest(encoded)
        return encoded

    def infer(
        self,
        seeds: Sequence[str] | None = None,
        image_triplets: Sequence[TripletImage] | None = None,
    ) -> dict[str, object]:
        working_seeds: list[str] = list(seeds or [])
        if image_triplets:
            encoded = self.register_image_triplets(image_triplets)
            working_seeds.extend([triplet[0] for triplet in encoded])
        if not working_seeds:
            raise ValueError(
                "At least one textual or image seed must be supplied."
            )

        simulations = self.simulator.rollout(working_seeds)
        ranked = self.evaluator.score(simulations)
        best = ranked[0] if ranked else None
        return {
            "best": best.path if best else [],
            "score": best.score if best else 0.0,
            "seeds": working_seeds,
            "gpt51_preview": is_gpt51_enabled(self.config.runtime),
        }

    @staticmethod
    def _token_signature(token: torch.Tensor) -> str:
        mean_val = float(token.float().mean().item())
        return f"tok-{hash(round(mean_val, 6)) & 0xFFFF_FFFF}"
