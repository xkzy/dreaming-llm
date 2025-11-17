"""Composite multimodal LLM that reasons over images, graphs, and text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_token_llm.config import (  # type: ignore[import]
    ExperimentConfig,
)
from image_token_llm.graph_attention import (  # type: ignore[import]
    GraphRAGEnhanced,
)
from image_token_llm.graph_rag import (  # type: ignore[import]
    Triplet as GraphTriplet,
)
from image_token_llm.rl_learning import (  # type: ignore[import]
    PolicyNetwork,
    RewardModel,
    RLContinuousLearner,
)
from image_token_llm.text_generation import (  # type: ignore[import]
    ImageAwareTextDecoder,
    SimpleTokenizer,
    build_context_memory,
)
from image_token_llm.vision_encoder import (  # type: ignore[import]
    TripletEncoder,
)

TripletImage = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]




class MoEGatingNetwork(nn.Module):
    """Simple gating network for MoE selection (softmax over experts)."""

    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.fc(x), dim=-1)


class ImageTokenReasoningLLM(nn.Module):
    """
    MoE: Orchestrates vision encoding, graph reasoning, RL, and text decoding
    with multiple experts.
    """

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        device: str | None = None,
        vision_backbone: str = "lite",
        enable_rl: bool = True,
        num_vision_experts: int = None,
        num_graph_experts: int = None,
        num_text_experts: int = None,
        gating_hidden_dim: int = None,
    ) -> None:
        super().__init__()
        self.config = config or ExperimentConfig()
        requested_device = device or self.config.runtime.device
        if (
            requested_device.startswith("cuda")
            and not torch.cuda.is_available()
        ):
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        embed_dim = self.config.image_tokenizer.embedding_dim
        moe_cfg = getattr(self.config, "moe", None)
        n_vision = num_vision_experts or (moe_cfg.num_vision_experts if moe_cfg else 2)
        n_graph = num_graph_experts or (moe_cfg.num_graph_experts if moe_cfg else 2)
        n_text = num_text_experts or (moe_cfg.num_text_experts if moe_cfg else 2)
        gate_hidden = gating_hidden_dim or (moe_cfg.gating_hidden_dim if moe_cfg else 128)

        # MoE: Multiple vision experts
        self.vision_experts = nn.ModuleList([
            TripletEncoder(
                self.config.image_tokenizer,
                backbone=vision_backbone
            ).to(self.device)
            for _ in range(n_vision)
        ])
        self.vision_gate = MoEGatingNetwork(
            embed_dim, n_vision
        ).to(self.device)

        # MoE: Multiple graph experts
        self.graph_experts = nn.ModuleList([
            GraphRAGEnhanced(
                self.config.graph_rag,
                embedding_dim=embed_dim,
                device=str(self.device)
            )
            for _ in range(n_graph)
        ])
        self.graph_gate = MoEGatingNetwork(
            embed_dim, n_graph
        ).to(self.device)

        self.tokenizer = SimpleTokenizer(
            self.config.tokenizer,
            target_vocab_size=self.config.text_decoder.vocab_size,
        )

        # MoE: Multiple text experts
        self.text_experts = nn.ModuleList([
            ImageAwareTextDecoder(
                self.config.text_decoder,
                embedding_dim=embed_dim
            ).to(self.device)
            for _ in range(n_text)
        ])
        self.text_gate = MoEGatingNetwork(
            embed_dim, n_text
        ).to(self.device)

        # Use optimizer for all text experts
        self.generator_optimizer = torch.optim.AdamW(
            [p for expert in self.text_experts for p in expert.parameters()],
            lr=3e-4,
        )

        self.enable_rl = enable_rl
        # RL only uses first expert for now (can be extended)
        if enable_rl:
            reward_model = RewardModel(embedding_dim=embed_dim).to(self.device)
            policy_network = PolicyNetwork(
                embedding_dim=embed_dim
            ).to(self.device)
            self.rl_learner = RLContinuousLearner(
                self.vision_experts[0],
                self.graph_experts[0],
                reward_model,
                policy_network,
            )
        else:
            self.rl_learner = None

        # For backward compatibility, keep single-expert aliases
        self.triplet_encoder = self.vision_experts[0]
        self.graph_rag = self.graph_experts[0]
        self.text_decoder = self.text_experts[0]

        self.last_metadata: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Encoding utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            return tensor.unsqueeze(0)
        return tensor

    def _encode_triplets(
        self,
        triplets: Sequence[TripletImage],
    ) -> torch.Tensor:
        if not triplets:
            return torch.zeros(
                1,
                self.config.image_tokenizer.embedding_dim,
                device=self.device,
            )

        embeddings = []
        with torch.no_grad():
            for what, action, result in triplets:
                what_b = self._ensure_batch(what).to(self.device)
                action_b = self._ensure_batch(action).to(self.device)
                result_b = self._ensure_batch(result).to(self.device)
                # MoE: Each expert encodes, gate weights combine
                expert_outputs = []
                for expert in self.vision_experts:
                    fused, _ = expert(what_b, action_b, result_b)
                    expert_outputs.append(fused.squeeze(0))
                expert_outputs = torch.stack(expert_outputs, dim=0)
                # Use mean of inputs as gate input
                gate_input = torch.mean(expert_outputs, dim=0, keepdim=True)
                gate_weights = self.vision_gate(gate_input)
                fused_moe = torch.sum(
                    gate_weights.view(-1, 1) * expert_outputs, dim=0
                )
                embeddings.append(fused_moe)

        stacked = torch.stack(embeddings).to(self.device)
        return stacked

    def _graph_context_embeddings(
        self,
        seeds: Sequence[str],
        query_embedding: torch.Tensor,
        max_nodes: int = 8,
    ) -> torch.Tensor | None:
        if not seeds:
            return None

        # MoE: Each graph expert reasons, gate weights combine
        expert_scores = []
        for expert in self.graph_experts:
            # Cast to correct type for static checkers
            graph_expert = expert  # type: GraphRAGEnhanced
            scores = graph_expert.reason_over_graph(
                list(seeds),
                query_embedding=query_embedding.squeeze(0),
            )
            expert_scores.append(scores)
        # Average scores for node ranking
        if not expert_scores or not any(expert_scores):
            return None
        # Merge node scores (mean across experts)
        all_nodes = set()
        for scores in expert_scores:
            all_nodes.update(scores.keys())
        merged_scores = {}
        for node in all_nodes:
            merged_scores[node] = float(
                sum(scores.get(node, 0.0) for scores in expert_scores) /
                len(expert_scores)
            )
        sorted_nodes = sorted(
            merged_scores.items(),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:max_nodes]

        # Combine node embeddings from all experts using gate
        embeddings: List[torch.Tensor] = []
        for node, _ in sorted_nodes:
            node_embeds = []
            for expert in self.graph_experts:
                graph_expert = expert  # type: GraphRAGEnhanced
                if node in graph_expert.node_embeddings:
                    node_embeds.append(
                        graph_expert.node_embeddings[node]
                        .detach()
                        .unsqueeze(0)
                        .to(self.device)
                    )
            if node_embeds:
                node_embeds = torch.stack(node_embeds, dim=0)
                gate_input = torch.mean(node_embeds, dim=0, keepdim=True)
                gate_weights = self.graph_gate(gate_input)
                fused_node = torch.sum(
                    gate_weights.view(-1, 1) * node_embeds, dim=0
                )
                embeddings.append(fused_node)
        if not embeddings:
            return None
        return torch.stack(embeddings, dim=1)

    def ingest_graph(self, triplets: Iterable[GraphTriplet]) -> None:
        # Ingest into all graph experts
        for expert in self.graph_experts:
            graph_expert = expert  # type: GraphRAGEnhanced
            graph_expert.ingest(list(triplets))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _prepare_memory(
        self,
        fused_embedding: torch.Tensor,
        seeds: Sequence[str],
    ) -> torch.Tensor:
        graph_ctx = self._graph_context_embeddings(seeds, fused_embedding)
        return build_context_memory(fused_embedding, graph_ctx)

    def _resolve_seeds(
        self,
        prompt: str,
        explicit: Sequence[str] | None,
    ) -> List[str]:
        if explicit:
            return list(explicit)
        return [token for token in prompt.split()[:4] if token]

    def generate(
        self,
        prompt: str,
        image_triplets: Sequence[TripletImage] | None = None,
        graph_triplets: Sequence[GraphTriplet] | None = None,
        seeds: Sequence[str] | None = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        stream: bool = False,
    ) -> str | Iterator[str]:
        if graph_triplets:
            self.ingest_graph(graph_triplets)

        fused_embeddings = (
            self._encode_triplets(image_triplets)
            if image_triplets
            else torch.zeros(
                1,
                self.config.image_tokenizer.embedding_dim,
                device=self.device,
            )
        )
        fused = fused_embeddings.mean(dim=0, keepdim=True)

        resolved_seeds = self._resolve_seeds(prompt, seeds)
        memory = self._prepare_memory(fused, resolved_seeds)

        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            max_length=self.config.text_decoder.max_seq_len,
        )
        max_tokens = max_new_tokens or (
            self.config.text_decoder.max_seq_len // 2
        )

        # MoE: All text experts generate, gate weights combine logits
        expert_logits = []
        for expert in self.text_experts:
            logits = expert(
                torch.tensor(prompt_ids, device=self.device).unsqueeze(0),
                memory
            )
            expert_logits.append(logits)
        expert_logits = torch.stack(
            expert_logits, dim=0
        )  # (num_experts, B, T, V)
        gate_input = fused
        gate_weights = self.text_gate(gate_input)
        # Weighted sum over experts
        combined_logits = torch.sum(
            gate_weights.view(-1, 1, 1, 1) * expert_logits, dim=0
        )
        # Use first expert's generate for now (streaming/generation logic can be extended)
        if stream:
            return self._stream_tokens(
                prompt_ids,
                memory,
                max_tokens,
                temperature,
            )
        # Greedy decode from combined logits
        new_token_ids = torch.argmax(combined_logits, dim=-1).squeeze(0).tolist()
        generated_text = self.tokenizer.decode(new_token_ids)

        rl_metrics: Dict[str, float] = {}
        if self.rl_learner is not None and image_triplets:
            rl_metrics = self._run_rl_update(image_triplets[0])

        self.last_metadata = {
            "seeds": resolved_seeds,
            "rl_metrics": rl_metrics,
            "num_graph_nodes": sum(
                len(getattr((expert if hasattr(expert, 'graph') else None), 'graph', []).nodes)
                if hasattr(getattr(expert, 'graph', None), 'nodes') else 0
                for expert in self.graph_experts
            ),
        }
        return generated_text

    def _stream_tokens(
        self,
        prompt_ids: Sequence[int],
        memory: torch.Tensor,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        # Use first text expert for streaming for now
        text_expert = self.text_experts[0]  # type: ImageAwareTextDecoder
        for token_id in text_expert.stream_generate(
            prompt_ids,
            memory,
            max_tokens,
            temperature=temperature,
            eos_id=self.tokenizer.eos_id,
        ):
            decoded = self.tokenizer.decode([token_id])
            if decoded:
                yield decoded

    def _run_rl_update(self, triplet: TripletImage) -> Dict[str, float]:
        if self.rl_learner is None:
            return {}

        what = self._ensure_batch(triplet[0]).to(self.device)
        action = self._ensure_batch(triplet[1]).to(self.device)
        result = self._ensure_batch(triplet[2]).to(self.device)
        _, metrics = self.rl_learner.online_inference_with_learning(
            what,
            action,
            result,
        )
        return metrics

    # ------------------------------------------------------------------
    # Knowledge transfer & export
    # ------------------------------------------------------------------
    def distill_from_ollama(
        self,
        prompts: Sequence[str],
        teacher_model: str = "llama2",
        num_samples: int = 1,
    ) -> Dict[str, float]:
        """Distill traces from an Ollama teacher into the text decoder."""

        # Lazy import keeps heavy optional deps out of the hot path.
        from .knowledge_transfer import OllamaDistillationPipeline

        pipeline = OllamaDistillationPipeline(teacher_model=teacher_model)
        traces = pipeline.generate_reasoning_traces(list(prompts), num_samples)

        # Train all text experts
        for expert in self.text_experts:
            expert.train()
        losses: List[float] = []
        vocab_size = self.config.text_decoder.vocab_size
        
        for trace in traces:
            target_text = trace.get("response", "")
            target_ids = self.tokenizer.encode(
                target_text,
                max_length=self.config.text_decoder.max_seq_len,
            )
            if len(target_ids) < 2:
                continue

            # Clamp tokens to valid vocab range
            target_ids = [min(tid, vocab_size - 1) for tid in target_ids]

            inputs = torch.tensor(
                target_ids[:-1],
                device=self.device,
            ).unsqueeze(0)
            targets = torch.tensor(
                target_ids[1:],
                device=self.device,
            ).unsqueeze(0)
            memory = build_context_memory(
                torch.zeros(
                    1,
                    self.config.image_tokenizer.embedding_dim,
                    device=self.device,
                )
            )
            # Use all text experts, combine with gate
            logits_list = [expert(inputs, memory) for expert in self.text_experts]
            logits_stack = torch.stack(logits_list, dim=0)
            gate_input = torch.mean(logits_stack, dim=0, keepdim=True)
            gate_weights = self.text_gate(gate_input)
            logits = torch.sum(gate_weights.view(-1, 1, 1) * logits_stack, dim=0)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            self.generator_optimizer.zero_grad()
            loss.backward()
            # Clip all text expert grads
            for expert in self.text_experts:
                torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
            self.generator_optimizer.step()
            losses.append(loss.item())

        avg_loss = sum(losses) / max(len(losses), 1)
        self.last_metadata["distillation_loss"] = avg_loss
        return {"distillation_loss": avg_loss, "traces": len(traces)}

    def export_ollama_bundle(
        self,
        output_dir: str | Path,
        bundle_name: str = "image-token-llm",
    ) -> Path:
        """Export weights, tokenizer, and Modelfile artifacts for Ollama."""

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

        weights_path = target / f"{bundle_name}_weights.pt"
        torch.save(
            {
                "triplet_encoder": self.triplet_encoder.state_dict(),
                "text_decoder": self.text_decoder.state_dict(),
            },
            weights_path,
        )

        tokenizer_path = target / "tokenizer.json"
        self.tokenizer.save_pretrained(tokenizer_path)

        config_path = target / "config.json"
        config_path.write_text(
            json.dumps(self.config.model_dump(), indent=2),
            encoding="utf-8",
        )

        modelfile = target / "Modelfile"
        modelfile_text = (
            "FROM mistral\n"
            "PARAMETER temperature 0.2\n"
            'SYSTEM """\n'
            "This bundle routes prompts through the image-token reasoning "
            "runtime.\n"
            "Launch `uvicorn image_token_llm.ollama_adapter:app` and point "
            "the\n"
            "`OLLAMA_HOST` environment variable to http://localhost:8000.\n"
            '"""\n'
        )
        modelfile.write_text(modelfile_text, encoding="utf-8")

        readme = target / "README.md"
        readme_text = (
            f"# {bundle_name} Ollama Bundle\n\n"
            "Artifacts generated by `ImageTokenReasoningLLM.`\n"
            "export_ollama_bundle`.\n"
            "\n"
            "1. Start the compatible server:\n"
            "   uvicorn image_token_llm.ollama_adapter:app --host 0.0.0.0\n"
            "   --port 8000\n"
            "2. Point the Ollama CLI to it: \n"
            "   export OLLAMA_HOST=http://localhost:8000\n"
            f"3. The serialized weights live at `{weights_path.name}`.\n"
            "4. Tokenizer + config files mirror the runtime state for"
            " reproducibility.\n"
        )
        readme.write_text(readme_text, encoding="utf-8")

        self.last_metadata["bundle_path"] = str(target)
        return target

    @classmethod
    def load_from_bundle(
        cls,
        bundle_dir: str | Path,
        device: str | None = None,
        vision_backbone: str = "lite",
        enable_rl: bool = True,
    ) -> "ImageTokenReasoningLLM":
        """Recreate a model from ``export_ollama_bundle`` artifacts."""

        bundle_path = Path(bundle_dir)
        config_path = bundle_path / "config.json"
        weights_path = next(bundle_path.glob("*_weights.pt"), None)
        tokenizer_path = bundle_path / "tokenizer.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Missing config.json in bundle directory: {bundle_path}"
            )
        if weights_path is None or not weights_path.exists():
            raise FileNotFoundError(
                f"Missing *_weights.pt file in bundle directory: {bundle_path}"
            )

        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        config = ExperimentConfig.from_dict(config_data)

        model = cls(
            config=config,
            device=device,
            vision_backbone=vision_backbone,
            enable_rl=enable_rl,
        )
        state_dict = torch.load(weights_path, map_location=model.device)
        model.triplet_encoder.load_state_dict(state_dict["triplet_encoder"])
        model.text_decoder.load_state_dict(state_dict["text_decoder"])

        if tokenizer_path.exists():
            model.tokenizer = SimpleTokenizer.from_file(tokenizer_path)

        model.last_metadata["bundle_path"] = str(bundle_path)
        return model
