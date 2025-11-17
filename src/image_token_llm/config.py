"""Configuration schemas for the image-token reasoning LLM playground."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImageTokenizerConfig(BaseModel):
    embedding_dim: int = Field(512, ge=64)
    patch_size: int = Field(16, ge=4)


class TokenizerConfig(BaseModel):
    vocab: list[str] | None = None
    pad_token: str = "<pad>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    unk_token: str = "<unk>"
    extra_tokens: list[str] = Field(
        default_factory=lambda: ["<img>", "<graph>", "<ctx>"]
    )


class TextDecoderConfig(BaseModel):
    vocab_size: int = Field(4096, ge=128)
    max_seq_len: int = Field(256, ge=32)
    num_layers: int = Field(4, ge=1)
    num_heads: int = Field(8, ge=1)
    ff_dim: int = Field(2048, ge=128)
    dropout: float = Field(0.1, ge=0.0)
    top_k: int = Field(50, ge=1)


class GraphRAGConfig(BaseModel):
    top_k_neighbors: int = Field(8, ge=1)
    max_hops: int = Field(3, ge=1)


class SimulatorConfig(BaseModel):
    branches: int = Field(4, ge=1)
    max_depth: int = Field(5, ge=1)


class EvaluatorConfig(BaseModel):
    scoring: str = "confidence"
    penalty_alpha: float = Field(0.1, ge=0.0)


class RuntimeConfig(BaseModel):
    device: str = "cuda"
    enable_gpt51_codex_preview: bool = True


def _image_tokenizer_defaults() -> ImageTokenizerConfig:
    return ImageTokenizerConfig()  # type: ignore[call-arg]


def _graph_rag_defaults() -> GraphRAGConfig:
    return GraphRAGConfig()  # type: ignore[call-arg]


def _simulator_defaults() -> SimulatorConfig:
    return SimulatorConfig()  # type: ignore[call-arg]


def _evaluator_defaults() -> EvaluatorConfig:
    return EvaluatorConfig()  # type: ignore[call-arg]


def _runtime_defaults() -> RuntimeConfig:
    return RuntimeConfig()  # type: ignore[call-arg]


def _tokenizer_defaults() -> TokenizerConfig:
    return TokenizerConfig()  # type: ignore[call-arg]


def _text_decoder_defaults() -> TextDecoderConfig:
    return TextDecoderConfig()  # type: ignore[call-arg]



class MoEConfig(BaseModel):
    num_vision_experts: int = Field(2, ge=1)
    num_graph_experts: int = Field(2, ge=1)
    num_text_experts: int = Field(2, ge=1)
    gating_hidden_dim: int = Field(128, ge=8)


class DreamingConfig(BaseModel):
    """Configuration for dreaming-based reasoning architecture."""
    num_dream_sequences: int = Field(4, ge=1)
    dream_length: int = Field(5, ge=1)
    graph_reasoning_hops: int = Field(3, ge=1)
    output_mode: str = Field("text", pattern="^(text|image|both)$")
    enable_visualization: bool = True
    
    # MoE gating improvements
    moe_top_k: int = Field(2, ge=1)
    moe_noise_std: float = Field(0.1, ge=0.0)
    load_balance_loss_weight: float = Field(0.01, ge=0.0)
    
    # Expert architecture
    expert_num_layers: int = Field(2, ge=1)
    expert_num_heads: int = Field(8, ge=1)
    use_cross_expert_attention: bool = True
    
    # Input tokenization
    use_contrastive_loss: bool = True
    contrastive_temperature: float = Field(0.07, ge=0.01)
    token_mask_prob: float = Field(0.1, ge=0.0, le=0.5)
    embedding_dropout: float = Field(0.15, ge=0.0, le=0.5)
    
    # Graph reasoning
    use_learned_edges: bool = True
    num_edge_types: int = Field(4, ge=1)
    use_node_memory: bool = True
    graph_num_heads: int = Field(8, ge=1)
    
    # RL improvements
    use_ppo: bool = True
    ppo_clip_epsilon: float = Field(0.2, ge=0.0)
    ppo_gae_lambda: float = Field(0.95, ge=0.0, le=1.0)
    reward_components: bool = True


class ExperimentConfig(BaseModel):
    hf_tokenizer_name: str = "bert-base-uncased"
    image_tokenizer: ImageTokenizerConfig = Field(
        default_factory=_image_tokenizer_defaults
    )
    graph_rag: GraphRAGConfig = Field(default_factory=_graph_rag_defaults)
    simulator: SimulatorConfig = Field(default_factory=_simulator_defaults)
    evaluator: EvaluatorConfig = Field(default_factory=_evaluator_defaults)
    runtime: RuntimeConfig = Field(default_factory=_runtime_defaults)
    tokenizer: TokenizerConfig = Field(default_factory=_tokenizer_defaults)
    text_decoder: TextDecoderConfig = Field(
        default_factory=_text_decoder_defaults
    )
    moe: MoEConfig = Field(
        default_factory=lambda: MoEConfig(
            num_vision_experts=2,
            num_graph_experts=2,
            num_text_experts=2,
            gating_hidden_dim=128
        )
    )
    dreaming: DreamingConfig = Field(
        default_factory=lambda: DreamingConfig(
            num_dream_sequences=4,
            dream_length=5,
            graph_reasoning_hops=3,
            output_mode="text",
            enable_visualization=True
        )
    )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ExperimentConfig":
        return cls(**data)  # type: ignore[arg-type]
