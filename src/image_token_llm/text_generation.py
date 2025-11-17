"""Lightweight text generation utilities for the image-token LLM."""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Iterator, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: import transformers for loading pretrained weights
try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None

from .config import TextDecoderConfig, TokenizerConfig


class SimpleTokenizer:
    """Whitespace-aware tokenizer with a fallback character vocabulary."""

    def __init__(
        self,
        config: TokenizerConfig,
        target_vocab_size: int | None = None,
    ) -> None:
        self.config = config
        base_vocab = config.vocab or self._default_vocab()
        specials = [
            config.pad_token,
            config.bos_token,
            config.eos_token,
            config.unk_token,
            *config.extra_tokens,
        ]

        vocab: List[str] = []
        for token in specials + base_vocab:
            if token not in vocab:
                vocab.append(token)

        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {
            idx: token for token, idx in self.token_to_id.items()
        }
        self.pad_id = self.token_to_id[config.pad_token]
        self.bos_id = self.token_to_id[config.bos_token]
        self.eos_id = self.token_to_id[config.eos_token]
        self.unk_id = self.token_to_id[config.unk_token]
        self.target_vocab_size = target_vocab_size or len(self.token_to_id)

        while len(self.token_to_id) < self.target_vocab_size:
            token = f"<extra_{len(self.token_to_id)}>"
            self._register_token(token)

    def _register_token(self, token: str) -> None:
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token

    @staticmethod
    def _default_vocab() -> List[str]:
        letters = list(string.ascii_lowercase)
        digits = list(string.digits)
        punctuation = list(" .,:;!?" + string.punctuation)
        return letters + digits + punctuation

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: int | None = None,
    ) -> List[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_id)

        for word in text.split():
            token_id = self.token_to_id.get(word.lower())
            if token_id is not None:
                tokens.append(token_id)
            else:
                for char in word:
                    token_id = self.token_to_id.get(char.lower(), self.unk_id)
                    tokens.append(token_id)
            tokens.append(self.token_to_id.get(" ", self.unk_id))

        if add_special_tokens:
            tokens.append(self.eos_id)

        if max_length is not None:
            tokens = tokens[:max_length]
        return tokens

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        words: List[str] = []
        for idx in token_ids:
            token = self.id_to_token.get(idx, self.config.unk_token)
            if skip_special_tokens and token in {
                self.config.pad_token,
                self.config.bos_token,
                self.config.eos_token,
                self.config.unk_token,
                *self.config.extra_tokens,
            }:
                continue
            words.append(token)
        return "".join(words).strip()

    def save_pretrained(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": self.config.model_dump(),
                    "token_to_id": self.token_to_id,
                    "target_vocab_size": self.target_vocab_size,
                },
                handle,
                indent=2,
            )

    @classmethod
    def from_file(cls, path: str | Path) -> "SimpleTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(
            TokenizerConfig(**data["config"]),
            target_vocab_size=data.get("target_vocab_size"),
        )
        tokenizer.token_to_id = {
            k: int(v) for k, v in data["token_to_id"].items()
        }
        tokenizer.id_to_token = {
            idx: token for token, idx in tokenizer.token_to_id.items()
        }
        tokenizer.pad_id = tokenizer.token_to_id[tokenizer.config.pad_token]
        tokenizer.bos_id = tokenizer.token_to_id[tokenizer.config.bos_token]
        tokenizer.eos_id = tokenizer.token_to_id[tokenizer.config.eos_token]
        tokenizer.unk_id = tokenizer.token_to_id[tokenizer.config.unk_token]
        return tokenizer


class ImageAwareTextDecoder(nn.Module):
    """Transformer decoder conditioned on image/graph context."""

    def __init__(
        self,
        config: TextDecoderConfig,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim

        self.token_embeddings = nn.Embedding(config.vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(
            config.max_seq_len,
            embedding_dim,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.num_layers)
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(embedding_dim, config.vocab_size, bias=False)

    def load_pretrained_from_hf(self, hf_model_name_or_model, tokenizer=None):
        """
        Load embedding and output (lm_head) weights from a HuggingFace model.
        Args:
            hf_model_name_or_model: str or HuggingFace model instance
                (e.g., BertModel)
            tokenizer: Optional HuggingFace tokenizer (if not provided,
                will be loaded)
        """
        if AutoModel is None:
            raise ImportError("transformers is not installed.")
        if isinstance(hf_model_name_or_model, str):
            hf_model = AutoModel.from_pretrained(hf_model_name_or_model)
            if tokenizer is None:
                if AutoTokenizer is None:
                    raise ImportError(
                        "transformers.AutoTokenizer is not available."
                    )
                tokenizer = AutoTokenizer.from_pretrained(
                    hf_model_name_or_model
                )
        else:
            hf_model = hf_model_name_or_model
            if tokenizer is None:
                raise ValueError(
                    "Tokenizer must be provided if passing a model instance."
                )

        # Map vocab
        vocab_size = self.config.vocab_size
        embedding_dim = self.embedding_dim
        # HuggingFace: usually model.embeddings.word_embeddings.weight
        hf_emb = hf_model.get_input_embeddings().weight.data
        if hf_emb.shape[1] != embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: {hf_emb.shape[1]} vs {embedding_dim}"
            )
        # Map tokens by tokenizer vocab
        for idx in range(vocab_size):
            token = tokenizer.convert_ids_to_tokens(idx)
            hf_id = tokenizer.convert_tokens_to_ids(token)
            if 0 <= hf_id < hf_emb.shape[0]:
                self.token_embeddings.weight.data[idx] = hf_emb[hf_id]
            else:
                # Random init for OOV tokens
                nn.init.normal_(
                    self.token_embeddings.weight.data[idx], mean=0.0, std=0.02
                )

        # Output layer: try to tie weights if possible
        if hasattr(hf_model, 'cls') and hasattr(hf_model.cls, 'predictions'):
            # BERT-style
            lm_head_w = hf_model.cls.predictions.decoder.weight.data
        elif hasattr(hf_model, 'lm_head'):
            lm_head_w = hf_model.lm_head.weight.data
        else:
            lm_head_w = hf_emb  # fallback: tie to embedding
        for idx in range(vocab_size):
            token = tokenizer.convert_ids_to_tokens(idx)
            hf_id = tokenizer.convert_tokens_to_ids(token)
            if 0 <= hf_id < lm_head_w.shape[0]:
                self.lm_head.weight.data[idx] = lm_head_w[hf_id]
            else:
                nn.init.normal_(
                    self.lm_head.weight.data[idx], mean=0.0, std=0.02
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(
            positions
        ).unsqueeze(0)
        hidden = self.dropout(hidden)
        decoded = self.decoder(hidden, memory)
        logits = self.lm_head(decoded)
        return logits

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        if temperature <= 0:
            temperature = 1.0
        logits = logits / temperature
        if top_k > 0:
            values, indices = torch.topk(
                logits,
                k=min(top_k, logits.shape[-1]),
            )
            probs = F.softmax(values, dim=-1)
            choice = torch.multinomial(probs, num_samples=1)
            next_token = indices.gather(-1, choice)
            return next_token.squeeze(-1)
        return torch.argmax(logits, dim=-1)

    def generate(
        self,
        prompt_ids: Sequence[int],
        memory: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> List[int]:
        device = memory.device
        input_ids = torch.tensor(prompt_ids, device=device).unsqueeze(0)
        generated: List[int] = []

        for _ in range(max_new_tokens):
            input_trimmed = input_ids[:, -self.config.max_seq_len:]
            logits = self.forward(input_trimmed, memory)
            next_token = self._sample_next_token(
                logits[:, -1, :],
                temperature,
                top_k or self.config.top_k,
            )
            token_id = next_token.item()
            generated.append(token_id)
            input_ids = torch.cat(
                [input_ids, next_token.unsqueeze(0)],
                dim=1,
            )
            if eos_id is not None and token_id == eos_id:
                break

        return generated

    def stream_generate(
        self,
        prompt_ids: Sequence[int],
        memory: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> Iterator[int]:
        output = self.generate(
            prompt_ids,
            memory,
            max_new_tokens,
            temperature,
            top_k,
            eos_id,
        )
        for token_id in output:
            yield token_id


def build_context_memory(
    fused_embedding: torch.Tensor,
    graph_embeddings: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create transformer memory tensor from fused and graph embeddings."""

    components = [fused_embedding.unsqueeze(1)]
    if graph_embeddings is not None and graph_embeddings.numel() > 0:
        components.append(graph_embeddings)
    memory = torch.cat(components, dim=1)
    return memory
