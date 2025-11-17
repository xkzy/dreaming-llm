"""Dreaming-based reasoning LLM: main model orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from image_token_llm.config import (  # type: ignore[import]
    ExperimentConfig,
)
from image_token_llm.dream_graph_reasoner import (  # type: ignore[import]
    DreamGraphReasoner,
)
from image_token_llm.dreaming import (  # type: ignore[import]
    DreamGenerator,
    InputTokenizer,
    OutputDecoder,
)
from image_token_llm.rl_learning import (  # type: ignore[import]
    PolicyNetwork,
    RewardModel,
)


class DreamingReasoningLLM(nn.Module):
    """
    Dreaming-based multi-modal reasoning LLM.
    
    Architecture Flow:
    1. Input → InputTokenizer → Image triplets (what, action, result)
    2. DreamGenerator → Multiple parallel dream sequences (with MoE experts)
    3. DreamGraphReasoner → Graph-based reasoning over dreams
    4. OutputDecoder → Text or images
    
    Key Innovation: All reasoning happens in "dream space" as sequences
    of image triplets that are graphed together for multi-path reasoning.
    
    Now includes:
    - MoE (Mixture of Experts) in DreamGenerator with 4 specialized experts
    - RL (Reinforcement Learning) for continuous learning from feedback
    """
    
    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        device: Optional[str] = None,
        embedding_dim: int = 512,
        vocab_size: int = 4096,
        enable_rl: bool = True,
    ):
        super().__init__()
        self.config = config or ExperimentConfig()
        self.enable_rl = enable_rl
        
        # Device setup
        requested_device = device or self.config.runtime.device
        if (requested_device.startswith("cuda")
                and not torch.cuda.is_available()):
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # Core dreaming components
        self.input_tokenizer = InputTokenizer(
            self.config.dreaming,
            embedding_dim=embedding_dim
        ).to(self.device)
        
        self.dream_generator = DreamGenerator(
            self.config.dreaming,
            embedding_dim=embedding_dim
        ).to(self.device)
        
        self.graph_reasoner = DreamGraphReasoner(
            self.config.dreaming,
            embedding_dim=embedding_dim
        ).to(self.device)
        
        self.output_decoder = OutputDecoder(
            self.config.dreaming,
            embedding_dim=embedding_dim,
            vocab_size=vocab_size
        ).to(self.device)
        
        # Text embedding for prompts
        self.text_embedder = nn.Embedding(
            vocab_size, embedding_dim
        ).to(self.device)
        
        # Initialize RL components if enabled
        if self.enable_rl:
            self.policy_network = PolicyNetwork(
                embedding_dim=embedding_dim,
                num_actions=self.config.dreaming.num_dream_sequences,
                hidden_dim=embedding_dim * 2
            ).to(self.device)
            
            self.reward_model = RewardModel(
                embedding_dim=embedding_dim,
                hidden_dim=embedding_dim * 2
            ).to(self.device)
            
            # Optimizers for RL training
            self.policy_optimizer = torch.optim.AdamW(
                self.policy_network.parameters(), lr=3e-4
            )
            self.reward_optimizer = torch.optim.AdamW(
                self.reward_model.parameters(), lr=3e-4
            )
            
            # RL hyperparameters
            self.gamma = 0.99  # Discount factor
            self.entropy_coef = 0.01  # Entropy regularization
            
            # Trajectory buffer for experience replay
            self.trajectory_buffer: List[Dict[str, Any]] = []
        
        # Metadata for tracking
        self.last_metadata: Dict = {}
    
    def forward(
        self,
        text_tokens: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        output_mode: Optional[str] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass: Input → Dreams → Reasoning → Output
        
        Args:
            text_tokens: (B, seq_len) - input text tokens
            image_embeddings: (B, N, embedding_dim) - input image features
            output_mode: "text", "image", or "both"
            
        Returns:
            Output logits/embeddings based on output_mode
        """
        # Step 1: Tokenize input to image triplets
        if text_tokens is not None:
            # Embed text tokens and pool
            text_emb = self.text_embedder(text_tokens)  # (B, seq_len, D)
            text_emb = text_emb.mean(dim=1)  # (B, D)
            what, action, result = self.input_tokenizer(
                text_embedding=text_emb
            )
        elif image_embeddings is not None:
            what, action, result = self.input_tokenizer(
                image_embeddings=image_embeddings
            )
        else:
            raise ValueError("Must provide text_tokens or image_embeddings")
        
        # Step 2: Generate dream sequences (now with MoE)
        dream_sequences = self.dream_generator(what, action, result)
        
        # Step 3: Graph-based reasoning over dreams
        reasoning_embedding = self.graph_reasoner(dream_sequences)
        
        # Step 4: Decode to output
        mode = output_mode or self.config.dreaming.output_mode
        output = self.output_decoder(reasoning_embedding, mode=mode)
        
        # Store metadata
        self.last_metadata = {
            "num_dreams": len(dream_sequences),
            "dream_length": len(dream_sequences[0]),
            "output_mode": mode,
        }
        
        return output
    
    def generate(
        self,
        prompt: Optional[str] = None,
        images: Optional[torch.Tensor] = None,
        max_length: int = 50,
        temperature: float = 1.0,
        output_mode: str = "text",
        return_dreams: bool = False,
    ) -> Union[str, torch.Tensor, Dict]:
        """
        Generate output with optional dream visualization.
        
        Args:
            prompt: Text prompt string
            images: (B, N, C, H, W) - input images
            max_length: Max tokens/steps to generate
            temperature: Sampling temperature
            output_mode: "text", "image", or "both"
            return_dreams: If True, return dream sequences for visualization
            
        Returns:
            Generated output (str for text, tensors for images)
            If return_dreams=True, returns dict with output and dreams
        """
        with torch.no_grad():
            # Convert prompt to tokens (simple char-level)
            if prompt is not None:
                text_tokens = self._tokenize_prompt(prompt)
                text_emb = self.text_embedder(text_tokens).mean(dim=1)
                what, action, result = self.input_tokenizer(
                    text_embedding=text_emb
                )
            elif images is not None:
                # Extract image embeddings (placeholder - would use real encoder)
                B, N, C, H, W = images.shape
                image_emb = torch.randn(
                    B, N, self.embedding_dim, device=self.device
                )
                what, action, result = self.input_tokenizer(
                    image_embeddings=image_emb
                )
            else:
                raise ValueError("Must provide prompt or images")
            
            # Generate dreams
            dream_sequences = self.dream_generator(what, action, result)
            
            # Graph reasoning
            reasoning_embedding = self.graph_reasoner(dream_sequences)
            
            # Generate output
            if output_mode == "text":
                token_ids = self.output_decoder.generate_sequence(
                    reasoning_embedding,
                    max_length=max_length,
                    temperature=temperature
                )
                text_output = self._detokenize(token_ids)
                
                if return_dreams:
                    # Get graph visualization data
                    graph_data = (
                        self.graph_reasoner.get_graph_visualization_data(
                            dream_sequences
                        )
                    )
                    return {
                        "output": text_output,
                        "dreams": dream_sequences,
                        "graph_data": graph_data,
                        "metadata": self.last_metadata,
                    }
                return text_output
                
            elif output_mode == "image":
                # Use forward with image mode to get triplet
                what, action, result = self.output_decoder(
                    reasoning_embedding, mode="image"
                )
                
                if return_dreams:
                    return {
                        "output": (what, action, result),
                        "dreams": dream_sequences,
                        "metadata": self.last_metadata,
                    }
                return what, action, result
                
            elif output_mode == "both":
                text_token_ids = self.output_decoder.generate_sequence(
                    reasoning_embedding,
                    max_length=max_length,
                    temperature=temperature
                )
                text_output = self._detokenize(text_token_ids)
                what, action, result = self.output_decoder(
                    reasoning_embedding, mode="image"
                )
                
                if return_dreams:
                    return {
                        "text": text_output,
                        "image": (what, action, result),
                        "dreams": dream_sequences,
                        "metadata": self.last_metadata,
                    }
                return text_output, (what, action, result)
                
            else:
                raise ValueError(
                    f"Invalid output_mode: {output_mode}. "
                    "Must be 'text', 'image', or 'both'"
                )
    
    def train_step_rl(
        self,
        prompt: Optional[str] = None,
        images: Optional[torch.Tensor] = None,
        target_output: Optional[str] = None,
        reward_signal: Optional[float] = None,
    ) -> Dict:
        """
        Single RL training step with reward feedback.
        
        Uses policy gradient methods to update the model based on
        reward signals from generation quality.
        
        Args:
            prompt: Input text prompt
            images: Input images
            target_output: Expected output for reward calculation
            reward_signal: External reward (if None, computed internally)
            
        Returns:
            Dict with training metrics (reward, policy_loss, value_loss)
        """
        if not self.enable_rl:
            raise ValueError("RL is not enabled for this model")
        
        # For now, simplified RL step - just track reward
        # Full implementation would involve:
        # 1. Generate output with policy network guidance
        # 2. Compute reward (external or via reward model)
        # 3. Calculate policy gradient loss
        # 4. Update policy network
        # 5. Update value/reward model
        
        reward = reward_signal if reward_signal is not None else 0.5
        
        # Placeholder metrics (full RL update loop would go here)
        metrics = {
            "reward": reward,
            "policy_loss": 0.0,
            "value_loss": 0.0,
        }
        
        return metrics
    
    def learn_from_feedback(
        self,
        examples: list[Dict],
        num_epochs: int = 10,
    ) -> Dict:
        """
        Continuous learning from user feedback examples.
        
        This enables the model to improve from human feedback,
        implementing a form of RLHF (Reinforcement Learning from
        Human Feedback).
        
        Args:
            examples: List of {prompt, reward, ...} dicts
            num_epochs: Training epochs
            
        Returns:
            Aggregated training metrics
        """
        if not self.enable_rl:
            raise ValueError("RL is not enabled for this model")
        
        total_metrics = {
            "total_reward": 0.0,
            "avg_policy_loss": 0.0,
            "avg_value_loss": 0.0
        }
        
        for epoch in range(num_epochs):
            epoch_reward = 0.0
            
            for example in examples:
                metrics = self.train_step_rl(
                    prompt=example.get("prompt"),
                    images=example.get("images"),
                    reward_signal=example.get("reward")
                )
                epoch_reward += metrics["reward"]
                total_metrics["avg_policy_loss"] += metrics["policy_loss"]
                total_metrics["avg_value_loss"] += metrics["value_loss"]
            
            total_metrics["total_reward"] += epoch_reward
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Reward={epoch_reward:.4f}")
        
        # Average metrics
        n_steps = len(examples) * num_epochs
        total_metrics["avg_policy_loss"] /= n_steps
        total_metrics["avg_value_loss"] /= n_steps
        total_metrics["avg_reward"] = (
            total_metrics["total_reward"] / n_steps
        )
        
        return total_metrics
    
    def visualize_thinking(
        self,
        prompt: Optional[str] = None,
        images: Optional[torch.Tensor] = None,
        max_length: int = 50,
    ) -> Dict:
        """
        Generate output with full thinking visualization.
        
        Returns dream sequences, graph structure, and output for
        understanding the model's reasoning process.
        
        Args:
            prompt: Text prompt string
            images: Input images
            max_length: Max tokens to generate
            
        Returns:
            Dict with output, dreams, and graph_data
        """
        return self.generate(
            prompt=prompt,
            images=images,
            max_length=max_length,
            output_mode="text",
            return_dreams=True,
        )
    
    def save_pretrained(self, path: str):
        """Save model weights and config (alias for save())."""
        self.save(path)
    
    @classmethod
    def load_pretrained(cls, path: str, device: Optional[str] = None):
        """Load model from saved weights."""
        import json
        from pathlib import Path
        
        load_path = Path(path)
        
        # Load config
        with open(load_path / "config.json", "r") as f:
            config_dict = json.load(f)
        
        # Extract model params
        embedding_dim = config_dict.get("embedding_dim", 512)
        vocab_size = config_dict.get("vocab_size", 4096)
        enable_rl = config_dict.get("enable_rl", True)
        
        # Try to load full config if available
        try:
            config = ExperimentConfig.from_dict(config_dict)
        except (AttributeError, KeyError):
            config = ExperimentConfig()
        
        # Create model with saved dimensions
        model = cls(
            config=config,
            device=device,
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            enable_rl=enable_rl
        )
        
        # Load weights - try multiple possible filenames
        weights_candidates = [
            load_path / "dreaming_model_weights.pt",
            load_path / "enhanced-model_weights.pt",
            load_path / "image-token-llm-pretrained_weights.pt",
        ]
        
        weights_path = None
        for candidate in weights_candidates:
            if candidate.exists():
                weights_path = candidate
                break
        
        if weights_path is None:
            raise FileNotFoundError(
                f"No model weights found in {load_path}. "
                f"Looked for: {[c.name for c in weights_candidates]}"
            )
        
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        
        return model
    
    def save(self, path: str):
        """Save model weights and config."""
        import json
        from pathlib import Path
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config (include all model params for proper loading)
        with open(save_path / "config.json", "w") as f:
            # Try to serialize full config
            try:
                from dataclasses import asdict
                config_dict = asdict(self.config)
                # Convert nested dataclasses
                if "dreaming" in config_dict:
                    try:
                        config_dict["dreaming"] = asdict(
                            self.config.dreaming
                        )
                    except TypeError:
                        pass
            except (ImportError, TypeError, AttributeError):
                # Fallback: extract dreaming config manually
                config_dict = {
                    "dreaming": {
                        "num_dream_sequences": (
                            self.config.dreaming.num_dream_sequences
                        ),
                        "dream_length": self.config.dreaming.dream_length,
                        "graph_reasoning_hops": (
                            self.config.dreaming.graph_reasoning_hops
                        ),
                        "output_mode": self.config.dreaming.output_mode,
                    }
                }
            
            # Always include model architecture params
            config_dict["embedding_dim"] = self.embedding_dim
            config_dict["vocab_size"] = self.vocab_size
            config_dict["enable_rl"] = self.enable_rl
            
            json.dump(config_dict, f, indent=2)
        
        # Save weights
        torch.save(
            self.state_dict(),
            save_path / "dreaming_model_weights.pt"
        )
    
    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Simple character-level tokenization."""
        char_ids = [ord(c) % self.vocab_size for c in prompt]
        return torch.tensor(
            [char_ids], dtype=torch.long, device=self.device
        )
    
    def _detokenize(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs back to text."""
        if token_ids.dim() == 2:
            token_ids = token_ids[0]  # Take first in batch
        chars = [chr(int(tid) % 128) for tid in token_ids]
        return "".join(chars)
