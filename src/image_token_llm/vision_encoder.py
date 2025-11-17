"""Vision encoder for converting images into semantic tokens."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

try:  # Optional heavy dependency
    from transformers import CLIPModel, CLIPProcessor
except ImportError:  # pragma: no cover - optional dependency
    CLIPModel = None  # type: ignore[assignment]
    CLIPProcessor = None  # type: ignore[assignment]

from .config import ImageTokenizerConfig


class VisionEncoder(nn.Module):
    """
    Multi-modal vision encoder that converts images into semantic embeddings.
    Supports CNN (ResNet/lite) and transformer (CLIP) backbones.
    """

    def __init__(
        self,
        config: ImageTokenizerConfig,
        backbone: str = "clip",
        freeze_backbone: bool = False,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            config: ImageTokenizerConfig
            backbone: "clip", "resnet", or "lite"
            freeze_backbone: If True, disables gradient for backbone
            device: "cpu", "cuda", or torch.device
        """
        super().__init__()
        self.config = config
        self.backbone_type = backbone
        self.embedding_dim = config.embedding_dim
        self.device = torch.device(device)

        if backbone == "clip":
            if CLIPModel is None or CLIPProcessor is None:
                raise ImportError(
                    "transformers is required for the CLIP backbone. "
                    "Install it via `pip install transformers` or use the "
                    "`lite` backbone."
                )
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self.device)
            self.feature_dim = self.model.vision_model.config.projection_dim
            self.embedding_dim = self.feature_dim
            config.embedding_dim = self.feature_dim
        elif backbone == "resnet":
            self.model = resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V2
            ).to(self.device)
            self.model.fc = nn.Identity()
            self.feature_dim = 2048
            self.transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        elif backbone == "lite":
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, config.embedding_dim),
                nn.LayerNorm(config.embedding_dim),
                nn.GELU(),
            ).to(self.device)
            self.feature_dim = config.embedding_dim
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        if backbone == "clip":
            self.projector = nn.Sequential(
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Linear(512, 512),
            ).to(self.device)
        else:
            self.projector = nn.Sequential(
                nn.Linear(self.feature_dim, config.embedding_dim),
                nn.LayerNorm(config.embedding_dim),
                nn.GELU(),
                nn.Linear(config.embedding_dim, config.embedding_dim),
            ).to(self.device)

    def forward(
        self,
        images: torch.Tensor,
        device: str = None,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Encode images into semantic embeddings.
        Supports large batches and device selection.
        Args:
            images: Tensor [B, C, H, W] or [C, H, W]
            device: Optional override device
            batch_size: Batch size for chunked processing (for large B)
        Returns:
            Embeddings [B, embedding_dim]
        """
        dev = torch.device(device) if device is not None else self.device
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.to(dev)
        outputs = []
        n = images.shape[0]
        for i in range(0, n, batch_size):
            batch = images[i : i + batch_size]
            with torch.no_grad():
                if self.backbone_type == "clip":
                    feats = self.model.get_image_features(pixel_values=batch)
                elif self.backbone_type == "resnet":
                    feats = self.model(self.transform(batch))
                else:
                    feats = self.model(batch)
                emb = self.projector(feats)
                outputs.append(emb.cpu())
        return torch.cat(outputs, dim=0)


class TripletEncoder(nn.Module):
    """
    Encodes (what, action, result) triplets into a unified representation.
    Each component is encoded separately, then fused through attention.
    """

    def __init__(
        self,
        config: ImageTokenizerConfig,
        backbone: str = "clip",
    ) -> None:
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoder(config, backbone=backbone)
        
        # Use the vision encoder's actual embedding dimension
        # (CLIP may have forced config.embedding_dim to 512)
        embedding_dim = self.vision_encoder.embedding_dim
        
        # Multi-head attention for triplet fusion
        self.triplet_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            batch_first=True,
        )
        
        # Learnable role embeddings for what/action/result
        self.role_embeddings = nn.Parameter(
            torch.randn(3, embedding_dim) * 0.02
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        what_images: torch.Tensor,
        action_images: torch.Tensor,
        result_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a triplet of images into semantic tokens.
        
        Args:
            what_images: [B, C, H, W]
            action_images: [B, C, H, W]
            result_images: [B, C, H, W]
        
        Returns:
            fused_embedding: [B, embedding_dim] - unified representation
            component_embeddings: [B, 3, embedding_dim] - individual components
        """
        # Encode each component
        what_emb = self.vision_encoder(what_images)
        action_emb = self.vision_encoder(action_images)
        result_emb = self.vision_encoder(result_images)
        
        # Stack and add role embeddings
        components = torch.stack([what_emb, action_emb, result_emb], dim=1)
        components = components + self.role_embeddings.unsqueeze(0)
        
        # Attend across components
        fused, _ = self.triplet_attention(
            components, components, components
        )  # [B, 3, D]
        
        # Pool across triplet dimension
        fused_embedding = fused.mean(dim=1)  # [B, D]
        fused_embedding = self.fusion(fused_embedding)
        
        return fused_embedding, components


class ImageTokenizerV2(nn.Module):
    """Image tokenizer producing semantic tokens with graph-aware context."""

    def __init__(
        self,
        config: ImageTokenizerConfig,
        backbone: str = "clip",
    ) -> None:
        super().__init__()
        self.config = config
        self.triplet_encoder = TripletEncoder(config, backbone=backbone)
        
    def forward(
        self,
        what_images: torch.Tensor,
        action_images: torch.Tensor,
        result_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tokenize image triplets into embeddings.
        
        Returns:
            tokens: [B, embedding_dim]
        """
        tokens, _ = self.triplet_encoder(
            what_images,
            action_images,
            result_images,
        )
        return tokens

    def encode_batch(
        self,
        triplet_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Encode a batch of triplets.
        
        Args:
            triplet_batch: List of (what, action, result) tensors
        
        Returns:
            Stacked embeddings [N, embedding_dim]
        """
        embeddings = []
        for what, action, result in triplet_batch:
            # Add batch dimension if needed
            if what.dim() == 3:
                what = what.unsqueeze(0)
                action = action.unsqueeze(0)
                result = result.unsqueeze(0)
            
            token = self(what, action, result)
            embeddings.append(token.squeeze(0))
        
        return torch.stack(embeddings)
