#!/usr/bin/env python3
"""
Train image tokenizer on COCO dataset.

Usage:
    python scripts/train_image_tokenizer.py \
        --data ./data/coco/train2017 \
        --output ./models/image_tokenizer \
        --epochs 10 \
        --batch-size 32
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image_token_llm.vision_encoder import VisionEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class COCOImageDataset(Dataset):
    """Simple COCO image dataset for tokenizer training."""
    
    def __init__(
        self,
        image_dir: str,
        transform=None,
        limit: int = None,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Get all images
        self.image_paths = sorted(self.image_dir.glob("*.jpg"))
        
        if limit:
            self.image_paths = self.image_paths[:limit]
        
        logger.info(f"Loaded {len(self.image_paths)} images from {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            return image
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, 224, 224)


class ImageTokenizerTrainer:
    """Trainer for image tokenizer using reconstruction loss."""
    
    def __init__(
        self,
        encoder: VisionEncoder,
        device: str = "cuda",
        learning_rate: float = 1e-4,
    ):
        self.device = device
        self.encoder = encoder.to(device)
        
        # Create decoder for reconstruction
        self.decoder = self._create_decoder().to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
        )
        
        # Loss
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
    
    def _create_decoder(self) -> nn.Module:
        """Create a simple decoder for reconstruction."""
        return nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 3 * 224 * 224),
            nn.Tanh(),
        )
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0.0
        num_batches = 0
        
        with tqdm(dataloader, desc="Training") as pbar:
            for batch in pbar:
                images = batch.to(self.device)
                
                # Encode
                embeddings = self.encoder.encode_triplet(
                    what=images,
                    action=images,  # Use same image for simplicity
                    result=images,
                )
                
                # Decode
                reconstructed = self.decoder(embeddings)
                reconstructed = reconstructed.view(-1, 3, 224, 224)
                
                # Loss
                loss = self.criterion(reconstructed, images)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Track
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def save(self, output_dir: str):
        """Save trained tokenizer."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save encoder
        torch.save(
            self.encoder.state_dict(),
            output_path / "image_tokenizer.pt",
        )
        
        # Save decoder (for reconstruction)
        torch.save(
            self.decoder.state_dict(),
            output_path / "image_decoder.pt",
        )
        
        # Save config
        config = {
            "embedding_dim": 512,
            "backbone": self.encoder.backbone_name,
            "train_losses": self.train_losses,
        }
        
        with open(output_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✓ Model saved to {output_path}")


def train_image_tokenizer(
    data_dir: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    backbone: str = "lite",
    limit: int = None,
) -> None:
    """
    Train image tokenizer on COCO dataset.
    
    Args:
        data_dir: Directory containing images
        output_dir: Directory to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use ('cuda' or 'cpu')
        backbone: Vision backbone ('lite', 'resnet', or 'clip')
        limit: Limit number of images (for testing)
    """
    
    logger.info("=" * 70)
    logger.info("Image Tokenizer Training")
    logger.info("=" * 70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Device: {device}")
    logger.info(f"Backbone: {backbone}")
    if limit:
        logger.info(f"Image limit: {limit}")
    logger.info("=" * 70)
    
    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    dataset = COCOImageDataset(
        image_dir=data_dir,
        transform=transform,
        limit=limit,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == "cuda" else False,
    )
    
    # Create encoder
    logger.info("Initializing vision encoder...")
    encoder = VisionEncoder(backbone=backbone)
    
    # Create trainer
    trainer = ImageTokenizerTrainer(
        encoder=encoder,
        device=device,
        learning_rate=learning_rate,
    )
    
    # Train
    logger.info("Starting training...")
    for epoch in range(1, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")
        
        avg_loss = trainer.train_epoch(dataloader)
        
        logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    
    # Save
    logger.info("\nSaving trained model...")
    trainer.save(output_dir)
    
    # Summary
    logger.info("=" * 70)
    logger.info("Training Summary")
    logger.info("=" * 70)
    logger.info(f"Final loss: {trainer.train_losses[-1]:.4f}")
    logger.info(f"Best loss: {min(trainer.train_losses):.4f}")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 70)
    logger.info("✓ Training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train image tokenizer on COCO dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Directory containing COCO images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/image_tokenizer",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="lite",
        choices=["lite", "resnet", "clip"],
        help="Vision backbone (default: lite)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of images (for testing)",
    )
    
    args = parser.parse_args()
    
    # Check data directory
    if not Path(args.data).exists():
        logger.error(f"Data directory not found: {args.data}")
        logger.info("\nDownload COCO dataset first:")
        logger.info("  python scripts/download_coco.py --output ./data/coco --subset train --limit 1000")
        sys.exit(1)
    
    train_image_tokenizer(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        backbone=args.backbone,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
