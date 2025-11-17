#!/usr/bin/env python3
"""
Test trained image tokenizer with sample images.

Usage:
    python scripts/test_image_tokenizer.py \
        --model ./models/image_tokenizer \
        --image ./data/coco/train2017/000000000009.jpg
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image_token_llm.config import ImageTokenizerConfig
from image_token_llm.vision_encoder import VisionEncoder


from image_token_llm.config import ImageTokenizerConfig
from image_token_llm.vision_encoder import VisionEncoder


def test_tokenizer(model_dir: str, image_path: str, device: str = "cpu"):
    """Test trained image tokenizer."""
    
    model_path = Path(model_dir)
    
    print("=" * 60)
    print("Image Tokenizer Test")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    config = ImageTokenizerConfig(embedding_dim=256, patch_size=16)
    encoder = VisionEncoder(config, backbone="lite")
    encoder.load_state_dict(
        torch.load(model_path / "image_tokenizer.pt", map_location=device)
    )
    encoder.to(device)
    encoder.eval()
    print("✓ Model loaded")
    
    # Load image
    print("\nLoading image...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    import torch
    if isinstance(image_tensor, torch.Tensor):
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
    print(f"✓ Image loaded: {image.size}")
    
    # Encode
    print("\nEncoding image...")
    with torch.no_grad():
        # For single image, just use encoder.forward
        embeddings = encoder(image_tensor)
    
    print(f"✓ Embeddings shape: {embeddings.shape}")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  Min value: {embeddings.min().item():.4f}")
    print(f"  Max value: {embeddings.max().item():.4f}")
    print(f"  Mean value: {embeddings.mean().item():.4f}")
    print(f"  Std value: {embeddings.std().item():.4f}")
    
    print("=" * 60)
    print("✓ Test complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Test trained image tokenizer"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to test image",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device to use (default: cpu)",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.model).exists():
        print(f"Error: Model directory not found: {args.model}")
        print("\nTrain a model first:")
        print("  python scripts/train_image_tokenizer.py \\")
        print("      --data ./data/coco/train2017 \\")
        print("      --output ./models/image_tokenizer")
        sys.exit(1)
    
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    test_tokenizer(
        model_dir=args.model,
        image_path=args.image,
        device=args.device,
    )


if __name__ == "__main__":
    main()
