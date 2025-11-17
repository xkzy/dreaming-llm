# COCO Dataset & Image Tokenizer Training Guide

This guide shows how to download the COCO dataset and train an image tokenizer.

## Quick Start

### Step 1: Download COCO Dataset (Small Sample for Testing)

```bash
# Download 1000 training images (~500MB)
python scripts/download_coco.py \
    --output ./data/coco \
    --subset train \
    --limit 1000

# Expected output:
# - ./data/coco/train2017/ (1000 images)
# - ./data/coco/annotations/ (JSON files)
```

### Step 2: Train Image Tokenizer

```bash
# Quick training (10 epochs, ~5-10 minutes on GPU)
python scripts/train_image_tokenizer.py \
    --data ./data/coco/train2017 \
    --output ./models/image_tokenizer \
    --epochs 10 \
    --batch-size 32 \
    --device cuda

# Or on CPU (slower)
python scripts/train_image_tokenizer.py \
    --data ./data/coco/train2017 \
    --output ./models/image_tokenizer \
    --epochs 5 \
    --batch-size 16 \
    --device cpu
```

### Step 3: Use Trained Tokenizer

```python
from image_token_llm.vision_encoder import VisionEncoder
import torch

# Load trained tokenizer
encoder = VisionEncoder(backbone="lite")
encoder.load_state_dict(
    torch.load("./models/image_tokenizer/image_tokenizer.pt")
)
encoder.eval()

# Use it
with torch.no_grad():
    embeddings = encoder.encode_triplet(what, action, result)
    print(f"Embeddings shape: {embeddings.shape}")  # (batch, 512)
```

## Full Dataset (Production)

### Download Full Training Set (~19GB)

```bash
# Full training set (118,287 images)
python scripts/download_coco.py \
    --output ./data/coco \
    --subset train

# This will take 20-30 minutes depending on connection
```

### Download Validation Set (~1GB)

```bash
# Validation set (5,000 images)
python scripts/download_coco.py \
    --output ./data/coco \
    --subset val
```

### Train on Full Dataset

```bash
# Production training (50 epochs, ~2-4 hours on GPU)
python scripts/train_image_tokenizer.py \
    --data ./data/coco/train2017 \
    --output ./models/image_tokenizer_full \
    --epochs 50 \
    --batch-size 64 \
    --device cuda \
    --backbone resnet
```

## Dataset Structure

After downloading, your directory will look like:

```
data/coco/
├── train2017/          # Training images
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ... (118,287 images for full set)
├── val2017/            # Validation images (if downloaded)
│   └── ... (5,000 images)
└── annotations/        # COCO annotations
    ├── instances_train2017.json
    ├── instances_val2017.json
    ├── captions_train2017.json
    └── ...
```

## Training Options

### Backbones

```bash
# Lite (fast, smaller model)
--backbone lite

# ResNet (better quality, slower)
--backbone resnet

# CLIP (best quality, requires transformers)
--backbone clip
```

### Device Selection

```bash
# GPU (10-50x faster)
--device cuda

# CPU (slower but works everywhere)
--device cpu
```

### Batch Size

```bash
# Small batch (4GB GPU)
--batch-size 16

# Medium batch (8GB GPU)
--batch-size 32

# Large batch (16GB+ GPU)
--batch-size 64
```

## Expected Training Times

| Dataset Size | Backbone | Device | Batch Size | Epochs | Time    |
|--------------|----------|--------|------------|--------|---------|
| 1K images    | lite     | GPU    | 32         | 10     | 5 min   |
| 1K images    | lite     | CPU    | 16         | 10     | 20 min  |
| 10K images   | resnet   | GPU    | 32         | 20     | 30 min  |
| 118K images  | resnet   | GPU    | 64         | 50     | 3 hours |
| 118K images  | clip     | GPU    | 32         | 50     | 6 hours |

## Using with Main Model

After training, integrate the tokenizer into your LLM:

```python
from image_token_llm.model import ImageTokenReasoningLLM
from image_token_llm.vision_encoder import VisionEncoder
import torch

# Load trained tokenizer
trained_encoder = VisionEncoder(backbone="lite")
trained_encoder.load_state_dict(
    torch.load("./models/image_tokenizer/image_tokenizer.pt")
)

# Create model with trained encoder
model = ImageTokenReasoningLLM(
    device="cuda",
    vision_backbone="lite",
    enable_rl=True,
)

# Replace the encoder
model.vision_encoder = trained_encoder

# Now use normally
output = model.generate(
    prompt="Describe this",
    image_triplets=[(what_img, action_img, result_img)],
)
```

## Monitoring Training

Training will show:
- Progress bar for each epoch
- Current loss per batch
- Average loss per epoch
- Final training summary

Example output:
```
Epoch 1/10
Training: 100%|████████| 32/32 [00:15<00:00, loss=0.234]
Epoch 1 - Average Loss: 0.2567

Epoch 10/10
Training: 100%|████████| 32/32 [00:14<00:00, loss=0.089]
Epoch 10 - Average Loss: 0.0912

Training Summary
====================
Final loss: 0.0912
Best loss: 0.0856
Model saved to: ./models/image_tokenizer
✓ Training complete!
```

## Troubleshooting

### "Data directory not found"
```bash
# Download dataset first
python scripts/download_coco.py --output ./data/coco --subset train --limit 1000
```

### "CUDA out of memory"
```bash
# Reduce batch size or use CPU
python scripts/train_image_tokenizer.py \
    --data ./data/coco/train2017 \
    --batch-size 8 \
    --device cpu
```

### "Download too slow"
- Try a different time of day
- Use `--limit 1000` for smaller sample
- Download subset: `--subset val` (smaller than train)

### "Training too slow"
- Use GPU: `--device cuda`
- Reduce epochs: `--epochs 5`
- Use lite backbone: `--backbone lite`
- Limit dataset: pass `--limit 5000` to training script

## Next Steps

1. **Train tokenizer on full dataset** for production quality
2. **Integrate with LLM** for multi-modal generation
3. **Fine-tune on domain-specific images** if needed
4. **Evaluate reconstruction quality** visually
5. **Export and deploy** with main model

## Advanced: Custom Dataset

To use your own images:

```bash
# Organize images in a directory
mkdir -p ./data/my_images
# ... copy your images ...

# Train on custom images
python scripts/train_image_tokenizer.py \
    --data ./data/my_images \
    --output ./models/custom_tokenizer \
    --epochs 20
```

Images should be:
- Format: JPG or PNG
- Size: Any (will be resized to 224x224)
- Content: Diverse scenes for better generalization

---

**Ready to start?**

```bash
# Download sample dataset
python scripts/download_coco.py --output ./data/coco --subset train --limit 1000

# Train tokenizer
python scripts/train_image_tokenizer.py --data ./data/coco/train2017 --epochs 10
```
