# Data Directory

This directory stores datasets for training and experimentation.

## COCO Dataset

The COCO (Common Objects in Context) dataset is used for training the image tokenizer.

### Download COCO Dataset

```bash
# Quick sample (1,000 images, ~500MB)
python scripts/download_coco.py \
    --output ./data/coco \
    --subset train \
    --limit 1000

# Full training set (118,287 images, ~19GB)
python scripts/download_coco.py \
    --output ./data/coco \
    --subset train

# Validation set (5,000 images, ~1GB)
python scripts/download_coco.py \
    --output ./data/coco \
    --subset val
```

### Expected Structure

```
data/
├── coco/
│   ├── train2017/          # Training images
│   ├── val2017/            # Validation images (optional)
│   └── annotations/        # COCO annotations
└── custom/                 # Your custom datasets
```

## Custom Datasets

Add your own image datasets here for training:

```bash
data/
├── custom/
│   ├── medical_images/
│   ├── satellite_images/
│   └── product_images/
```

Train on custom data:

```bash
python scripts/train_image_tokenizer.py \
    --data ./data/custom/my_images \
    --output ./models/custom_tokenizer
```

## Dataset Guidelines

- **Format:** JPG or PNG images
- **Size:** Any size (will be resized to 224x224)
- **Organization:** Place all images in a single directory
- **Naming:** Any naming convention works
- **Diversity:** More diverse images = better tokenizer

## Git Ignore

Large datasets are excluded from git:
- `data/coco/` - COCO dataset files
- `data/*.jpg`, `data/*.png` - Image files
- `data/*.zip` - Downloaded archives

Only small fixture files (<1MB) should be committed.

## See Also

- **[COCO_TRAINING.md](../docs/COCO_TRAINING.md)** - Complete training guide
- **[scripts/download_coco.py](../scripts/download_coco.py)** - Download script
- **[scripts/train_image_tokenizer.py](../scripts/train_image_tokenizer.py)** - Training script
