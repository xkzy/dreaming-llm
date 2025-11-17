#!/usr/bin/env python3
"""
Download COCO dataset for image tokenizer training.

Usage:
    python scripts/download_coco.py --output ./data/coco --subset train
    python scripts/download_coco.py --output ./data/coco --subset val --limit 1000
"""

import argparse
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


COCO_URLS = {
    "train2017_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017_images": "http://images.cocodataset.org/zips/val2017.zip",
    "test2017_images": "http://images.cocodataset.org/zips/test2017.zip",
    "train2017_annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "val2017_annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download_progress(block_num, block_size, total_size):
    """Show download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, (downloaded / total_size) * 100)
        progress = int(percent / 2)
        bar = "=" * progress + " " * (50 - progress)
        logger.info(f"[{bar}] {percent:.1f}%")


def download_file(url: str, output_path: Path) -> None:
    """Download a file from URL to output path."""
    logger.info(f"Downloading from {url}...")
    logger.info(f"Saving to {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        urlretrieve(url, output_path, reporthook=download_progress)
        logger.info(f"✓ Download complete: {output_path}")
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        raise


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file."""
    logger.info(f"Extracting {zip_path.name}...")
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"✓ Extraction complete: {extract_to}")
    except Exception as e:
        logger.error(f"✗ Extraction failed: {e}")
        raise


def download_coco_subset(
    output_dir: str,
    subset: str = "train",
    download_images: bool = True,
    download_annotations: bool = True,
    keep_zip: bool = False,
    limit: Optional[int] = None,
) -> None:
    """
    Download COCO dataset subset.
    
    Args:
        output_dir: Directory to save dataset
        subset: Dataset subset ('train', 'val', or 'test')
        download_images: Whether to download images
        download_annotations: Whether to download annotations
        keep_zip: Keep zip files after extraction
        limit: Limit number of images (for testing)
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("COCO Dataset Downloader")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Subset: {subset}2017")
    logger.info(f"Download images: {download_images}")
    logger.info(f"Download annotations: {download_annotations}")
    if limit:
        logger.info(f"Image limit: {limit}")
    logger.info("=" * 70)
    
    # Download images
    if download_images:
        image_key = f"{subset}2017_images"
        if image_key not in COCO_URLS:
            logger.error(f"Invalid subset: {subset}")
            sys.exit(1)
        
        zip_path = output_path / f"{subset}2017.zip"
        
        if not zip_path.exists():
            download_file(COCO_URLS[image_key], zip_path)
        else:
            logger.info(f"✓ Already downloaded: {zip_path}")
        
        # Extract
        extract_zip(zip_path, output_path)
        
        # Remove zip if requested
        if not keep_zip:
            logger.info(f"Removing {zip_path.name}...")
            zip_path.unlink()
        
        # Apply limit if specified
        if limit:
            image_dir = output_path / f"{subset}2017"
            images = sorted(image_dir.glob("*.jpg"))
            logger.info(f"Found {len(images)} images, keeping first {limit}")
            
            for img in images[limit:]:
                img.unlink()
            
            logger.info(f"✓ Limited to {limit} images")
    
    # Download annotations
    if download_annotations and subset in ["train", "val"]:
        anno_key = f"{subset}2017_annotations"
        zip_path = output_path / "annotations_trainval2017.zip"
        
        if not zip_path.exists():
            download_file(COCO_URLS[anno_key], zip_path)
        else:
            logger.info(f"✓ Already downloaded: {zip_path}")
        
        # Extract
        extract_zip(zip_path, output_path)
        
        # Remove zip if requested
        if not keep_zip:
            logger.info(f"Removing {zip_path.name}...")
            zip_path.unlink()
    
    # Print summary
    logger.info("=" * 70)
    logger.info("Download Summary")
    logger.info("=" * 70)
    
    image_dir = output_path / f"{subset}2017"
    if image_dir.exists():
        num_images = len(list(image_dir.glob("*.jpg")))
        logger.info(f"Images: {num_images} files in {image_dir}")
    
    anno_dir = output_path / "annotations"
    if anno_dir.exists():
        num_annos = len(list(anno_dir.glob("*.json")))
        logger.info(f"Annotations: {num_annos} files in {anno_dir}")
    
    total_size = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    )
    logger.info(f"Total size: {total_size / (1024**3):.2f} GB")
    logger.info("=" * 70)
    logger.info("✓ Download complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Download COCO dataset for image tokenizer training"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/coco",
        help="Output directory (default: ./data/coco)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset subset to download (default: train)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image download",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Skip annotation download",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep zip files after extraction",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of images (for testing)",
    )
    
    args = parser.parse_args()
    
    download_coco_subset(
        output_dir=args.output,
        subset=args.subset,
        download_images=not args.no_images,
        download_annotations=not args.no_annotations,
        keep_zip=args.keep_zip,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
