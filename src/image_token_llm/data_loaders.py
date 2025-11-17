"""Data loaders for open-source multi-modal datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class VisualGenomeTripletDataset(Dataset):
    """
    Dataset loader for Visual Genome scene graphs.
    Converts scene graphs into (what, action, result) triplets.
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        image_size: int = 224,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.split = split
        self.max_samples = max_samples
        self.image_size = image_size

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"Loading Visual Genome {split} split...")
        try:
            self.dataset = load_dataset(
                "visual_genome",
                "scene_graphs_v1.2",
                split=split,
                cache_dir=cache_dir,
            )
            if max_samples:
                self.dataset = self.dataset.select(range(max_samples))
        except Exception as e:
            print(f"Warning: Could not load Visual Genome: {e}")
            self.dataset = []

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        image = item["image"]
        relationships = item.get("relationships", [])

        if not relationships:
            # Create dummy triplet
            return self._create_dummy_triplet(image)

        # Extract first relationship as triplet
        rel = relationships[0]
        subject = rel.get("subject", {}).get("name", "object")
        predicate = rel.get("predicate", "relates_to")
        obj = rel.get("object", {}).get("name", "object")

        # Transform image (use same image for all three components)
        if isinstance(image, Image.Image):
            img_tensor = self.transform(image)
        else:
            img_tensor = torch.randn(3, self.image_size, self.image_size)

        return {
            "what": img_tensor,
            "action": img_tensor,
            "result": img_tensor,
            "triplet": (subject, predicate, obj),
        }

    def _create_dummy_triplet(self, image: Any) -> Dict[str, Any]:
        if isinstance(image, Image.Image):
            img_tensor = self.transform(image)
        else:
            img_tensor = torch.randn(3, self.image_size, self.image_size)

        return {
            "what": img_tensor,
            "action": img_tensor,
            "result": img_tensor,
            "triplet": ("entity", "relates_to", "entity"),
        }


class COCOCaptionDataset(Dataset):
    """
    COCO Captions dataset adapted for triplet learning.
    Uses captions to synthesize (what, action, result) structure.
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        image_size: int = 224,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.split = split
        self.max_samples = max_samples
        self.image_size = image_size

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"Loading COCO {split} split...")
        try:
            # Use COCO captions
            year = "2017" if split == "train" else "2017"
            self.dataset = load_dataset(
                "HuggingFaceM4/COCO",
                split=split,
                cache_dir=cache_dir,
            )
            if max_samples:
                self.dataset = self.dataset.select(range(max_samples))
        except Exception as e:
            print(f"Warning: Could not load COCO: {e}")
            self.dataset = []

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        image = item.get("image", None)
        captions = item.get("sentences", [])

        caption = captions[0]["raw"] if captions else "An image"

        # Parse caption into triplet using NLP (spaCy), fallback to heuristic
        try:
            import spacy
            nlp = getattr(self, "_spacy_nlp", None)
            if nlp is None:
                nlp = spacy.load("en_core_web_sm")
                self._spacy_nlp = nlp
            doc = nlp(caption)
            subject = None
            action = None
            obj = None
            for token in doc:
                if token.dep_ in ("nsubj", "nsubjpass") and subject is None:
                    subject = token.text
                if token.pos_ == "VERB" and action is None:
                    action = token.lemma_
                if token.dep_ in ("dobj", "attr", "pobj") and obj is None:
                    obj = token.text
            # Fallbacks if not found
            subject = subject or (doc[0].text if len(doc) > 0 else "subject")
            action = action or (doc[1].lemma_ if len(doc) > 1 and doc[1].pos_ == "VERB" else "is")
            obj = obj or (doc[-1].text if len(doc) > 2 else "object")
        except Exception:
            # Fallback: simple heuristic
            words = caption.split()
            subject = words[0] if len(words) > 0 else "subject"
            action = words[1] if len(words) > 1 else "is"
            obj = words[2] if len(words) > 2 else "object"

        # Transform image
        if image and isinstance(image, Image.Image):
            img_tensor = self.transform(image)
        else:
            img_tensor = torch.randn(3, self.image_size, self.image_size)

        return {
            "what": img_tensor,
            "action": img_tensor,
            "result": img_tensor,
            "triplet": (subject, action, obj),
            "caption": caption,
        }


class ConceptNetTripletDataset(Dataset):
    """
    ConceptNet knowledge graph as triplet dataset.
    Provides pure symbolic triplets for graph reasoning.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        max_samples: Optional[int] = 10000,
    ) -> None:
        self.max_samples = max_samples
        self.triplets: List[Tuple[str, str, str]] = []

        if data_path and os.path.exists(data_path):
            self._load_conceptnet(data_path)
        else:
            self._create_synthetic_triplets()

    def _load_conceptnet(self, path: str) -> None:
        """Load ConceptNet triplets from file."""
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if self.max_samples and i >= self.max_samples:
                    break
                try:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        self.triplets.append((parts[0], parts[1], parts[2]))
                except Exception:
                    continue

    def _create_synthetic_triplets(self) -> None:
        """Create synthetic knowledge triplets."""
        templates = [
            ("person", "performs", "action"),
            ("object", "has_property", "attribute"),
            ("animal", "lives_in", "habitat"),
            ("tool", "used_for", "task"),
            ("concept", "related_to", "concept"),
        ]

        for i in range(self.max_samples or 1000):
            template = templates[i % len(templates)]
            self.triplets.append(template)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        return self.triplets[idx]


class SyntheticTripletDataset(Dataset):
    """
    Synthetic dataset for testing and development.
    Generates random image triplets with labels.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 224,
        embedding_dim: int = 512,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.embedding_dim = embedding_dim

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "what": torch.randn(3, self.image_size, self.image_size),
            "action": torch.randn(3, self.image_size, self.image_size),
            "result": torch.randn(3, self.image_size, self.image_size),
            "triplet": (f"entity_{idx}", "relates_to", f"entity_{idx+1}"),
        }


def create_dataloader(
    dataset_name: str = "synthetic",
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    **kwargs: Any,
) -> DataLoader:
    """
    Factory function to create dataloaders for various datasets.
    
    Args:
        dataset_name: One of ["synthetic", "visual_genome", "coco", "conceptnet"]
        split: "train" or "val"
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_samples: Maximum samples to load (for debugging)
    
    Returns:
        DataLoader instance
    """
    if dataset_name == "synthetic":
        dataset = SyntheticTripletDataset(
            num_samples=max_samples or 1000,
            **kwargs,
        )
    elif dataset_name == "visual_genome":
        dataset = VisualGenomeTripletDataset(
            split=split,
            max_samples=max_samples,
            **kwargs,
        )
    elif dataset_name == "coco":
        dataset = COCOCaptionDataset(
            split=split,
            max_samples=max_samples,
            **kwargs,
        )
    elif dataset_name == "conceptnet":
        # ConceptNet returns pure triplets, different collation needed
        dataset = ConceptNetTripletDataset(
            max_samples=max_samples,
            **kwargs,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,  # Simple dataset, no multiprocessing needed
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
