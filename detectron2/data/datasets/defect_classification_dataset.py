"""
Dataset loader for defect classification with multi-label support.

Loads share_manifest.json, performs iterative stratification train/val split,
and provides a PyTorch Dataset with augmentation for training a multi-label
defect classifier.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DEFECT_CLASSES = [
    "bad_hands",
    "bad_feet",
    "bad_proportions",
    "backwards_joints",
    "extra_missing_parts",
    "merged_fused",
]
NUM_CLASSES = len(DEFECT_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(DEFECT_CLASSES)}


def load_manifest(manifest_path):
    """Load share_manifest.json and return list of (filename, multi-hot label) tuples."""
    with open(manifest_path, "r") as f:
        data = json.load(f)

    samples = []
    for entry in data:
        label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
        for defect in entry.get("defects", []):
            if defect in CLASS_TO_IDX:
                label_vec[CLASS_TO_IDX[defect]] = 1.0
        samples.append((entry["filename"], label_vec))
    return samples


def iterative_stratification_split(samples, train_ratio=0.8, seed=42):
    """
    Iterative stratification for multi-label data.

    Uses the algorithm from Sechidis et al. (2011) to produce balanced
    train/val splits across all label combinations.
    """
    from skmultilearn.model_selection import iterative_train_test_split

    rng = np.random.RandomState(seed)
    filenames = np.array([s[0] for s in samples]).reshape(-1, 1)
    labels = np.array([s[1] for s in samples])

    # Shuffle before split for reproducibility
    indices = rng.permutation(len(samples))
    filenames = filenames[indices]
    labels = labels[indices]

    test_size = 1.0 - train_ratio
    X_train, y_train, X_val, y_val = iterative_train_test_split(
        filenames, labels, test_size=test_size
    )

    train_samples = [
        (fn[0], lbl.tolist()) for fn, lbl in zip(X_train, y_train)
    ]
    val_samples = [
        (fn[0], lbl.tolist()) for fn, lbl in zip(X_val, y_val)
    ]
    return train_samples, val_samples


def iterative_stratification_train_val_test_split(
    samples, train_ratio=0.7, val_ratio=0.15, seed=42
):
    """
    Three-way iterative stratification split for multi-label data.

    Splits into train/val/test sets by first separating out the test set,
    then splitting the remainder into train and val.
    """
    from skmultilearn.model_selection import iterative_train_test_split

    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    rng = np.random.RandomState(seed)
    filenames = np.array([s[0] for s in samples]).reshape(-1, 1)
    labels = np.array([s[1] for s in samples])

    # Shuffle before split for reproducibility
    indices = rng.permutation(len(samples))
    filenames = filenames[indices]
    labels = labels[indices]

    # First split: separate test set from train+val
    X_trainval, y_trainval, X_test, y_test = iterative_train_test_split(
        filenames, labels, test_size=test_ratio
    )

    # Second split: separate val from train
    # val_ratio relative to the train+val portion
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X_trainval, y_trainval, test_size=val_ratio_adjusted
    )

    train_samples = [
        (fn[0], lbl.tolist()) for fn, lbl in zip(X_train, y_train)
    ]
    val_samples = [
        (fn[0], lbl.tolist()) for fn, lbl in zip(X_val, y_val)
    ]
    test_samples = [
        (fn[0], lbl.tolist()) for fn, lbl in zip(X_test, y_test)
    ]
    return train_samples, val_samples, test_samples


def compute_pos_weight(samples):
    """
    Compute per-class pos_weight for BCEWithLogitsLoss.

    pos_weight[c] = num_negative[c] / num_positive[c]
    """
    labels = np.array([s[1] for s in samples])
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    # Avoid division by zero
    pos_weight = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def get_class_distribution(samples):
    """Return per-class counts and multi-label stats."""
    labels = np.array([s[1] for s in samples])
    per_class = {
        DEFECT_CLASSES[i]: int(labels[:, i].sum()) for i in range(NUM_CLASSES)
    }
    num_multi = int((labels.sum(axis=1) > 1).sum())
    num_any_defect = int((labels.sum(axis=1) > 0).sum())
    num_clean = len(labels) - num_any_defect
    return {
        "total": len(labels),
        "clean": num_clean,
        "any_defect": num_any_defect,
        "multi_label": num_multi,
        "per_class": per_class,
    }


class DefectClassificationDataset(Dataset):
    """PyTorch Dataset for multi-label defect classification."""

    def __init__(self, samples, image_dir, transform=None):
        """
        Args:
            samples: list of (filename, multi_hot_label) tuples
            image_dir: path to directory containing images
            transform: torchvision transform pipeline
        """
        self.samples = samples
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = self.image_dir / filename
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.float32)
        return image, label


def build_train_transform():
    """Augmentation pipeline for training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(1024, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_val_transform():
    """Transform pipeline for validation (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(1024),
        transforms.CenterCrop(1024),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
