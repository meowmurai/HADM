#!/usr/bin/env python3
"""
Prepare train/val splits for the defect classification dataset.

Usage:
    python tools/prepare_dataset.py \
        --manifest defect_training_dataset/share_manifest.json \
        --output-dir data/splits \
        --train-ratio 0.8
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detectron2.data.datasets.defect_classification_dataset import (
    DEFECT_CLASSES,
    compute_pos_weight,
    get_class_distribution,
    iterative_stratification_split,
    load_manifest,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/val splits for defect classification"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to share_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write train.json and val.json",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # Load manifest
    print(f"Loading manifest from {args.manifest}...")
    samples = load_manifest(args.manifest)
    print(f"Loaded {len(samples)} samples")

    # Print overall distribution
    print("\n=== Overall Distribution ===")
    dist = get_class_distribution(samples)
    print(f"  Total samples: {dist['total']}")
    print(f"  Clean (no defects): {dist['clean']}")
    print(f"  Any defect: {dist['any_defect']}")
    print(f"  Multi-label: {dist['multi_label']}")
    print("  Per-class counts:")
    for cls, count in dist["per_class"].items():
        pct = 100.0 * count / dist["total"]
        print(f"    {cls}: {count} ({pct:.1f}%)")

    # Perform iterative stratification split
    print(f"\nPerforming iterative stratification split "
          f"(train={args.train_ratio:.0%}, val={1 - args.train_ratio:.0%})...")
    train_samples, val_samples = iterative_stratification_split(
        samples, train_ratio=args.train_ratio, seed=args.seed
    )

    # Print split distributions
    for split_name, split_samples in [("Train", train_samples), ("Val", val_samples)]:
        print(f"\n=== {split_name} Distribution ({len(split_samples)} samples) ===")
        split_dist = get_class_distribution(split_samples)
        print(f"  Clean: {split_dist['clean']}")
        print(f"  Any defect: {split_dist['any_defect']}")
        print(f"  Multi-label: {split_dist['multi_label']}")
        print("  Per-class counts:")
        for cls, count in split_dist["per_class"].items():
            pct = 100.0 * count / split_dist["total"]
            print(f"    {cls}: {count} ({pct:.1f}%)")

    # Compute pos_weight for training set
    pos_weight = compute_pos_weight(train_samples)
    print("\n=== BCEWithLogitsLoss pos_weight (from train set) ===")
    for cls, w in zip(DEFECT_CLASSES, pos_weight.tolist()):
        print(f"  {cls}: {w:.3f}")

    # Save splits
    os.makedirs(args.output_dir, exist_ok=True)

    def save_split(split_samples, filename):
        records = [
            {"filename": fn, "labels": lbl}
            for fn, lbl in split_samples
        ]
        path = os.path.join(args.output_dir, filename)
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Saved {len(records)} samples to {path}")

    save_split(train_samples, "train.json")
    save_split(val_samples, "val.json")

    # Also save pos_weight for use in training
    pw_path = os.path.join(args.output_dir, "pos_weight.json")
    pw_data = {
        "classes": DEFECT_CLASSES,
        "pos_weight": pos_weight.tolist(),
    }
    with open(pw_path, "w") as f:
        json.dump(pw_data, f, indent=2)
    print(f"Saved pos_weight to {pw_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
