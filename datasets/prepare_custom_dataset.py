#!/usr/bin/env python3
"""
Dataset Preparation Script for HADM (Human Artifact Detection Model)

This script helps you prepare a custom training dataset from your own images
(both normal and AI-generated with deformities) for fine-tuning the HADM model.

It handles:
  1. Organizing images into the expected directory structure
  2. Generating annotation templates for labeling
  3. Validating annotations before training
  4. Splitting data into train/val sets

Expected annotation format (per-image JSON):
{
  "image": {
    "file_name": "image_001.jpg",
    "height": 1024,
    "width": 1024
  },
  "annotation": [
    {
      "body_parts": "hand",       # one of: face, torso, arm, leg, hand, feet
      "level": "severe",          # "mild" or "severe"
      "bbox": [x1, y1, x2, y2]   # absolute pixel coordinates (XYXY format)
    }
  ],
  "human": [
    {
      "tag": ["human with extra hand"],  # global artifact tags (see GLOBAL_TAGS below)
      "bbox": [x1, y1, x2, y2]          # bounding box around the whole person
    }
  ]
}

For normal (real) images with no artifacts, use empty annotation and human lists.

Usage:
  # Step 1: Organize images
  python datasets/prepare_custom_dataset.py organize \
      --source-dir /path/to/your/images \
      --dataset-dir datasets/human_artifact_dataset \
      --split custom_train \
      --normal-subdir normal \
      --deformity-subdir deformity

  # Step 2: Generate annotation templates
  python datasets/prepare_custom_dataset.py generate-templates \
      --dataset-dir datasets/human_artifact_dataset \
      --split custom_train

  # Step 3: After manual labeling, validate annotations
  python datasets/prepare_custom_dataset.py validate \
      --dataset-dir datasets/human_artifact_dataset \
      --split custom_train

  # Step 4: Create train/val split from a single labeled set
  python datasets/prepare_custom_dataset.py split \
      --dataset-dir datasets/human_artifact_dataset \
      --source-split custom_all \
      --train-split custom_train \
      --val-split custom_val \
      --val-ratio 0.15
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# --- Constants matching detectron2/data/datasets/human_artifact_dataset.py ---

LOCAL_BODY_PARTS = ["face", "torso", "arm", "leg", "hand", "feet"]
SEVERITY_LEVELS = ["mild", "severe"]

GLOBAL_TAGS = [
    "human missing arm",
    "human missing face",
    "human missing feet",
    "human missing hand",
    "human missing leg",
    "human missing torso",
    "human with extra arm",
    "human with extra face",
    "human with extra feet",
    "human with extra hand",
    "human with extra leg",
    "human with extra torso",
]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def ensure_jpeg(src_path, dst_path):
    """Convert image to JPEG if needed (HADM expects JPEG input)."""
    ext = os.path.splitext(src_path)[1].lower()
    dst_jpeg = os.path.splitext(dst_path)[0] + ".jpg"
    if ext in (".jpg", ".jpeg"):
        shutil.copy2(src_path, dst_jpeg)
    else:
        img = Image.open(src_path).convert("RGB")
        img.save(dst_jpeg, "JPEG", quality=95)
    return dst_jpeg


def cmd_organize(args):
    """Organize source images into the HADM directory structure."""
    dataset_dir = Path(args.dataset_dir)
    img_dir = dataset_dir / "images" / args.split
    anno_dir = dataset_dir / "annotations" / args.split

    img_dir.mkdir(parents=True, exist_ok=True)
    anno_dir.mkdir(parents=True, exist_ok=True)

    source = Path(args.source_dir)
    copied = 0

    # Collect all image files
    image_files = []
    if args.normal_subdir:
        normal_dir = source / args.normal_subdir
        if normal_dir.exists():
            for f in sorted(normal_dir.iterdir()):
                if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    image_files.append((f, "normal"))

    if args.deformity_subdir:
        deformity_dir = source / args.deformity_subdir
        if deformity_dir.exists():
            for f in sorted(deformity_dir.iterdir()):
                if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    image_files.append((f, "deformity"))

    # If no subdirs specified, just scan source_dir directly
    if not args.normal_subdir and not args.deformity_subdir:
        for f in sorted(source.iterdir()):
            if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_files.append((f, "unknown"))

    print(f"Found {len(image_files)} images to organize")

    for src_file, img_type in tqdm(image_files, desc="Organizing images"):
        dst_path = img_dir / src_file.name
        jpeg_path = ensure_jpeg(str(src_file), str(dst_path))
        copied += 1

    print(f"\nOrganized {copied} images into {img_dir}")
    print(f"Next: run 'generate-templates' to create annotation templates,")
    print(f"       then label the JSON files in {anno_dir}")


def cmd_generate_templates(args):
    """Generate annotation template JSON files for each image."""
    dataset_dir = Path(args.dataset_dir)
    img_dir = dataset_dir / "images" / args.split
    anno_dir = dataset_dir / "annotations" / args.split

    anno_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        print(f"Error: Image directory {img_dir} does not exist.")
        print("Run 'organize' first, or place images manually.")
        sys.exit(1)

    images = sorted([f for f in img_dir.iterdir()
                     if f.suffix.lower() in SUPPORTED_EXTENSIONS])
    print(f"Generating annotation templates for {len(images)} images...")

    generated = 0
    skipped = 0
    for img_path in tqdm(images, desc="Generating templates"):
        anno_path = anno_dir / (img_path.stem + ".json")

        # Skip if annotation already exists (don't overwrite manual labels)
        if anno_path.exists() and not args.overwrite:
            skipped += 1
            continue

        im = Image.open(img_path)
        width, height = im.size

        template = {
            "image": {
                "file_name": img_path.name,
                "height": height,
                "width": width,
                "tag": "unlabeled"
            },
            "annotation": [],
            "human": []
        }

        with open(anno_path, "w") as f:
            json.dump(template, f, indent=2)

        generated += 1

    print(f"\nGenerated {generated} templates, skipped {skipped} existing")
    print(f"\nAnnotation templates saved to: {anno_dir}")
    print(f"\n--- HOW TO LABEL ---")
    print(f"For NORMAL images (no artifacts): leave 'annotation' and 'human' as empty lists []")
    print(f"For images WITH LOCAL ARTIFACTS, add entries to 'annotation':")
    print(f"  {{\"body_parts\": \"<part>\", \"level\": \"<severity>\", \"bbox\": [x1, y1, x2, y2]}}")
    print(f"  body_parts: {LOCAL_BODY_PARTS}")
    print(f"  level: {SEVERITY_LEVELS}")
    print(f"  bbox: [x_min, y_min, x_max, y_max] in absolute pixels (XYXY format)")
    print(f"\nFor images WITH GLOBAL ARTIFACTS, add entries to 'human':")
    print(f"  {{\"tag\": [\"<global_tag>\", ...], \"bbox\": [x1, y1, x2, y2]}}")
    print(f"  tags: {GLOBAL_TAGS}")
    print(f"  bbox: bounding box around the entire person")
    print(f"\nAfter labeling, update the 'tag' field in 'image' from 'unlabeled' to 'labeled'")


def cmd_validate(args):
    """Validate that annotations are correctly formatted."""
    dataset_dir = Path(args.dataset_dir)
    img_dir = dataset_dir / "images" / args.split
    anno_dir = dataset_dir / "annotations" / args.split

    if not anno_dir.exists():
        print(f"Error: Annotation directory {anno_dir} does not exist.")
        sys.exit(1)

    images = sorted([f for f in img_dir.iterdir()
                     if f.suffix.lower() in SUPPORTED_EXTENSIONS])
    annos = sorted(anno_dir.glob("*.json"))

    print(f"Images: {len(images)}, Annotations: {len(annos)}")

    errors = []
    warnings = []
    stats = {
        "total": 0,
        "with_local_artifacts": 0,
        "with_global_artifacts": 0,
        "empty_annotations": 0,
        "unlabeled": 0,
        "local_by_part": {},
        "global_by_tag": {},
    }

    # Check all images have annotations
    img_stems = {f.stem for f in images}
    anno_stems = {f.stem for f in annos}
    missing_annos = img_stems - anno_stems
    if missing_annos:
        errors.append(f"{len(missing_annos)} images missing annotations: {list(missing_annos)[:5]}...")

    for anno_path in tqdm(annos, desc="Validating"):
        stats["total"] += 1
        try:
            with open(anno_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"{anno_path.name}: Invalid JSON - {e}")
            continue

        # Check required keys
        for key in ["image", "annotation", "human"]:
            if key not in data:
                errors.append(f"{anno_path.name}: Missing required key '{key}'")

        if "image" not in data:
            continue

        img_info = data["image"]
        if "height" not in img_info or "width" not in img_info:
            errors.append(f"{anno_path.name}: Missing height/width in image info")

        if img_info.get("tag") == "unlabeled":
            stats["unlabeled"] += 1

        h = img_info.get("height", 0)
        w = img_info.get("width", 0)

        # Validate local annotations
        local_annos = data.get("annotation", [])
        if local_annos:
            stats["with_local_artifacts"] += 1
        for i, anno in enumerate(local_annos):
            if "body_parts" not in anno:
                errors.append(f"{anno_path.name}: annotation[{i}] missing 'body_parts'")
            elif anno["body_parts"] not in LOCAL_BODY_PARTS:
                errors.append(f"{anno_path.name}: annotation[{i}] invalid body_parts '{anno['body_parts']}'. Must be one of {LOCAL_BODY_PARTS}")
            else:
                part = anno["body_parts"]
                stats["local_by_part"][part] = stats["local_by_part"].get(part, 0) + 1

            if "level" not in anno:
                errors.append(f"{anno_path.name}: annotation[{i}] missing 'level'")
            elif anno["level"] not in SEVERITY_LEVELS:
                errors.append(f"{anno_path.name}: annotation[{i}] invalid level '{anno['level']}'. Must be one of {SEVERITY_LEVELS}")

            if "bbox" not in anno:
                errors.append(f"{anno_path.name}: annotation[{i}] missing 'bbox'")
            else:
                bbox = anno["bbox"]
                if len(bbox) != 4:
                    errors.append(f"{anno_path.name}: annotation[{i}] bbox must have 4 values [x1,y1,x2,y2]")
                elif bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    warnings.append(f"{anno_path.name}: annotation[{i}] bbox has x1>=x2 or y1>=y2 (will be auto-swapped at load time)")

        # Validate global annotations
        global_annos = data.get("human", [])
        if global_annos:
            stats["with_global_artifacts"] += 1
        for i, human in enumerate(global_annos):
            if "tag" not in human:
                errors.append(f"{anno_path.name}: human[{i}] missing 'tag'")
            else:
                for tag in human["tag"]:
                    if tag not in GLOBAL_TAGS:
                        errors.append(f"{anno_path.name}: human[{i}] invalid tag '{tag}'. Must be one of {GLOBAL_TAGS}")
                    else:
                        stats["global_by_tag"][tag] = stats["global_by_tag"].get(tag, 0) + 1

            if "bbox" not in human:
                errors.append(f"{anno_path.name}: human[{i}] missing 'bbox'")
            else:
                bbox = human["bbox"]
                if len(bbox) != 4:
                    errors.append(f"{anno_path.name}: human[{i}] bbox must have 4 values [x1,y1,x2,y2]")

        if not local_annos and not global_annos:
            stats["empty_annotations"] += 1

    # Print report
    print(f"\n{'='*60}")
    print(f"VALIDATION REPORT for split: {args.split}")
    print(f"{'='*60}")
    print(f"Total annotations:        {stats['total']}")
    print(f"With local artifacts:      {stats['with_local_artifacts']}")
    print(f"With global artifacts:     {stats['with_global_artifacts']}")
    print(f"Empty (normal images):     {stats['empty_annotations']}")
    print(f"Still unlabeled:           {stats['unlabeled']}")

    if stats["local_by_part"]:
        print(f"\nLocal artifact counts by body part:")
        for part, count in sorted(stats["local_by_part"].items()):
            print(f"  {part}: {count}")

    if stats["global_by_tag"]:
        print(f"\nGlobal artifact counts by tag:")
        for tag, count in sorted(stats["global_by_tag"].items()):
            print(f"  {tag}: {count}")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings[:20]:
            print(f"  ⚠ {w}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings)-20} more")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors[:30]:
            print(f"  ✗ {e}")
        if len(errors) > 30:
            print(f"  ... and {len(errors)-30} more")
        print(f"\nValidation FAILED. Fix errors before training.")
        return False
    else:
        print(f"\nValidation PASSED.")
        if stats["unlabeled"] > 0:
            print(f"Note: {stats['unlabeled']} images are still unlabeled.")
        return True


def cmd_split(args):
    """Split a labeled dataset into train/val sets."""
    dataset_dir = Path(args.dataset_dir)
    src_img_dir = dataset_dir / "images" / args.source_split
    src_anno_dir = dataset_dir / "annotations" / args.source_split

    train_img_dir = dataset_dir / "images" / args.train_split
    train_anno_dir = dataset_dir / "annotations" / args.train_split
    val_img_dir = dataset_dir / "images" / args.val_split
    val_anno_dir = dataset_dir / "annotations" / args.val_split

    for d in [train_img_dir, train_anno_dir, val_img_dir, val_anno_dir]:
        d.mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in src_img_dir.iterdir()
                     if f.suffix.lower() in SUPPORTED_EXTENSIONS])

    random.seed(args.seed)
    random.shuffle(images)

    val_count = max(1, int(len(images) * args.val_ratio))
    val_images = images[:val_count]
    train_images = images[val_count:]

    print(f"Splitting {len(images)} images: {len(train_images)} train, {len(val_images)} val")

    for img_path in tqdm(train_images, desc="Train split"):
        shutil.copy2(img_path, train_img_dir / img_path.name)
        anno_src = src_anno_dir / (img_path.stem + ".json")
        if anno_src.exists():
            shutil.copy2(anno_src, train_anno_dir / (img_path.stem + ".json"))

    for img_path in tqdm(val_images, desc="Val split"):
        shutil.copy2(img_path, val_img_dir / img_path.name)
        anno_src = src_anno_dir / (img_path.stem + ".json")
        if anno_src.exists():
            shutil.copy2(anno_src, val_anno_dir / (img_path.stem + ".json"))

    print(f"\nTrain: {train_img_dir} ({len(train_images)} images)")
    print(f"Val:   {val_img_dir} ({len(val_images)} images)")


def main():
    parser = argparse.ArgumentParser(
        description="HADM Custom Dataset Preparation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # organize
    org = subparsers.add_parser("organize", help="Organize source images into HADM structure")
    org.add_argument("--source-dir", required=True, help="Directory containing your images")
    org.add_argument("--dataset-dir", default="datasets/human_artifact_dataset",
                     help="Root dataset directory (default: datasets/human_artifact_dataset)")
    org.add_argument("--split", default="custom_train", help="Split name (default: custom_train)")
    org.add_argument("--normal-subdir", default=None,
                     help="Subdirectory within source-dir containing normal images")
    org.add_argument("--deformity-subdir", default=None,
                     help="Subdirectory within source-dir containing deformity images")

    # generate-templates
    gen = subparsers.add_parser("generate-templates",
                                help="Generate annotation template JSONs for images")
    gen.add_argument("--dataset-dir", default="datasets/human_artifact_dataset")
    gen.add_argument("--split", required=True, help="Split name to generate templates for")
    gen.add_argument("--overwrite", action="store_true",
                     help="Overwrite existing annotation files")

    # validate
    val = subparsers.add_parser("validate", help="Validate annotation format and completeness")
    val.add_argument("--dataset-dir", default="datasets/human_artifact_dataset")
    val.add_argument("--split", required=True, help="Split name to validate")

    # split
    spl = subparsers.add_parser("split", help="Split dataset into train/val")
    spl.add_argument("--dataset-dir", default="datasets/human_artifact_dataset")
    spl.add_argument("--source-split", required=True, help="Source split to divide")
    spl.add_argument("--train-split", default="custom_train", help="Output train split name")
    spl.add_argument("--val-split", default="custom_val", help="Output val split name")
    spl.add_argument("--val-ratio", type=float, default=0.15, help="Fraction for validation")
    spl.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "organize": cmd_organize,
        "generate-templates": cmd_generate_templates,
        "validate": cmd_validate,
        "split": cmd_split,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
