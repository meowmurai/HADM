#!/usr/bin/env python3
"""
Convert defect_training_dataset (image-level classification labels) into HADM
annotation format (per-artifact bounding boxes).

Source format (share_manifest.json):
  {"filename": "img.jpg", "label": "good"|"defect", "defects": ["bad_hands", ...]}

Target format (per-image JSON):
  {"image": {...}, "annotation": [{body_parts, level, bbox}], "human": [{tag, bbox}]}

Because the source dataset has NO bounding boxes, this script uses the full image
as a single person bbox and maps defect categories to HADM body parts/tags.
This is a "weak annotation" approach — suitable for initial training but can be
improved later with VLM-generated or manual bounding boxes.

Category mapping:
  bad_hands          -> local: hand (severe)
  bad_feet           -> local: feet (severe)
  bad_proportions    -> local: torso (mild)
  backwards_joints   -> local: leg (severe)
  extra_missing_parts-> global: "human with extra arm" (+ local: arm severe)
  merged_fused       -> local: torso (severe)

Usage:
  # Convert and copy images into HADM structure
  python3 datasets/convert_defect_dataset.py \
      --manifest defect_training_dataset/share_manifest.json \
      --image-dir defect_training_dataset/images \
      --output-dir datasets/human_artifact_dataset \
      --split custom_all

  # Then split into train/val with the existing tool:
  python3 datasets/prepare_custom_dataset.py split \
      --source-split custom_all \
      --train-split custom_train \
      --val-split custom_val

  # Validate:
  python3 datasets/prepare_custom_dataset.py validate --split custom_train
"""

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

# Mapping from source defect categories to HADM local annotations
DEFECT_TO_LOCAL = {
    "bad_hands": {"body_parts": "hand", "level": "severe"},
    "bad_feet": {"body_parts": "feet", "level": "severe"},
    "bad_proportions": {"body_parts": "torso", "level": "mild"},
    "backwards_joints": {"body_parts": "leg", "level": "severe"},
    "extra_missing_parts": {"body_parts": "arm", "level": "severe"},
    "merged_fused": {"body_parts": "torso", "level": "severe"},
}

# Defect categories that also produce global annotations
DEFECT_TO_GLOBAL_TAGS = {
    "extra_missing_parts": ["human with extra arm"],
    "merged_fused": ["human with extra torso"],
}


def get_jpeg_dimensions(filepath):
    """Get JPEG image dimensions without PIL (stdlib only)."""
    with open(filepath, "rb") as f:
        data = f.read()

    if data[:2] != b"\xff\xd8":
        return None, None

    i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        i += 2

        # Skip padding bytes
        if marker == 0xFF:
            continue
        # No-payload markers
        if marker in (0x00, 0x01, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9):
            continue

        if i + 2 > len(data):
            break
        length = struct.unpack(">H", data[i : i + 2])[0]

        # SOF markers contain dimensions
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            if i + 7 <= len(data):
                height = struct.unpack(">H", data[i + 3 : i + 5])[0]
                width = struct.unpack(">H", data[i + 5 : i + 7])[0]
                return width, height

        i += length

    return None, None


def get_image_dimensions(filepath):
    """Get image dimensions (JPEG only for now, stdlib-only)."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return get_jpeg_dimensions(filepath)
    # For non-JPEG, try PIL if available, otherwise return None
    try:
        from PIL import Image

        im = Image.open(filepath)
        return im.size
    except ImportError:
        return None, None


def convert_entry(entry, image_path):
    """Convert a single manifest entry to HADM annotation format."""
    width, height = get_image_dimensions(image_path)
    if width is None or height is None:
        # Fallback: assume 1024x1024 if we can't read dimensions
        width, height = 1024, 1024

    annotation_json = {
        "image": {
            "file_name": os.path.basename(image_path),
            "height": height,
            "width": width,
            "tag": "labeled",
        },
        "annotation": [],
        "human": [],
    }

    if entry["label"] == "good" or not entry["defects"]:
        return annotation_json

    # Full-image bbox as weak annotation (no precise boxes available)
    full_bbox = [0, 0, width, height]

    # Collect unique local annotations (avoid duplicates for same body part)
    seen_parts = set()
    global_tags = []

    for defect in entry["defects"]:
        # Local annotation
        if defect in DEFECT_TO_LOCAL:
            mapping = DEFECT_TO_LOCAL[defect]
            part = mapping["body_parts"]
            if part not in seen_parts:
                seen_parts.add(part)
                annotation_json["annotation"].append(
                    {
                        "body_parts": part,
                        "level": mapping["level"],
                        "bbox": full_bbox[:],
                    }
                )

        # Global annotation tags
        if defect in DEFECT_TO_GLOBAL_TAGS:
            global_tags.extend(DEFECT_TO_GLOBAL_TAGS[defect])

    # Add a single human entry with all global tags if any
    if global_tags:
        annotation_json["human"].append(
            {"tag": list(set(global_tags)), "bbox": full_bbox[:]}
        )
    elif annotation_json["annotation"]:
        # Even without explicit global tags, add a human entry so the global
        # detector also learns from this data
        first_defect = entry["defects"][0]
        part = DEFECT_TO_LOCAL.get(first_defect, {}).get("body_parts", "arm")
        annotation_json["human"].append(
            {"tag": [f"human with extra {part}"], "bbox": full_bbox[:]}
        )

    return annotation_json


def main():
    parser = argparse.ArgumentParser(
        description="Convert defect_training_dataset to HADM annotation format"
    )
    parser.add_argument(
        "--manifest",
        default="defect_training_dataset/share_manifest.json",
        help="Path to share_manifest.json",
    )
    parser.add_argument(
        "--image-dir",
        default="defect_training_dataset/images",
        help="Directory containing source images",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/human_artifact_dataset",
        help="Root HADM dataset directory",
    )
    parser.add_argument(
        "--split",
        default="custom_all",
        help="Target split name (default: custom_all)",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying images (saves disk space)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing files",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    if not manifest_path.exists():
        print(f"Error: manifest not found at {manifest_path}")
        sys.exit(1)
    if not image_dir.exists():
        print(f"Error: image directory not found at {image_dir}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Loaded manifest: {len(manifest)} entries")

    # Stats
    stats = {
        "total": len(manifest),
        "good": 0,
        "defect": 0,
        "missing_images": 0,
        "defect_types": {},
    }

    for entry in manifest:
        if entry["label"] == "good":
            stats["good"] += 1
        else:
            stats["defect"] += 1
        for d in entry["defects"]:
            stats["defect_types"][d] = stats["defect_types"].get(d, 0) + 1

    print(f"  Good: {stats['good']}, Defect: {stats['defect']}")
    print(f"  Defect breakdown: {json.dumps(stats['defect_types'], indent=4)}")

    if args.dry_run:
        print("\n[Dry run] No files written.")
        return

    # Create output directories
    img_out = output_dir / "images" / args.split
    anno_out = output_dir / "annotations" / args.split
    img_out.mkdir(parents=True, exist_ok=True)
    anno_out.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    for idx, entry in enumerate(manifest):
        src_img = image_dir / entry["filename"]
        if not src_img.exists():
            stats["missing_images"] += 1
            skipped += 1
            continue

        dst_img = img_out / entry["filename"]

        # Copy or symlink image
        if not dst_img.exists():
            if args.symlink:
                dst_img.symlink_to(src_img.resolve())
            else:
                shutil.copy2(src_img, dst_img)

        # Generate annotation
        anno_data = convert_entry(entry, str(src_img))
        anno_path = anno_out / (Path(entry["filename"]).stem + ".json")
        with open(anno_path, "w") as f:
            json.dump(anno_data, f, indent=2)

        converted += 1

        if (idx + 1) % 500 == 0:
            print(f"  Progress: {idx + 1}/{len(manifest)}")

    print(f"\nConverted {converted} images, skipped {skipped}")
    if stats["missing_images"]:
        print(f"  Warning: {stats['missing_images']} images not found in {image_dir}")

    print(f"\nOutput:")
    print(f"  Images:      {img_out}")
    print(f"  Annotations: {anno_out}")
    print(f"\nNext steps:")
    print(f"  1. Split into train/val:")
    print(
        f"     python3 datasets/prepare_custom_dataset.py split --source-split {args.split} --train-split custom_train --val-split custom_val"
    )
    print(f"  2. Validate:")
    print(
        f"     python3 datasets/prepare_custom_dataset.py validate --split custom_train"
    )
    print(f"  3. Train (local artifacts):")
    print(
        f"     python3 tools/lazyconfig_train_net.py --config-file projects/ViTDet/configs/eva2_o365_to_coco/eva02_large_local_custom.py --num-gpus=1 train.output_dir=./outputs/custom_local"
    )
    print(f"\nNote: Bounding boxes are set to full-image (weak annotation).")
    print(
        f"For better results, replace full-image bboxes with precise VLM-generated boxes."
    )


if __name__ == "__main__":
    main()
