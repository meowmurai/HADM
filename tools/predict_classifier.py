#!/usr/bin/env python3
"""Inference script for multi-label defect classifier.

Supports single image or batch directory inference with per-class thresholds.

Usage:
    # Single image
    python tools/predict_classifier.py \
        --checkpoint checkpoints/classifier/best_model.pth \
        --thresholds checkpoints/classifier/thresholds.json \
        --image path/to/test_image.jpg

    # Directory batch
    python tools/predict_classifier.py \
        --checkpoint checkpoints/classifier/best_model.pth \
        --thresholds checkpoints/classifier/thresholds.json \
        --image-dir path/to/test_images/ \
        --output results.json

    # Binary mode (any defect = defect)
    python tools/predict_classifier.py \
        --checkpoint checkpoints/classifier/best_model.pth \
        --thresholds checkpoints/classifier/thresholds.json \
        --image-dir path/to/test_images/ \
        --binary
"""

import argparse
import json
import logging
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detectron2.data.datasets.defect_classification_dataset import (
    DEFECT_CLASSES,
    NUM_CLASSES,
)
from detectron2.modeling.classifier import build_classifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def build_inference_transform():
    """Preprocessing pipeline matching validation transform."""
    return transforms.Compose([
        transforms.Resize(1024),
        transforms.CenterCrop(1024),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_model(checkpoint_path, device):
    """Load trained classifier from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_classifier(
        checkpoint_path=None, num_classes=NUM_CLASSES, dropout=0.0
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    logger.info(
        "Loaded model from %s (epoch %d, mAP=%.4f)",
        checkpoint_path,
        checkpoint.get("epoch", -1),
        checkpoint.get("mAP", 0.0),
    )
    return model


def load_thresholds(thresholds_path):
    """Load per-class thresholds from JSON file."""
    with open(thresholds_path, "r") as f:
        data = json.load(f)
    thresholds = data["thresholds"]
    logger.info("Loaded thresholds from %s", thresholds_path)
    return thresholds


@torch.no_grad()
def predict_image(model, image_path, transform, thresholds, device):
    """Run inference on a single image.

    Returns dict with per-class confidence and detection status.
    """
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    defects = {}
    has_defects = False
    for i, cls_name in enumerate(DEFECT_CLASSES):
        confidence = float(probs[i])
        threshold = thresholds.get(cls_name, 0.5)
        detected = confidence >= threshold
        if detected:
            has_defects = True
        defects[cls_name] = {
            "confidence": round(confidence, 4),
            "detected": detected,
        }

    return {
        "image": str(image_path),
        "has_defects": has_defects,
        "defects": defects,
    }


def collect_images(image_dir):
    """Collect all image files from a directory."""
    images = []
    for fname in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(os.path.join(image_dir, fname))
    return images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run defect classifier inference"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained model checkpoint (best_model.pth)",
    )
    parser.add_argument(
        "--thresholds", required=True,
        help="Path to thresholds.json from training",
    )
    parser.add_argument(
        "--image", default=None,
        help="Path to a single image for inference",
    )
    parser.add_argument(
        "--image-dir", default=None,
        help="Path to directory of images for batch inference",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save JSON results (default: print to stdout)",
    )
    parser.add_argument(
        "--binary", action="store_true",
        help="Binary mode: output only good/defect classification",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (default: auto-detect)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.image and not args.image_dir:
        logger.error("Must specify --image or --image-dir")
        sys.exit(1)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    model = load_model(args.checkpoint, device)
    thresholds = load_thresholds(args.thresholds)
    transform = build_inference_transform()

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        image_paths.extend(collect_images(args.image_dir))

    if not image_paths:
        logger.error("No images found")
        sys.exit(1)

    logger.info("Processing %d image(s)...", len(image_paths))

    results = []
    for img_path in image_paths:
        result = predict_image(model, img_path, transform, thresholds, device)
        if args.binary:
            result = {
                "image": result["image"],
                "classification": "defect" if result["has_defects"] else "good",
                "has_defects": result["has_defects"],
            }
        results.append(result)

    # Output
    output_data = results if len(results) > 1 else results[0]

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("Results saved to %s", args.output)
    else:
        print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
