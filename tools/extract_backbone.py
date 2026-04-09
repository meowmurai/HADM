#!/usr/bin/env python3
"""Extract EVA-02 ViT backbone weights from an HADM-L detection checkpoint.

Usage:
    python tools/extract_backbone.py --checkpoint HADM-L_0249999.pth --output weights/backbone_vit_l.pth
"""

import argparse
import logging
import os
import sys

import torch

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detectron2.modeling.classifier import extract_backbone_weights

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Extract ViT backbone weights from HADM-L detection checkpoint"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to HADM-L detection checkpoint (.pth)"
    )
    parser.add_argument(
        "--output", required=True, help="Path to save extracted backbone weights"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    backbone_weights = extract_backbone_weights(args.checkpoint)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(backbone_weights, args.output)
    logger.info("Saved %d backbone keys to %s", len(backbone_weights), args.output)


if __name__ == "__main__":
    main()
