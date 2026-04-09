"""
Defect classifier model wrapping EVA-02 ViT backbone.

Provides:
- DefectClassifier: multi-label classification head on top of ViT backbone
- extract_backbone_weights: extract backbone state dict from detection checkpoint
- build_classifier: factory to construct classifier with pretrained backbone weights
"""

import logging
from functools import partial

import torch
import torch.nn as nn

from detectron2.modeling.backbone.vit import ViT

logger = logging.getLogger(__name__)

# Default ViT-L (EVA-02-Large) configuration matching HADM-L training config
VIT_L_CONFIG = dict(
    img_size=1024,
    patch_size=16,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4 * 2 / 3,
    drop_path_rate=0.3,
    window_size=16,
    window_block_indexes=(
        list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8))
        + list(range(9, 11)) + list(range(12, 14)) + list(range(15, 17))
        + list(range(18, 20)) + list(range(21, 23))
    ),
    use_act_checkpoint=False,
    use_rel_pos=False,
    rope=True,
    xattn=True,
)


class DefectClassifier(nn.Module):
    """Multi-label defect classifier wrapping an EVA-02 ViT backbone.

    Takes a full image and outputs logits for each defect class.
    """

    def __init__(self, backbone, num_classes=6, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        embed_dim = backbone._out_feature_channels[backbone._out_features[0]]
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        x = list(features.values())[0]  # (B, embed_dim, H/patch, W/patch)
        x = self.pool(x).flatten(1)     # (B, embed_dim)
        x = self.dropout(x)
        return self.head(x)             # (B, num_classes) raw logits


def extract_backbone_weights(checkpoint_path):
    """Extract standalone ViT weights from a Detectron2 detection checkpoint.

    The HADM-L checkpoint stores backbone weights under the `backbone.net.*`
    prefix (from SimpleFeaturePyramid wrapping). This strips that prefix so
    the weights match a standalone ViT state dict.

    Returns:
        dict: cleaned state dict mapping ViT parameter names to tensors.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint.get("model", checkpoint)

    prefix = "backbone.net."
    backbone_weights = {}
    skipped = 0
    for key, value in model_state.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            backbone_weights[new_key] = value
        else:
            skipped += 1

    logger.info(
        "Extracted %d backbone keys, skipped %d non-backbone keys",
        len(backbone_weights), skipped,
    )
    return backbone_weights


def build_classifier(checkpoint_path=None, num_classes=6, dropout=0.3, vit_config=None):
    """Build a DefectClassifier with optional pretrained backbone weights.

    Args:
        checkpoint_path: path to HADM-L detection checkpoint. If provided,
            backbone weights are extracted and loaded.
        num_classes: number of output classes (default: 6 defect types).
        dropout: dropout rate before classification head.
        vit_config: optional dict overriding VIT_L_CONFIG defaults.

    Returns:
        DefectClassifier instance.
    """
    config = dict(VIT_L_CONFIG)
    if vit_config:
        config.update(vit_config)

    backbone = ViT(**config)

    if checkpoint_path is not None:
        weights = extract_backbone_weights(checkpoint_path)
        result = backbone.load_state_dict(weights, strict=False)
        if result.missing_keys:
            logger.warning("Missing keys when loading backbone: %s", result.missing_keys)
        if result.unexpected_keys:
            logger.warning("Unexpected keys when loading backbone: %s", result.unexpected_keys)
        logger.info("Loaded backbone weights from %s", checkpoint_path)

    model = DefectClassifier(backbone, num_classes=num_classes, dropout=dropout)
    return model
