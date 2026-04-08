"""
Training config for HADM-L (Local) on a custom dataset.

This config fine-tunes from a pretrained HADM-L checkpoint on your custom
labeled data. It uses the custom_train/custom_val splits prepared by
datasets/prepare_custom_dataset.py.

Usage:
  python tools/lazyconfig_train_net.py \
      --config-file projects/ViTDet/configs/eva2_o365_to_coco/eva02_large_local_custom.py \
      --num-gpus=1 \
      train.eval_period=2000 \
      train.log_period=100 \
      train.output_dir=./outputs/eva02_large_local_custom \
      dataloader.evaluator.output_dir=cache/large_local_custom_val \
      dataloader.train.total_batch_size=4

  To fine-tune from existing HADM-L weights (recommended):
      train.init_checkpoint=pretrained_models/HADM-L_0249999.pth

  To train from EVA-02 base weights:
      train.init_checkpoint=pretrained_models/eva02_L_coco_det_sys_o365.pth
"""

from functools import partial

from ..common.coco_loader_lsj_1024 import dataloader
from .cascade_mask_rcnn_vitdet_b_100ep import (
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)
from detectron2.evaluation import COCOEvaluator
import detectron2.data.transforms as T

from detectron2.config import LazyCall as L
from fvcore.common.param_scheduler import *
from detectron2.solver import WarmupParamScheduler
from detectron2.data import get_detection_dataset_dicts

# --- Dataset: custom splits ---
# Uses your custom_train images + annotations prepared by prepare_custom_dataset.py
# You can also combine with the original training data by adding more names:
#   "local_human_artifact_train_ALL", "local_human_artifact_CrowdHuman", etc.
dataloader.train.dataset = L(get_detection_dataset_dicts)(
    names=[
        "local_human_artifact_custom_train",
    ],
    filter_empty=False)

# Data augmentation
dataloader.train.mapper.augmentations = [
    L(T.RandomBrightness)(intensity_min=0.5, intensity_max=1.5),
    L(T.RandomContrast)(intensity_min=0.5, intensity_max=1.5),
    L(T.RandomSaturation)(intensity_min=0.5, intensity_max=1.5),
] + dataloader.train.mapper.augmentations

[model.roi_heads.pop(k) for k in ["mask_in_features", "mask_pooler", "mask_head"]]

# 6 local artifact classes: face, torso, arm, leg, hand, feet
model.roi_heads.num_classes = 6
dataloader.train.total_batch_size = 4

# Fine-tune from pretrained HADM-L weights (recommended)
# Change to eva02_L_coco_det_sys_o365.pth for training from EVA-02 base
train.init_checkpoint = "pretrained_models/HADM-L_0249999.pth"

# Validation on custom val split
dataloader.test.dataset.names = "local_human_artifact_custom_val"
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="local_human_artifact_custom_val",
    output_dir="./cache/large_local_custom_val",
)

dataloader.train.mapper.recompute_boxes = False

# EVA-02-L backbone config
model.backbone.net.img_size = 1024
model.backbone.square_pad = 1024
model.backbone.net.patch_size = 16
model.backbone.net.window_size = 16
model.backbone.net.embed_dim = 1024
model.backbone.net.depth = 24
model.backbone.net.num_heads = 16
model.backbone.net.mlp_ratio = 4*2/3
model.backbone.net.use_act_checkpoint = True
model.backbone.net.drop_path_rate = 0.3

# 2, 5, 8, 11, 14, 17, 20, 23 for global attention
model.backbone.net.window_block_indexes = (
    list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + list(range(12, 14)) + list(range(15, 17)) + list(range(18, 20)) + list(range(21, 23))
)

def get_vit_lr_with_custom_logic(name, lr_decay_rate, num_layers, fix_blocks=-1):
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
        else:
            return get_vit_lr_decay_rate(name, lr_decay_rate, num_layers)

        if layer_id <= fix_blocks:
            return 0.0
    return get_vit_lr_decay_rate(name, lr_decay_rate, num_layers)

# Lower learning rate for fine-tuning from HADM checkpoint
optimizer.lr=5e-6
optimizer.params.lr_factor_func = partial(get_vit_lr_with_custom_logic, lr_decay_rate=0.8, num_layers=24, fix_blocks=-1)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

# Shorter training for fine-tuning (adjust based on dataset size)
train.max_iter = 50000

train.model_ema.enabled=True
train.model_ema.device="cuda"
train.model_ema.decay=0.9999
train.checkpointer.period = 5000

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(
        start_value=1,
        end_value=0.1,
    ),
    warmup_length=0.02,
    warmup_factor=0.001,
)

dataloader.test.num_workers=0
