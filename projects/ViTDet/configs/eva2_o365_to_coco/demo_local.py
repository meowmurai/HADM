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

from omegaconf import OmegaConf

inference = OmegaConf.create()
inference.input_dir = "demo/images"
inference.output_dir = "demo/outputs"
# Enhanced inference for subtle artifact detection
# Multi-scale runs inference at multiple resolutions and merges results
inference.multiscale = True
inference.multiscale_scales = [0.5, 0.75, 1.0, 1.25, 1.5]
# Crop-based inference splits non-square images into overlapping crops
# to avoid heavy downscaling of portrait/landscape images
inference.crop_inference = True
inference.crop_overlap = 0.5
inference.crop_nms_thresh = 0.6
# Horizontal flip TTA — run flipped inference and merge for +2-5% recall
inference.flip_tta = True

dataloader.train.dataset = L(get_detection_dataset_dicts)(
    names=[
        "local_human_artifact_train_ALL",
        "local_human_artifact_CrowdHuman",
        "local_human_artifact_HCD",
        "local_human_artifact_MHP",
        "local_human_artifact_OCHuman",
        "local_human_artifact_FD",
        "local_human_artifact_coco_train",
    ], 
    filter_empty=False)

dataloader.train.mapper.augmentations = [
    L(T.RandomBrightness)(intensity_min=0.5, intensity_max=1.5),
    L(T.RandomContrast)(intensity_min=0.5, intensity_max=1.5),
    L(T.RandomSaturation)(intensity_min=0.5, intensity_max=1.5),
] + dataloader.train.mapper.augmentations 

[model.roi_heads.pop(k) for k in ["mask_in_features", "mask_pooler", "mask_head"]]

model.roi_heads.num_classes = 6
dataloader.train.total_batch_size = 4

train.init_checkpoint = (
    "/net/ivcfs5/mnt/data/kwang/adobe/detectron2/pretrained_models/eva02_L_coco_det_sys_o365.pth"
)
# arguments that don't exist for Cascade R-CNN
dataloader.test.dataset.names = "local_human_artifact_val_ALL"
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="local_human_artifact_val_ALL",
    output_dir="./cache/large_local_human_artifact_ALL_val",
)

dataloader.train.mapper.recompute_boxes = False

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

optimizer.lr=1e-5
optimizer.params.lr_factor_func = partial(get_vit_lr_with_custom_logic, lr_decay_rate=0.8, num_layers=24, fix_blocks=-1)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

train.max_iter = 250000

train.model_ema.enabled=True
train.model_ema.device="cuda"
train.model_ema.decay=0.9999
train.checkpointer.period = 10000

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(
        start_value=1,
        end_value=1,
    ),
    warmup_length=0.01,
    warmup_factor=0.001,
)

dataloader.test.num_workers=0

