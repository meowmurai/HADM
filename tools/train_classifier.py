#!/usr/bin/env python3
"""Two-phase training script for multi-label defect classifier.

Phase 1: Head-only training with frozen backbone.
Phase 2: End-to-end fine-tuning with layer-wise LR decay and early stopping.

Single-GPU usage:
    python tools/train_classifier.py \
        --backbone-weights weights/backbone_vit_l.pth \
        --data-dir data/splits \
        --output-dir checkpoints/classifier \
        --phase1-epochs 15 \
        --phase2-epochs 40 \
        --batch-size 8

Multi-GPU usage (e.g. 3 GPUs):
    torchrun --nproc_per_node=3 tools/train_classifier.py \
        --backbone-weights weights/backbone_vit_l.pth \
        --data-dir data/splits \
        --output-dir checkpoints/classifier \
        --phase1-epochs 15 \
        --phase2-epochs 40 \
        --batch-size 8
"""

import argparse
import copy
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detectron2.data.datasets.defect_classification_dataset import (
    DEFECT_CLASSES,
    NUM_CLASSES,
    DefectClassificationDataset,
    build_train_transform,
    build_val_transform,
)
from detectron2.modeling.classifier import build_classifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_dist_available():
    """Return True when launched via torchrun / torch.distributed.launch."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_available() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_available() else 1


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    """Initialize the process group when launched with torchrun."""
    if "RANK" not in os.environ:
        return  # single-GPU run
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_distributed():
    if is_dist_available():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply(self, model):
        """Load shadow weights into model (for evaluation)."""
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_split(path):
    """Load a JSON split file and return list of (filename, label) tuples."""
    with open(path, "r") as f:
        records = json.load(f)
    return [(r["filename"], r["labels"]) for r in records]


def load_pos_weight(data_dir):
    """Load pos_weight.json saved by prepare_dataset.py."""
    pw_path = os.path.join(data_dir, "pos_weight.json")
    with open(pw_path, "r") as f:
        data = json.load(f)
    return torch.tensor(data["pos_weight"], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run evaluation and return metrics dict."""
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Gather from all ranks for consistent metrics
    if is_dist_available():
        gathered_logits = [torch.zeros_like(all_logits) for _ in range(get_world_size())]
        gathered_labels = [torch.zeros_like(all_labels) for _ in range(get_world_size())]
        dist.all_gather(gathered_logits, all_logits)
        dist.all_gather(gathered_labels, all_labels)
        all_logits = torch.cat(gathered_logits, dim=0)
        all_labels = torch.cat(gathered_labels, dim=0)

    all_logits = all_logits.numpy()
    all_labels = all_labels.numpy()
    probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid

    # Per-class AP and mAP
    per_class_ap = {}
    for i, cls_name in enumerate(DEFECT_CLASSES):
        if all_labels[:, i].sum() > 0:
            per_class_ap[cls_name] = float(
                average_precision_score(all_labels[:, i], probs[:, i])
            )
        else:
            per_class_ap[cls_name] = 0.0
    mAP = float(np.mean(list(per_class_ap.values())))

    # Optimal per-class thresholds (maximize F1)
    thresholds = {}
    for i, cls_name in enumerate(DEFECT_CLASSES):
        best_f1 = 0.0
        best_t = 0.5
        for t in np.arange(0.1, 0.91, 0.05):
            preds = (probs[:, i] >= t).astype(np.float32)
            f1 = float(f1_score(all_labels[:, i], preds, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[cls_name] = round(best_t, 2)

    # Per-class F1 at optimal thresholds
    per_class_f1 = {}
    pred_matrix = np.zeros_like(probs)
    for i, cls_name in enumerate(DEFECT_CLASSES):
        t = thresholds[cls_name]
        pred_matrix[:, i] = (probs[:, i] >= t).astype(np.float32)
        per_class_f1[cls_name] = float(
            f1_score(all_labels[:, i], pred_matrix[:, i], zero_division=0)
        )

    # Exact match ratio
    exact_match = float(np.all(pred_matrix == all_labels, axis=1).mean())

    # Hamming loss
    hamming = float(np.mean(pred_matrix != all_labels))

    return {
        "loss": total_loss / max(n_batches, 1),
        "mAP": mAP,
        "per_class_ap": per_class_ap,
        "per_class_f1": per_class_f1,
        "thresholds": thresholds,
        "exact_match_ratio": exact_match,
        "hamming_loss": hamming,
    }


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, ema=None):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    # For grad clipping, get the underlying model params when using DDP
    params = model.module.parameters() if isinstance(model, DDP) else model.parameters()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        if ema is not None:
            ema.update(model.module if isinstance(model, DDP) else model)

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    # Average loss across ranks for consistent logging
    if is_dist_available():
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

    return avg_loss


def build_phase2_optimizer(model, lr_backbone, lr_head, weight_decay, layer_decay):
    """Build AdamW optimizer with layer-wise LR decay for backbone."""
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append((name, param))
        else:
            head_params.append((name, param))

    # Group backbone params by block index for layer-wise decay
    # ViT blocks are named backbone.blocks.{idx}.*
    param_groups = []

    # Find max block index
    max_block = 0
    for name, _ in backbone_params:
        parts = name.split(".")
        if "blocks" in parts:
            idx = int(parts[parts.index("blocks") + 1])
            max_block = max(max_block, idx)

    num_layers = max_block + 1

    # Non-block backbone params (patch_embed, pos_embed, etc.) get lowest LR
    non_block_params = []
    block_params = {i: [] for i in range(num_layers)}

    for name, param in backbone_params:
        parts = name.split(".")
        if "blocks" in parts:
            idx = int(parts[parts.index("blocks") + 1])
            block_params[idx].append(param)
        else:
            non_block_params.append(param)

    # Non-block params: deepest decay
    if non_block_params:
        scale = layer_decay ** num_layers
        param_groups.append({
            "params": non_block_params,
            "lr": lr_backbone * scale,
            "weight_decay": weight_decay,
        })

    # Block params: later blocks get higher LR
    for i in range(num_layers):
        if block_params[i]:
            scale = layer_decay ** (num_layers - 1 - i)
            param_groups.append({
                "params": block_params[i],
                "lr": lr_backbone * scale,
                "weight_decay": weight_decay,
            })

    # Head params
    head_p = [p for _, p in head_params]
    if head_p:
        param_groups.append({
            "params": head_p,
            "lr": lr_head,
            "weight_decay": weight_decay,
        })

    return torch.optim.AdamW(param_groups)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-label defect classifier")
    parser.add_argument("--backbone-weights", type=str, default=None,
                        help="Path to extracted backbone weights (.pth)")
    parser.add_argument("--detection-checkpoint", type=str, default=None,
                        help="Path to full HADM-L detection checkpoint (alternative to --backbone-weights)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing train.json, val.json, pos_weight.json")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory containing images (default: inferred from data-dir)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save checkpoints and thresholds")

    # Phase 1 hyperparams
    parser.add_argument("--phase1-epochs", type=int, default=15,
                        help="Number of epochs for head-only training (default: 15)")
    parser.add_argument("--phase1-lr", type=float, default=1e-3,
                        help="Learning rate for phase 1 (default: 1e-3)")

    # Phase 2 hyperparams
    parser.add_argument("--phase2-epochs", type=int, default=40,
                        help="Max epochs for end-to-end fine-tuning (default: 40)")
    parser.add_argument("--phase2-lr-backbone", type=float, default=1e-5,
                        help="Backbone learning rate for phase 2 (default: 1e-5)")
    parser.add_argument("--phase2-lr-head", type=float, default=1e-4,
                        help="Head learning rate for phase 2 (default: 1e-4)")
    parser.add_argument("--layer-decay", type=float, default=0.8,
                        help="Layer-wise LR decay factor (default: 0.8)")
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                        help="Early stopping patience in epochs (default: 5)")

    # General hyperparams
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="Weight decay (default: 0.05)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout before classification head (default: 0.3)")
    parser.add_argument("--ema-decay", type=float, default=0.9999,
                        help="EMA decay rate (default: 0.9999)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto-detect)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Distributed setup (no-op when run without torchrun)
    setup_distributed()

    # Seed (offset per rank for data diversity)
    torch.manual_seed(args.seed + get_rank())
    np.random.seed(args.seed + get_rank())

    # Device
    if args.device:
        device = torch.device(args.device)
    elif is_dist_available():
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if is_main_process():
        logger.info("Using device: %s (world_size=%d)", device, get_world_size())

    # Output dir & TensorBoard (rank 0 only)
    os.makedirs(args.output_dir, exist_ok=True)
    writer = None
    if is_main_process():
        tb_dir = os.path.join(args.output_dir, "tensorboard")
        writer = SummaryWriter(log_dir=tb_dir)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_samples = load_split(os.path.join(args.data_dir, "train.json"))
    val_samples = load_split(os.path.join(args.data_dir, "val.json"))
    pos_weight = load_pos_weight(args.data_dir).to(device)
    logger.info("Train: %d samples, Val: %d samples", len(train_samples), len(val_samples))

    image_dir = args.image_dir or os.path.join(
        os.path.dirname(args.data_dir.rstrip("/")), "defect_training_dataset", "images"
    )

    train_ds = DefectClassificationDataset(
        train_samples, image_dir, transform=build_train_transform()
    )
    val_ds = DefectClassificationDataset(
        val_samples, image_dir, transform=build_val_transform()
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist_available() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_dist_available() else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    checkpoint_path = args.detection_checkpoint  # build_classifier extracts backbone
    if args.backbone_weights:
        # Load pre-extracted backbone weights directly
        model = build_classifier(
            checkpoint_path=None, num_classes=NUM_CLASSES, dropout=args.dropout
        )
        backbone_weights = torch.load(args.backbone_weights, map_location="cpu")
        result = model.backbone.load_state_dict(backbone_weights, strict=False)
        if result.missing_keys:
            logger.warning("Missing backbone keys: %s", result.missing_keys)
        if result.unexpected_keys:
            logger.warning("Unexpected backbone keys: %s", result.unexpected_keys)
        logger.info("Loaded pre-extracted backbone weights from %s", args.backbone_weights)
    else:
        model = build_classifier(
            checkpoint_path=checkpoint_path, num_classes=NUM_CLASSES, dropout=args.dropout
        )

    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Wrap with DDP if distributed
    if is_dist_available():
        model = DDP(model, device_ids=[device.index])
    raw_model = model.module if isinstance(model, DDP) else model

    # -----------------------------------------------------------------------
    # Phase 1: Head-only training (backbone frozen)
    # -----------------------------------------------------------------------
    if is_main_process():
        logger.info("=" * 60)
        logger.info("Phase 1: Head-only training (%d epochs)", args.phase1_epochs)
        logger.info("=" * 60)

    # Freeze backbone
    for param in raw_model.backbone.parameters():
        param.requires_grad = False

    optimizer1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.phase1_lr,
        weight_decay=args.weight_decay,
    )
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, T_max=args.phase1_epochs
    )

    global_step = 0
    for epoch in range(1, args.phase1_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer1, criterion, device)
        scheduler1.step()
        val_metrics = evaluate(raw_model, val_loader, criterion, device)

        global_step += 1
        if is_main_process() and writer is not None:
            writer.add_scalar("phase1/train_loss", train_loss, epoch)
            writer.add_scalar("phase1/val_loss", val_metrics["loss"], epoch)
            writer.add_scalar("phase1/val_mAP", val_metrics["mAP"], epoch)
            writer.add_scalar("phase1/val_exact_match", val_metrics["exact_match_ratio"], epoch)
            writer.add_scalar("phase1/val_hamming_loss", val_metrics["hamming_loss"], epoch)
            for cls_name in DEFECT_CLASSES:
                writer.add_scalar(f"phase1/val_AP/{cls_name}", val_metrics["per_class_ap"][cls_name], epoch)
                writer.add_scalar(f"phase1/val_F1/{cls_name}", val_metrics["per_class_f1"][cls_name], epoch)

        if is_main_process():
            logger.info(
                "Phase1 Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_mAP=%.4f  exact_match=%.4f",
                epoch, args.phase1_epochs, train_loss,
                val_metrics["loss"], val_metrics["mAP"], val_metrics["exact_match_ratio"],
            )

    # -----------------------------------------------------------------------
    # Phase 2: End-to-end fine-tuning
    # -----------------------------------------------------------------------
    if is_main_process():
        logger.info("=" * 60)
        logger.info("Phase 2: End-to-end fine-tuning (%d epochs max)", args.phase2_epochs)
        logger.info("=" * 60)

    # Unfreeze backbone
    for param in raw_model.backbone.parameters():
        param.requires_grad = True

    optimizer2 = build_phase2_optimizer(
        raw_model,
        lr_backbone=args.phase2_lr_backbone,
        lr_head=args.phase2_lr_head,
        weight_decay=args.weight_decay,
        layer_decay=args.layer_decay,
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=args.phase2_epochs
    )

    ema = EMA(raw_model, decay=args.ema_decay)
    best_mAP = 0.0
    patience_counter = 0
    best_thresholds = {}

    for epoch in range(1, args.phase2_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model, train_loader, optimizer2, criterion, device, ema=ema
        )
        scheduler2.step()

        # Evaluate with EMA weights on the raw model
        orig_state = copy.deepcopy(raw_model.state_dict())
        ema.apply(raw_model)
        val_metrics = evaluate(raw_model, val_loader, criterion, device)
        raw_model.load_state_dict(orig_state)

        if is_main_process() and writer is not None:
            writer.add_scalar("phase2/train_loss", train_loss, epoch)
            writer.add_scalar("phase2/val_loss", val_metrics["loss"], epoch)
            writer.add_scalar("phase2/val_mAP", val_metrics["mAP"], epoch)
            writer.add_scalar("phase2/val_exact_match", val_metrics["exact_match_ratio"], epoch)
            writer.add_scalar("phase2/val_hamming_loss", val_metrics["hamming_loss"], epoch)
            for cls_name in DEFECT_CLASSES:
                writer.add_scalar(f"phase2/val_AP/{cls_name}", val_metrics["per_class_ap"][cls_name], epoch)
                writer.add_scalar(f"phase2/val_F1/{cls_name}", val_metrics["per_class_f1"][cls_name], epoch)

        if is_main_process():
            logger.info(
                "Phase2 Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_mAP=%.4f  exact_match=%.4f",
                epoch, args.phase2_epochs, train_loss,
                val_metrics["loss"], val_metrics["mAP"], val_metrics["exact_match_ratio"],
            )

        # Checkpointing: save best by mAP (rank 0 only)
        if val_metrics["mAP"] > best_mAP:
            best_mAP = val_metrics["mAP"]
            best_thresholds = val_metrics["thresholds"]
            patience_counter = 0

            if is_main_process():
                # Save best model (EMA weights)
                ema.apply(raw_model)
                best_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save({
                    "model": raw_model.state_dict(),
                    "epoch": epoch,
                    "mAP": best_mAP,
                    "thresholds": best_thresholds,
                }, best_path)
                raw_model.load_state_dict(orig_state)
                logger.info("Saved best model (mAP=%.4f) to %s", best_mAP, best_path)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                if is_main_process():
                    logger.info(
                        "Early stopping at epoch %d (no mAP improvement for %d epochs)",
                        epoch, args.early_stopping_patience,
                    )
                break

    # Save final model and thresholds (rank 0 only)
    if is_main_process():
        ema.apply(raw_model)
        final_path = os.path.join(args.output_dir, "final_model.pth")
        torch.save({
            "model": raw_model.state_dict(),
            "epoch": epoch,
            "mAP": val_metrics["mAP"],
        }, final_path)
        logger.info("Saved final model to %s", final_path)

        # Save thresholds
        thresholds_path = os.path.join(args.output_dir, "thresholds.json")
        with open(thresholds_path, "w") as f:
            json.dump({
                "classes": DEFECT_CLASSES,
                "thresholds": best_thresholds,
                "best_mAP": best_mAP,
            }, f, indent=2)
        logger.info("Saved thresholds to %s", thresholds_path)

        # Final evaluation summary
        logger.info("=" * 60)
        logger.info("Training complete. Best val mAP: %.4f", best_mAP)
        logger.info("Per-class AP:")
        # Re-evaluate best model for final report
        best_ckpt = torch.load(best_path, map_location=device)
        raw_model.load_state_dict(best_ckpt["model"])
        final_metrics = evaluate(raw_model, val_loader, criterion, device)
        for cls_name in DEFECT_CLASSES:
            logger.info("  %s: AP=%.4f  F1=%.4f  threshold=%.2f",
                         cls_name, final_metrics["per_class_ap"][cls_name],
                         final_metrics["per_class_f1"][cls_name],
                         final_metrics["thresholds"][cls_name])
        logger.info("Exact match ratio: %.4f", final_metrics["exact_match_ratio"])
        logger.info("Hamming loss: %.4f", final_metrics["hamming_loss"])
        logger.info("=" * 60)

        writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()
