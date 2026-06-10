#!/usr/bin/env python3
"""
Train R(2+1)D on SpaceJam basketball dataset — Mac mini (CPU/MPS) compatible.

Phase 1 Optimizations (2026-06-11):
- Freeze BN stats (pretrained Kinetics stats >> bs=2 noise)
- macro-F1 as best checkpoint metric (+ balanced_accuracy tracked)
- Weighted CE with inverse-sqrt-freq weights + label_smoothing=0.05
- AdamW with weight_decay + ReduceLROnPlateau scheduler
- Gradient accumulation (default: 4 steps → effective batch=8)
- Early stopping (default: patience=3)
- Reduced gc.collect() frequency (every 20 batches vs every batch)
- FC head with dropout (default: 0.3)

Usage:
    python train_mac.py
    python train_mac.py --resume model_checkpoints/r2plus1d_v3/best.pt
    python train_mac.py --epochs 30 --lr 3e-5 --device cpu
"""
from __future__ import print_function, division

import argparse
import copy
import gc
import json
import os
import signal
import sys
import time
import traceback

import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torch.utils.data._utils.collate import default_collate

from dataset import BasketballDataset, VideoTransform
from utils.checkpoints import init_session_history, save_weights, load_weights, write_history, read_history
from utils.metrics import get_acc_f1_precision_recall

# ── Labels ──────────────────────────────────────────────────────────────
LABELS = {
    0: "block", 1: "pass", 2: "run", 3: "dribble", 4: "shoot",
    5: "ball in hand", 6: "defense", 7: "pick", 8: "no_action", 9: "walk",
}

# ── GC interval (was every batch, now every 20) ────────────────────────
_GC_COLLECT_INTERVAL = 20

# ── Class counts from SpaceJam annotation_dict.json ────────────────────
# block(0)=996, pass(1)=1070, run(2)=5924, dribble(3)=3490, shoot(4)=426,
# ball_in_hand(5)=2362, defense(6)=3866, pick(7)=712, no_action(8)=6490, walk(9)=11749
_CLASS_COUNTS = [996, 1070, 5924, 3490, 426, 2362, 3866, 712, 6490, 11749]

_ARG_DEFAULTS = {
    "accum_steps": 4,
    "best_metric": "macro_f1",
    "early_stop_patience": 3,
    "fc_dropout": 0.3,
    "force_resplit": False,
    "label_smoothing": 0.05,
    "layers": ["layer3", "layer4", "fc"],
    "no_class_weights": False,
    "no_freeze_bn": False,
    "no_sampler": False,
    "progressive_unfreeze": False,
    "save_best_only": False,
    "unfreeze_after_epochs": 2,
    "use_augmentation": False,
    "weight_decay": 1e-4,
}


class SafeDataset(torch.utils.data.Dataset):
    """Wrap another dataset and convert __getitem__ exceptions into skip markers."""

    def __init__(self, dataset, phase_name=""):
        self.dataset = dataset
        self.phase_name = phase_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            return self.dataset[idx]
        except Exception as exc:
            return {
                "_dataset_error": True,
                "_dataset_error_index": idx,
                "_dataset_error_phase": self.phase_name,
                "_dataset_error_msg": str(exc),
            }


class TransformDataset(torch.utils.data.Dataset):
    """Apply a clip transform to samples returned by another dataset."""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.transform is None or "video" not in sample:
            return sample

        transformed = dict(sample)
        clip = transformed["video"]
        if isinstance(clip, torch.Tensor):
            clip_np = clip.detach().cpu().numpy()
            transformed["video"] = torch.from_numpy(self.transform(clip_np)).float()
        else:
            transformed["video"] = self.transform(clip)
        return transformed


def make_safe_collate_fn():
    """Drop failed samples from a batch, and return a marker batch if all are invalid."""

    def safe_collate_fn(batch):
        valid_samples = []
        skip_count = 0
        for sample in batch:
            if isinstance(sample, dict) and sample.get("_dataset_error"):
                skip_count += 1
                continue
            valid_samples.append(sample)

        if not valid_samples:
            return {"_skip_batch": True, "_skip_count": skip_count}

        collated = default_collate(valid_samples)
        collated["_skip_count"] = skip_count
        return collated

    return safe_collate_fn


# Graceful shutdown flag
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    print(f"\n⚠️  Signal {signum} received — will finish current epoch then save & exit")
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ── Phase 1: Freeze BN stats ───────────────────────────────────────────
def freeze_bn_stats(model: nn.Module) -> int:
    """Freeze all BatchNorm3d running stats during training.

    With batch_size=2, BN running stats are extremely noisy.
    Freezing them uses the pretrained Kinetics-400 statistics,
    which are far more stable than what bs=2 can produce.
    Affine params keep their existing requires_grad setting.
    """
    bn_count = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.eval()
            bn_count += 1
    return bn_count


def parse_args():
    p = argparse.ArgumentParser(description="Train R(2+1)D on SpaceJam (Mac)")
    p.add_argument("--device", default=None, help="Force device (cpu/mps/cuda). Auto-detect if omitted.")
    p.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2 for 16GB RAM)")
    p.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--start-epoch", type=int, default=1, help="Start epoch (for manual resume)")
    p.add_argument("--layers", nargs="+", default=["layer3", "layer4", "fc"],
                   help="Layers to unfreeze for fine-tuning")
    p.add_argument("--progressive-unfreeze", action="store_true",
                   help="Progressively unfreeze layers: fc+layer4 first, then layer3 after 2 epochs")
    p.add_argument("--unfreeze-after-epochs", type=int, default=2,
                   help="Epochs to wait before unfreezing next layer group")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    p.add_argument("--annotation-path", default="dataset/annotation_dict.json")
    p.add_argument("--augmented-path", default="dataset/augmented_annotation_dict.json")
    p.add_argument("--video-dir", default="dataset/examples/")
    p.add_argument("--augmented-dir", default="dataset/augmented-examples/")
    p.add_argument("--model-dir", default="model_checkpoints/r2plus1d_v3/")
    p.add_argument("--history-path", default="histories/history_r2plus1d_v3.txt")
    p.add_argument("--save-best-only", action="store_true", help="Only save checkpoint when metric improves")
    # Phase 1 new args
    p.add_argument("--accum-steps", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = accum_steps * batch_size)")
    p.add_argument("--early-stop-patience", type=int, default=3,
                   help="Early stopping patience (0 = disabled)")
    p.add_argument("--fc-dropout", type=float, default=0.3,
                   help="Dropout rate for FC head (0 = no dropout)")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Weight decay for AdamW")
    p.add_argument("--label-smoothing", type=float, default=0.05,
                   help="Label smoothing for CrossEntropyLoss")
    p.add_argument("--best-metric", default="macro_f1",
                   choices=["accuracy", "macro_f1", "balanced_acc"],
                   help="Metric for best checkpoint selection")
    p.add_argument("--no-freeze-bn", action="store_true",
                   help="Do NOT freeze BN running stats (use with larger batches)")
    p.add_argument("--no-class-weights", action="store_true",
                   help="Do NOT apply class weights (use uniform weighting)")
    p.add_argument("--use-augmentation", action="store_true",
                   help="Enable train-time augmentation")
    p.add_argument("--force-resplit", action="store_true",
                   help="Force regeneration of stratified splits")
    p.add_argument("--no-sampler", action="store_true",
                   help="Disable WeightedRandomSampler (use plain shuffle)")
    return p.parse_args()


def auto_device():
    """Pick the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_to_device(tensor, device):
    """Move tensor to device with MPS float32 guard."""
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor.to(device)


def validate_bn_stats(model):
    """Check that BatchNorm running stats have been updated (not all zeros/ones)."""
    issues = []
    for name, buf in model.named_buffers():
        if "running_mean" in name and torch.allclose(buf, torch.zeros_like(buf)):
            issues.append(name)
        if "running_var" in name and torch.allclose(buf, torch.ones_like(buf)):
            issues.append(name)
    if issues:
        print(f"⚠️  BN stats still at init for: {issues[:5]}{'...' if len(issues) > 5 else ''}")
    return len(issues) == 0


# ── Phase 1: Compute class weights ─────────────────────────────────────
def compute_class_weights() -> torch.Tensor:
    """Compute inverse-sqrt-frequency class weights, normalized to mean=1.

    Class order: block(0), pass(1), run(2), dribble(3), shoot(4),
                 ball_in_hand(5), defense(6), pick(7), no_action(8), walk(9)
    """
    total = sum(_CLASS_COUNTS)
    weights = []
    for count in _CLASS_COUNTS:
        freq = count / total
        weight = 1.0 / (freq ** 0.5)  # inverse-sqrt frequency
        weights.append(weight)
    # Normalize so mean = 1.0
    mean_w = sum(weights) / len(weights)
    weights = [w / mean_w for w in weights]
    tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"  🔢 Class weights (inv-sqrt, normalized): {[f'{w:.2f}' for w in weights]}")
    return tensor


def get_backbone_layers(layers):
    """Return trainable backbone groups, excluding the FC head."""
    return [layer for layer in layers if layer != "fc"]


def get_initial_unfrozen_layers(args):
    """Select the layer groups that should be trainable at the start of a run."""
    backbone_layers = get_backbone_layers(args.layers)
    if args.progressive_unfreeze and backbone_layers:
        return ["fc", backbone_layers[-1]]
    return ["fc"] + backbone_layers


def set_trainable_layers(model, layer_groups):
    """Freeze everything, then unfreeze parameters whose names match layer_groups."""
    trainable_groups = tuple(dict.fromkeys(layer_groups))
    trainable_count = 0
    for name, param in model.named_parameters():
        should_train = any(group in name for group in trainable_groups)
        param.requires_grad = should_train
        if should_train:
            trainable_count += 1
    return trainable_count


def build_optimizer_for_current_stage(model, args, lr=None, progressive_lrs=False):
    """Build an AdamW optimizer over the model's currently trainable params."""
    base_lr = args.lr if lr is None else lr
    if not progressive_lrs:
        params_to_update = [param for param in model.parameters() if param.requires_grad]
        return optim.AdamW(params_to_update, lr=base_lr, weight_decay=args.weight_decay)

    backbone_layers = get_backbone_layers(args.layers)
    last_layer = backbone_layers[-1] if backbone_layers else None
    second_last_layer = backbone_layers[-2] if len(backbone_layers) > 1 else None

    fc_params = []
    last_layer_params = []
    second_last_layer_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc" in name:
            fc_params.append(param)
        elif last_layer and last_layer in name:
            last_layer_params.append(param)
        elif second_last_layer and second_last_layer in name:
            second_last_layer_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if fc_params:
        param_groups.append({"params": fc_params, "lr": base_lr})
    if last_layer_params:
        param_groups.append({"params": last_layer_params, "lr": base_lr * 0.3})
    if second_last_layer_params:
        param_groups.append({"params": second_last_layer_params, "lr": base_lr * 0.1})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr * 0.1})

    return optim.AdamW(param_groups, weight_decay=args.weight_decay)


def rebuild_plateau_scheduler(scheduler, optimizer):
    """Recreate ReduceLROnPlateau so it tracks a rebuilt optimizer."""
    if scheduler is None:
        return None

    min_lr = scheduler.min_lrs[0] if getattr(scheduler, "min_lrs", None) else 1e-6
    new_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler.mode,
        factor=scheduler.factor,
        patience=scheduler.patience,
        threshold=scheduler.threshold,
        threshold_mode=scheduler.threshold_mode,
        cooldown=scheduler.cooldown,
        min_lr=min_lr,
        eps=scheduler.eps,
    )
    for attr in ("best", "cooldown_counter", "num_bad_epochs", "last_epoch", "mode_worse"):
        if hasattr(scheduler, attr):
            setattr(new_scheduler, attr, getattr(scheduler, attr))
    new_scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
    return new_scheduler


def ensure_arg_defaults(args):
    for name, default in _ARG_DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    if not hasattr(args, "model_path"):
        setattr(args, "model_path", getattr(args, "model_dir", "model_checkpoints/r2plus1d_v3/"))
    if not hasattr(args, "history_path"):
        setattr(args, "history_path", "histories/history_r2plus1d_v3.txt")
    return args


def extract_label_from_sample(sample):
    if isinstance(sample, dict):
        if "class" in sample:
            return int(sample["class"])
        if "action" in sample:
            action = sample["action"]
            if isinstance(action, torch.Tensor):
                if action.ndim == 0:
                    return int(action.item())
                return int(torch.argmax(action).item())
            if isinstance(action, (list, tuple, np.ndarray)):
                action_arr = np.asarray(action)
                if action_arr.ndim == 0:
                    return int(action_arr.item())
                return int(np.argmax(action_arr))
            return int(action)
    return 0


def get_dataset_labels(dataset):
    if hasattr(dataset, "video_list"):
        return np.array([int(label) for _, label in dataset.video_list], dtype=int)

    labels = []
    source_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
    for idx in range(len(dataset)):
        try:
            sample = source_dataset[idx] if source_dataset is not dataset else dataset[idx]
            labels.append(extract_label_from_sample(sample))
        except Exception:
            labels.append(0)
    return np.array(labels, dtype=int)


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    device,
    args,
    scheduler=None,
    start_epoch=1,
    num_epochs=20,
    initial_best_metric=-1.0,
    initial_best_epoch=None,
    initial_best_weights=None,
    initial_best_acc=None,
):
    """Train and validate the model with Phase 1 optimizations."""
    best_metric_explicit = hasattr(args, "best_metric")
    args = ensure_arg_defaults(args)
    init_session_history(args)
    since = time.time()
    accum_steps = max(1, int(args.accum_steps))

    if initial_best_acc is not None and initial_best_metric == -1.0:
        initial_best_metric = initial_best_acc
        if not best_metric_explicit:
            args.best_metric = "accuracy"

    best_metric_name = args.best_metric

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    train_f1_score, val_f1_score = [], []
    plot_epoch = []

    best_model_wts = copy.deepcopy(initial_best_weights) if initial_best_weights is not None else None
    best_optimizer_state = None
    best_metric = float(initial_best_metric)
    best_epoch = initial_best_epoch
    best_updated_this_run = False
    trained_any_epoch = False
    epochs_without_improvement = 0

    # Initialize to avoid unbound warnings
    train_loss = val_loss = 0.0
    train_accuracy = val_accuracy = 0.0
    train_cm_str = val_cm_str = ""
    train_f1 = val_f1 = 0.0
    train_precision = val_precision = 0.0
    train_recall = val_recall = 0.0
    val_macro_f1 = 0.0
    val_balanced_acc = 0.0
    train_macro_f1 = 0.0
    train_balanced_acc = 0.0

    global _shutdown_requested

    for epoch in range(start_epoch, num_epochs + 1):
        if _shutdown_requested:
            print("⚠️  Shutdown requested — saving and exiting early")
            break

        print(f"\n{'='*55}")
        print(f"  Epoch {epoch}/{num_epochs}")
        print(f"{'='*55}")

        if args.progressive_unfreeze:
            epochs_in_this_run = epoch - start_epoch + 1
            backbone_layers = get_backbone_layers(args.layers)
            if backbone_layers:
                if epochs_in_this_run == 1:
                    set_trainable_layers(model, ["fc", backbone_layers[-1]])
                    print(f"  Progressive unfreeze Phase 1: training fc + {backbone_layers[-1]}")
                elif epochs_in_this_run == args.unfreeze_after_epochs + 1 and len(backbone_layers) > 1:
                    newly_unfrozen = backbone_layers[-2]
                    set_trainable_layers(model, ["fc", backbone_layers[-1], newly_unfrozen])
                    old_lr = optimizer.param_groups[0]["lr"]
                    optimizer = build_optimizer_for_current_stage(
                        model,
                        args,
                        lr=old_lr,
                        progressive_lrs=True,
                    )
                    scheduler = rebuild_plateau_scheduler(scheduler, optimizer)
                    print(f"  Progressive unfreeze Phase 2: also unfreezing {newly_unfrozen}")
                    print(f"    Rebuilt optimizer with {len(optimizer.param_groups)} learning-rate groups")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                if not args.no_freeze_bn:
                    bn_count = freeze_bn_stats(model)
                    if epoch == start_epoch:
                        print(f"  🔒 BN stats frozen ({bn_count} BatchNorm3d layers)")
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            n_samples = 0
            pred_classes, ground_truths = [], []
            skip_count = 0
            batch_idx = 0

            pbar = tqdm(dataloaders[phase], desc=f"{phase} epoch {epoch}")
            for sample in pbar:
                if isinstance(sample, dict) and sample.get("_skip_batch"):
                    skip_count += int(sample.get("_skip_count", 0))
                    continue

                skip_count += int(sample.get("_skip_count", 0))
                if "_skip_count" in sample:
                    del sample["_skip_count"]

                try:
                    inputs = safe_to_device(sample["video"].float(), device)
                    labels = safe_to_device(sample["action"].float(), device)
                    label_indices = torch.max(labels, 1)[1]
                except Exception as e:
                    skip_count += 1
                    continue

                # ── Phase 1: Gradient accumulation ───────────────────
                if phase == "train":
                    if batch_idx % accum_steps == 0:
                        optimizer.zero_grad(set_to_none=True)

                try:
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = criterion(outputs, label_indices)
                        # Scale loss for accumulation
                        if phase == "train":
                            loss = loss / accum_steps
                        _, preds = torch.max(outputs, 1)

                        if phase == "train":
                            loss.backward()
                            # Step only every accum_steps
                            if (batch_idx + 1) % accum_steps == 0:
                                optimizer.step()

                    # Report unscaled loss for logging
                    unscaled_loss = loss.item() * (accum_steps if phase == "train" else 1)
                    batch_size = inputs.size(0)
                    running_loss += unscaled_loss * batch_size
                    running_corrects += (preds == label_indices).sum().item()
                    n_samples += batch_size

                    pred_classes.extend(preds.detach().cpu().numpy())
                    ground_truths.extend(label_indices.detach().cpu().numpy())

                    pbar.set_postfix(
                        loss=f"{running_loss/max(n_samples,1):.4f}",
                        acc=f"{running_corrects/max(n_samples,1):.3f}",
                        skip=skip_count if skip_count > 0 else ""
                    )
                except RuntimeError as e:
                    # MPS ops fallback → skip batch
                    if "mps" in str(e).lower() or "Metal" in str(e):
                        skip_count += 1
                        if skip_count <= 3:
                            print(f"\n  MPS error (skipping batch): {e}")
                        continue
                    raise

                # ── Phase 1: Reduced gc.collect() ─────────────────────
                batch_idx += 1
                del inputs, labels, label_indices, outputs, loss, preds
                if batch_idx % _GC_COLLECT_INTERVAL == 0:
                    gc.collect()

            # Flush remaining accumulated gradients at epoch end
            if phase == "train" and batch_idx > 0 and batch_idx % accum_steps != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Epoch-end gc
            gc.collect()

            if n_samples == 0:
                print(f"  ⚠️  No valid samples in {phase} this epoch (all skipped)")
                continue

            if phase == "train":
                trained_any_epoch = True

            epoch_loss = running_loss / n_samples
            epoch_acc = running_corrects / n_samples
            pred_arr = np.asarray(pred_classes)
            gt_arr = np.asarray(ground_truths)
            accuracy, f1_micro, precision, recall = get_acc_f1_precision_recall(pred_arr, gt_arr)
            cm = confusion_matrix(gt_arr, pred_arr, labels=list(range(10)))

            # ── Phase 1: macro-F1 and balanced accuracy ───────────────
            macro_f1 = f1_score(gt_arr, pred_arr, average="macro", labels=list(range(10)), zero_division=0)
            bal_acc = balanced_accuracy_score(gt_arr, pred_arr)

            # Per-class recall for diagnostics
            per_class_recall = recall_score(gt_arr, pred_arr, average=None, labels=list(range(10)), zero_division=0)
            per_class_recall_str = " | ".join(
                f"{LABELS[i]}:{per_class_recall[i]:.2f}" for i in range(10)
            )

            print(f"{phase} — Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}  "
                  f"macro-F1: {macro_f1:.4f}  bal-acc: {bal_acc:.4f}  skipped: {skip_count}")
            print(f"  Per-class recall: {per_class_recall_str}")
            print(f"Confusion matrix:\n{cm}")

            if phase == "val":
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                val_f1_score.append(f1_micro)
                val_loss = epoch_loss
                val_accuracy = accuracy
                val_f1 = f1_micro
                val_precision = precision
                val_recall = recall
                val_cm_str = np.array_str(cm)
                val_macro_f1 = macro_f1
                val_balanced_acc = bal_acc

                # ── Phase 1: Best metric selection ────────────────────
                if best_metric_name == "macro_f1":
                    current_metric = macro_f1
                elif best_metric_name == "balanced_acc":
                    current_metric = bal_acc
                else:
                    current_metric = epoch_acc

                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                    best_updated_this_run = True
                    epochs_without_improvement = 0
                    # Save best checkpoint immediately
                    os.makedirs(args.model_path, exist_ok=True)
                    best_path = os.path.join(args.model_path, "best.pt")
                    torch.save({
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": best_optimizer_state,
                        "best_val_acc": epoch_acc,
                        "best_macro_f1": macro_f1,
                        "best_balanced_acc": bal_acc,
                        "best_metric_name": best_metric_name,
                        "best_metric_value": best_metric,
                    }, best_path)
                    print(f"  🏆 New best {best_metric_name}: {best_metric:.4f} — saved to {best_path}")
                else:
                    epochs_without_improvement += 1

            if phase == "train":
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                train_f1_score.append(f1_micro)
                plot_epoch.append(epoch)
                train_loss = epoch_loss
                train_accuracy = accuracy
                train_f1 = f1_micro
                train_precision = precision
                train_recall = recall
                train_cm_str = np.array_str(cm)
                train_macro_f1 = macro_f1
                train_balanced_acc = bal_acc

        # ── Phase 1: LR scheduler step ───────────────────────────────
        if scheduler is not None:
            if best_metric_name == "macro_f1":
                scheduler.step(val_macro_f1)
            elif best_metric_name == "balanced_acc":
                scheduler.step(val_balanced_acc)
            else:
                scheduler.step(val_accuracy)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  📉 LR after scheduler: {current_lr:.2e}")

        # Validate BN stats every 5 epochs
        if epoch % 5 == 0:
            validate_bn_stats(model)

        if not args.save_best_only:
            # Save epoch checkpoint (for resume)
            os.makedirs(args.model_path, exist_ok=True)
            model_name = save_weights(model, args, epoch, optimizer)

            # Also save a latest.pt for easy resume
            latest_path = os.path.join(args.model_path, "latest.pt")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_acc": val_accuracy,
                "best_epoch": best_epoch,
                "best_state_dict": best_model_wts,
                "best_optimizer": best_optimizer_state,
                # Phase 1 new fields
                "best_macro_f1": val_macro_f1,
                "best_balanced_acc": val_balanced_acc,
                "best_metric_name": best_metric_name,
                "best_metric_value": best_metric,
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            }, latest_path)
        else:
            model_name = "best.pt"

        # Embed new metrics into history string for write_history compatibility
        extra_metrics = (
            f" || macro-F1: {val_macro_f1:.5f} || bal-acc: {val_balanced_acc:.5f}"
            f" || train-macro-F1: {train_macro_f1:.5f}"
        )
        val_cm_str_extended = val_cm_str + extra_metrics

        write_history(
            args.history_path, model_name,
            train_loss, val_loss,
            train_accuracy, val_accuracy,
            train_f1, val_f1,
            train_precision, val_precision,
            train_recall, val_recall,
            train_cm_str, val_cm_str_extended,
        )

        # ── Phase 1: Early stopping check ────────────────────────────
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"  ⏹️  Early stopping: no improvement for {args.early_stop_patience} epochs")
            break

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best {best_metric_name}: {best_metric:.4f} at epoch {best_epoch}")

    if trained_any_epoch and best_model_wts is not None:
        model.load_state_dict(best_model_wts)

        if best_updated_this_run:
            best_path = os.path.join(args.model_path, "best.pt")
            torch.save({
                "epoch": best_epoch,
                "state_dict": best_model_wts,
                "optimizer": best_optimizer_state or optimizer.state_dict(),
                "best_val_acc": best_metric if best_metric_name == "accuracy" else val_accuracy,
                "best_epoch": best_epoch,
                "best_macro_f1": val_macro_f1,
                "best_balanced_acc": val_balanced_acc,
                "best_metric_name": best_metric_name,
                "best_metric_value": best_metric,
            }, best_path)
            print(f"Best model saved to {best_path}")
        else:
            print("  Note: No validation improvement; existing best checkpoint was not overwritten.")
    else:
        print("  Note: No valid epoch completed; best checkpoint was not overwritten.")

    if validate_bn_stats(model):
        print("✅ All BN running stats are properly trained")

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history, train_f1_score, val_f1_score, plot_epoch


def check_accuracy(loader, model, device):
    """Run inference on test set, skipping bad samples."""
    model.eval()
    num_correct = 0
    num_samples = 0
    skip_count = 0
    with torch.no_grad():
        for sample in tqdm(loader, desc="Testing"):
            if isinstance(sample, dict) and sample.get("_skip_batch"):
                skip_count += int(sample.get("_skip_count", 0))
                continue

            skip_count += int(sample.get("_skip_count", 0))
            if "_skip_count" in sample:
                del sample["_skip_count"]

            try:
                x = safe_to_device(sample["video"].float(), device)
                y = safe_to_device(sample["action"].float(), device)
            except Exception:
                skip_count += 1
                continue
            try:
                scores = model(x)
                predictions = scores.argmax(1)
                y_idx = y.argmax(1)
                num_correct += (predictions == y_idx).sum().item()
                num_samples += predictions.size(0)
            except RuntimeError:
                skip_count += 1
                continue
            del x, y, scores, predictions, y_idx

    gc.collect()

    if num_samples > 0:
        acc = num_correct / num_samples * 100
        print(f"Test accuracy: {num_correct}/{num_samples} = {acc:.2f}% (skipped {skip_count})")
    else:
        acc = 0.0
        print(f"⚠️  No valid test samples (skipped {skip_count})")
    model.train()
    return acc


def main():
    args = ensure_arg_defaults(parse_args())

    # ── Device ──────────────────────────────────────────────────────
    device = torch.device(args.device) if args.device else auto_device()
    print(f"PyTorch {torch.__version__} | Device: {device}")
    print(f"  Phase 1 Optimizations:")
    print(f"    🔒 Freeze BN: {'NO' if args.no_freeze_bn else 'YES'}")
    print(f"    📊 Best metric: {args.best_metric}")
    print(f"    ⚖️  Class weights: {'NO' if args.no_class_weights else 'YES (inv-sqrt)'}")
    print(f"    🎯 Label smoothing: {args.label_smoothing}")
    print(f"    🔧 Optimizer: AdamW (weight_decay={args.weight_decay})")
    print(f"    📉 Scheduler: ReduceLROnPlateau (monitor {args.best_metric})")
    print(f"    📦 Gradient accumulation: {args.accum_steps} steps (eff. batch={args.accum_steps * args.batch_size})")
    print(f"    ⏹️  Early stopping patience: {args.early_stop_patience}")
    print(f"    💧 FC dropout: {args.fc_dropout}")
    print(f"    🧊 Progressive unfreeze: {'YES' if args.progressive_unfreeze else 'NO'}")
    if args.progressive_unfreeze:
        print(f"       next stage after {args.unfreeze_after_epochs} epoch(s)")
    print(f"    🧹 gc.collect() interval: every {_GC_COLLECT_INTERVAL} batches")
    print(f"    🎨 Data augmentation: {'ENABLED (train mode)' if args.use_augmentation else 'DISABLED'}")
    print(f"    🎲 Weighted sampler: {'DISABLED' if args.no_sampler else 'ENABLED'}")

    if device.type == "mps":
        print("  Note: MPS may have ops compatibility issues. Will skip bad batches.")

    # ── Dataset sizes ───────────────────────────────────────────────
    try:
        with open(args.annotation_path) as f:
            n_orig = len(json.load(f))
    except FileNotFoundError:
        print(f"❌ Annotation file not found: {args.annotation_path}")
        print("   Please download the SpaceJam dataset first. See docs/training-plan.md")
        sys.exit(1)

    try:
        with open(args.augmented_path) as f:
            n_aug = len(json.load(f))
    except FileNotFoundError:
        print(f"⚠️  Augmented annotation file not found: {args.augmented_path}")
        print("   Training with original data only.")
        n_aug = 0

    n_total = n_orig + n_aug
    test_n = min(4990, n_total // 10)
    val_n = min(9980, n_total // 5)
    train_n = n_total - test_n - val_n
    print(f"Dataset: {n_total} samples (train={train_n}, val={val_n}, test={test_n})")

    # ── Args namespace for checkpoint utils ─────────────────────────
    from easydict import EasyDict
    ckpt_dict = {
        "base_model_name": "r2plus1d_multiclass",
        "lr": args.lr,
        "start_epoch": args.start_epoch,
        "model_path": args.model_dir,
        "history_path": args.history_path,
    }
    ckpt_args = EasyDict(ckpt_dict)
    for k, v in ckpt_dict.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args.model_path = args.model_dir

    # ── Model ───────────────────────────────────────────────────────
    print("Loading R(2+1)D-18 with Kinetics-400 pretrained weights...")
    model = models.video.r2plus1d_18(weights=models.video.R2Plus1D_18_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    initial_layer_groups = get_initial_unfrozen_layers(args)
    trainable_count = set_trainable_layers(model, initial_layer_groups)

    # ── Phase 1: FC head with dropout ───────────────────────────────
    fc_in_features = model.fc.in_features
    if args.fc_dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(args.fc_dropout),
            nn.Linear(fc_in_features, 10, bias=True),
        )
        print(f"Trainable parameters: {trainable_count} + fc layer (dropout={args.fc_dropout})")
    else:
        model.fc = nn.Linear(fc_in_features, 10, bias=True)
        print(f"Trainable parameters: {trainable_count} + fc layer")
    if args.progressive_unfreeze:
        print(f"  Initial progressive stage: fc + {', '.join(initial_layer_groups[1:]) if len(initial_layer_groups) > 1 else 'fc only'}")

    # Resume from checkpoint if specified
    ckpt = None
    best_metric_so_far = -1.0
    best_epoch_so_far = None
    best_weights_for_resume = None
    scheduler_state_for_resume = None
    if args.resume:
        resume_path = args.resume
        if not os.path.exists(resume_path):
            # Try model_dir/latest.pt or best.pt
            for fallback in ["latest.pt", "best.pt"]:
                fallback_path = os.path.join(args.model_dir, fallback)
                if os.path.exists(fallback_path):
                    resume_path = fallback_path
                    break
        if os.path.exists(resume_path):
            print(f"Resuming from {resume_path}")
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["state_dict"], strict=False)
            if "epoch" in ckpt:
                args.start_epoch = ckpt["epoch"] + 1
                print(f"  Resuming from epoch {args.start_epoch}")
            # Phase 1: try new metric fields, fall back to old best_val_acc
            best_metric_so_far = ckpt.get("best_metric_value", ckpt.get("best_val_acc", -1.0))
            best_epoch_so_far = ckpt.get("best_epoch", ckpt.get("epoch"))
            best_state_from_ckpt = ckpt.get("best_state_dict")
            if isinstance(best_state_from_ckpt, dict):
                best_weights_for_resume = copy.deepcopy(best_state_from_ckpt)
            elif ckpt.get("epoch") == best_epoch_so_far:
                best_weights_for_resume = copy.deepcopy(ckpt.get("state_dict"))
            scheduler_state_for_resume = ckpt.get("scheduler_state_dict")
            print(f"  Best {args.best_metric} so far: {best_metric_so_far:.4f}")
        else:
            print(f"⚠️  Checkpoint not found at {resume_path}, starting from scratch")

    model = model.to(device)

    # ── Phase 1: Weighted CE + label smoothing ─────────────────────
    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"  ⚖️  Loss: CrossEntropyLoss (no class weights, label_smoothing={args.label_smoothing})")
    else:
        class_weights_tensor = compute_class_weights()
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor,
            label_smoothing=args.label_smoothing,
        )
        criterion.weight = criterion.weight.to(device)
        print(f"  ⚖️  Loss: WeightedCrossEntropyLoss (inv-sqrt, label_smoothing={args.label_smoothing})")

    # ── Phase 1: AdamW optimizer ───────────────────────────────────
    optimizer = build_optimizer_for_current_stage(model, args)

    if ckpt is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            for pg in optimizer.param_groups:
                pg["weight_decay"] = args.weight_decay
            print(f"  🔧 Loaded optimizer state from checkpoint (forced weight_decay={args.weight_decay})")
        except ValueError as exc:
            print(f"  ⚠️  Could not load optimizer state: {exc} (rebuilding for current trainable layers)")

    # ── Phase 1: ReduceLROnPlateau scheduler ───────────────────────
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6
    )
    if scheduler_state_for_resume is not None:
        try:
            scheduler.load_state_dict(scheduler_state_for_resume)
            print(f"  📉 Loaded scheduler state from checkpoint")
        except Exception as e:
            print(f"  ⚠️  Could not load scheduler state: {e} (starting fresh)")

    # ── Dataset & DataLoader ────────────────────────────────────────
    print("Loading dataset...")
    if args.use_augmentation:
        train_transform = VideoTransform(mode='train', target_size=112)
        val_transform = VideoTransform(mode='val', target_size=112)
    else:
        train_transform = None
        val_transform = None

    basketball_dataset = BasketballDataset(
        annotation_dict=args.annotation_path,
        augmented_dict=args.augmented_path,
        video_dir=args.video_dir,
        augmented_dir=args.augmented_dir,
        transform=None,
    )

    # ── Stratified Split ─────────────────────────────────────────────
    splits_path = os.path.join(args.model_dir, "stratified_split.json")
    dataset_labels = get_dataset_labels(basketball_dataset)
    indices = np.arange(len(dataset_labels))

    should_generate_split = args.force_resplit or not os.path.exists(splits_path)
    if not should_generate_split:
        print(f"Loading existing stratified split from {splits_path}")
        with open(splits_path) as f:
            splits = json.load(f)
        train_indices = splits["train"]
        val_indices = splits["val"]
        test_indices = splits["test"]
        if len(train_indices) + len(val_indices) + len(test_indices) != len(basketball_dataset):
            print("  Existing split size mismatch; regenerating")
            should_generate_split = True

    if should_generate_split:
        print("Generating stratified train/val/test split...")
        try:
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
            train_val_idx, test_idx = next(sss1.split(indices, dataset_labels))

            remaining_labels = dataset_labels[train_val_idx]
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.222, random_state=42)
            train_rel, val_rel = next(sss2.split(train_val_idx, remaining_labels))
            train_indices = train_val_idx[train_rel].tolist()
            val_indices = train_val_idx[val_rel].tolist()
            test_indices = test_idx.tolist()
        except ValueError as exc:
            print(f"  Stratified split unavailable for this dataset shape: {exc}")
            rng = np.random.default_rng(42)
            shuffled = rng.permutation(indices)
            test_count = max(1, int(round(len(shuffled) * 0.1)))
            val_count = max(1, int(round(len(shuffled) * 0.2)))
            test_indices = shuffled[:test_count].tolist()
            val_indices = shuffled[test_count:test_count + val_count].tolist()
            train_indices = shuffled[test_count + val_count:].tolist()

        os.makedirs(args.model_dir, exist_ok=True)
        with open(splits_path, "w") as f:
            json.dump({"train": train_indices, "val": val_indices, "test": test_indices}, f)
        print(f"  Saved split to {splits_path}")

    train_subset = Subset(basketball_dataset, train_indices)
    val_subset = Subset(basketball_dataset, val_indices)
    test_subset = Subset(basketball_dataset, test_indices)

    train_dataset = TransformDataset(train_subset, train_transform)
    val_dataset = TransformDataset(val_subset, val_transform)
    test_dataset = TransformDataset(test_subset, val_transform)

    train_labels = dataset_labels[np.array(train_indices, dtype=int)]
    class_counts = np.bincount(train_labels, minlength=10)
    class_weights = 1.0 / (class_counts + 1)
    sample_weights = class_weights[train_labels]
    sample_weights = torch.from_numpy(sample_weights).double()
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_indices),
        replacement=True,
    )

    safe_collate_fn = make_safe_collate_fn()
    train_loader = DataLoader(
        SafeDataset(train_dataset, "train"),
        shuffle=False if not args.no_sampler else True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=safe_collate_fn,
        sampler=None if args.no_sampler else train_sampler,
    )
    val_loader = DataLoader(
        SafeDataset(val_dataset, "val"),
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=safe_collate_fn,
    )
    test_loader = DataLoader(
        SafeDataset(test_dataset, "test"),
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=safe_collate_fn,
    )
    dataloaders = {"train": train_loader, "val": val_loader}

    print(
        f"DataLoader ready — batch_size={args.batch_size}, workers={args.num_workers}, "
        f"train/val/test={len(train_indices)}/{len(val_indices)}/{len(test_indices)}"
    )

    # ── Train ───────────────────────────────────────────────────────
    model, tlh, vlh, tah, vah, tf1, vf1, pe = train_model(
        model, dataloaders, criterion, optimizer, device, args,
        scheduler=scheduler,
        start_epoch=args.start_epoch,
        num_epochs=args.epochs,
        initial_best_metric=best_metric_so_far,
        initial_best_epoch=best_epoch_so_far,
        initial_best_weights=best_weights_for_resume,
    )

    # ── Test ────────────────────────────────────────────────────────
    check_accuracy(test_loader, model, device)


if __name__ == "__main__":
    main()
