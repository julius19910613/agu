#!/usr/bin/env python3
"""
Train R(2+1)D on SpaceJam basketball dataset — Mac mini (CPU/MPS) compatible.

Key changes from original train.py:
- Auto-detects device (MPS > CPU, no CUDA assumed), with MPS→CPU fallback
- batch_size=2, num_workers=0, gc.collect() per batch to fit 16GB RAM
- Uses new torchvision weights API (no deprecated `pretrained=True`)
- Saves best checkpoint (by val accuracy) separately
- Validates BN running stats periodically (guards against empty checkpoints)
- Skips corrupted/unreadable videos instead of crashing
- MPS float32 guard: cast before .to(device)
- Resume from any epoch checkpoint with full optimizer state
- Signals: handles SIGTERM gracefully for long runs

Usage:
    python train_mac.py
    python train_mac.py --resume model_checkpoints/r2plus1d_v3/best.pt
    python train_mac.py --epochs 30 --lr 3e-4 --device cpu
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
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, random_split
from torch.utils.data._utils.collate import default_collate

from dataset import BasketballDataset
from utils.checkpoints import init_session_history, save_weights, load_weights, write_history, read_history
from utils.metrics import get_acc_f1_precision_recall

# ── Labels ──────────────────────────────────────────────────────────────
LABELS = {
    0: "block", 1: "pass", 2: "run", 3: "dribble", 4: "shoot",
    5: "ball in hand", 6: "defense", 7: "pick", 8: "no_action", 9: "walk",
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


def parse_args():
    p = argparse.ArgumentParser(description="Train R(2+1)D on SpaceJam (Mac)")
    p.add_argument("--device", default=None, help="Force device (cpu/mps/cuda). Auto-detect if omitted.")
    p.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2 for 16GB RAM)")
    p.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--start-epoch", type=int, default=1, help="Start epoch (for manual resume)")
    p.add_argument("--layers", nargs="+", default=["layer3", "layer4", "fc"],
                   help="Layers to unfreeze for fine-tuning")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    p.add_argument("--annotation-path", default="dataset/annotation_dict.json")
    p.add_argument("--augmented-path", default="dataset/augmented_annotation_dict.json")
    p.add_argument("--video-dir", default="dataset/examples/")
    p.add_argument("--augmented-dir", default="dataset/augmented-examples/")
    p.add_argument("--model-dir", default="model_checkpoints/r2plus1d_v3/")
    p.add_argument("--history-path", default="histories/history_r2plus1d_v3.txt")
    p.add_argument("--save-best-only", action="store_true", help="Only save checkpoint when val acc improves")
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


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    device,
    args,
    start_epoch=1,
    num_epochs=20,
    initial_best_acc=-1.0,
    initial_best_epoch=None,
    initial_best_weights=None,
):
    """Train and validate the model with error recovery and memory management."""
    init_session_history(args)
    since = time.time()

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    train_f1_score, val_f1_score = [], []
    plot_epoch = []

    best_model_wts = copy.deepcopy(initial_best_weights) if initial_best_weights is not None else None
    best_optimizer_state = None
    best_acc = float(initial_best_acc)
    best_epoch = initial_best_epoch
    best_updated_this_run = False
    trained_any_epoch = False

    # Initialize to avoid unbound warnings
    train_loss = val_loss = 0.0
    train_accuracy = val_accuracy = 0.0
    train_cm_str = val_cm_str = ""
    train_f1 = val_f1 = 0.0
    train_precision = val_precision = 0.0
    train_recall = val_recall = 0.0

    global _shutdown_requested

    for epoch in range(start_epoch, num_epochs + 1):
        if _shutdown_requested:
            print("⚠️  Shutdown requested — saving and exiting early")
            break

        print(f"\n{'='*55}")
        print(f"  Epoch {epoch}/{num_epochs}")
        print(f"{'='*55}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            n_samples = 0
            pred_classes, ground_truths = [], []
            skip_count = 0

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

                optimizer.zero_grad()

                try:
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = criterion(outputs, label_indices)
                        _, preds = torch.max(outputs, 1)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    batch_size = inputs.size(0)
                    running_loss += loss.item() * batch_size
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

                # Free memory
                del inputs, labels, label_indices, outputs, loss, preds
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
            accuracy, f1, precision, recall = get_acc_f1_precision_recall(pred_arr, gt_arr)
            cm = confusion_matrix(gt_arr, pred_arr, labels=list(range(10)))

            print(f"{phase} — Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}  F1: {f1:.4f}  skipped: {skip_count}")
            print(f"Confusion matrix:\n{cm}")

            if phase == "val":
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                val_f1_score.append(f1)
                val_loss = epoch_loss
                val_accuracy = accuracy
                val_f1 = f1
                val_precision = precision
                val_recall = recall
                val_cm_str = np.array_str(cm)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                    best_updated_this_run = True
                    # Save best checkpoint immediately
                    os.makedirs(args.model_path, exist_ok=True)
                    best_path = os.path.join(args.model_path, "best.pt")
                    torch.save({
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": best_optimizer_state,
                        "best_val_acc": best_acc,
                    }, best_path)
                    print(f"  🏆 New best val acc: {best_acc:.4f} — saved to {best_path}")

            if phase == "train":
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                train_f1_score.append(f1)
                plot_epoch.append(epoch)
                train_loss = epoch_loss
                train_accuracy = accuracy
                train_f1 = f1
                train_precision = precision
                train_recall = recall
                train_cm_str = np.array_str(cm)

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
                "best_val_acc": best_acc,
                "best_epoch": best_epoch,
                "best_state_dict": best_model_wts,
                "best_optimizer": best_optimizer_state,
            }, latest_path)
        else:
            model_name = "best.pt"

        write_history(
            args.history_path, model_name,
            train_loss, val_loss,
            train_accuracy, val_accuracy,
            train_f1, val_f1,
            train_precision, val_precision,
            train_recall, val_recall,
            train_cm_str, val_cm_str,
        )

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    if trained_any_epoch and best_model_wts is not None:
        # Load best weights from the best epoch available in this run.
        model.load_state_dict(best_model_wts)

        if best_updated_this_run:
            # Final best checkpoint save keeps weights and optimizer from the same best epoch.
            best_path = os.path.join(args.model_path, "best.pt")
            torch.save({
                "epoch": best_epoch,
                "state_dict": best_model_wts,
                "optimizer": best_optimizer_state or optimizer.state_dict(),
                "best_val_acc": best_acc,
                "best_epoch": best_epoch,
            }, best_path)
            print(f"Best model saved to {best_path}")
        else:
            print("  Note: No validation improvement; existing best checkpoint was not overwritten.")
    else:
        print("  Note: No valid epoch completed; best checkpoint was not overwritten.")

    # Validate final BN stats
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
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────
    device = torch.device(args.device) if args.device else auto_device()
    print(f"PyTorch {torch.__version__} | Device: {device}")

    if device.type == "mps":
        print("  Note: MPS may have ops compatibility issues. Will skip bad batches.")

    # ── Dataset sizes ───────────────────────────────────────────────
    # Count samples from annotation files
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
    # Merge into args for checkpoint utils
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

    # Unfreeze specified layers
    trainable_count = 0
    for name, param in model.named_parameters():
        for layer in args.layers:
            if layer in name:
                param.requires_grad = True
                trainable_count += 1
                break

    # Replace fc head
    model.fc = nn.Linear(model.fc.in_features, 10, bias=True)
    print(f"Trainable parameters: {trainable_count} + fc layer")

    # Resume from checkpoint if specified
    ckpt = None
    best_acc_so_far = -1.0
    best_epoch_so_far = None
    best_weights_for_resume = None
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
            best_acc_so_far = ckpt.get("best_val_acc", -1.0)
            best_epoch_so_far = ckpt.get("best_epoch", ckpt.get("epoch"))
            best_state_from_ckpt = ckpt.get("best_state_dict")
            if isinstance(best_state_from_ckpt, dict):
                best_weights_for_resume = copy.deepcopy(best_state_from_ckpt)
            elif ckpt.get("epoch") == best_epoch_so_far:
                best_weights_for_resume = copy.deepcopy(ckpt.get("state_dict"))
            print(f"  Best val acc so far: {best_acc_so_far:.4f}")
        else:
            print(f"⚠️  Checkpoint not found at {resume_path}, starting from scratch")

    model = model.to(device)

    # ── Optimizer (only trainable params) ───────────────────────────
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if ckpt is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        # Move optimizer state to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # ── Dataset & DataLoader ────────────────────────────────────────
    print("Loading dataset...")
    basketball_dataset = BasketballDataset(
        annotation_dict=args.annotation_path,
        augmented_dict=args.augmented_path,
        video_dir=args.video_dir,
        augmented_dir=args.augmented_dir,
    )

    train_subset, test_subset = random_split(
        basketball_dataset, [n_total - test_n, test_n],
        generator=torch.Generator().manual_seed(1),
    )
    train_subset, val_subset = random_split(
        train_subset, [train_n, val_n],
        generator=torch.Generator().manual_seed(1),
    )

    safe_collate_fn = make_safe_collate_fn()
    train_loader = DataLoader(
        SafeDataset(train_subset, "train"),
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=safe_collate_fn
    )
    val_loader = DataLoader(
        SafeDataset(val_subset, "val"),
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=safe_collate_fn
    )
    test_loader = DataLoader(
        SafeDataset(test_subset, "test"),
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=safe_collate_fn
    )
    dataloaders = {"train": train_loader, "val": val_loader}

    print(f"DataLoader ready — batch_size={args.batch_size}, workers={args.num_workers}")

    # ── Train ───────────────────────────────────────────────────────
    model, tlh, vlh, tah, vah, tf1, vf1, pe = train_model(
        model, dataloaders, criterion, optimizer, device, args,
        start_epoch=args.start_epoch,
        num_epochs=args.epochs,
        initial_best_acc=best_acc_so_far,
        initial_best_epoch=best_epoch_so_far,
        initial_best_weights=best_weights_for_resume,
    )

    # ── Test ────────────────────────────────────────────────────────
    check_accuracy(test_loader, model, device)


if __name__ == "__main__":
    main()
