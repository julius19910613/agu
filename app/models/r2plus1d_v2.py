"""
Build an R(2+1)D model with Kinetics-400 pretrained backbone.

Strategies:
  - "pretrained": Load torchvision pretrained weights, replace fc with
    random 10-class head.  Good for quick validation.
  - "checkpoint": Load the SpaceJam checkpoint (may contain garbage weights).
  - "pretrained_finetune": Load pretrained backbone + attempt to restore
    fc weights that actually learned basketball classes (experimental).
"""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from torchvision import models

from app.config import Settings
from utils.checkpoints import load_weights

logger = logging.getLogger(__name__)


def build_r2plus1d_model(
    settings: Settings,
    device: torch.device | None = None,
    *,
    strategy: str = "checkpoint",
) -> nn.Module:
    """Build and return an R(2+1)D-18 model.

    Args:
        settings: Application settings.
        device: Target device. Auto-detected if None.
        strategy: Weight loading strategy — "checkpoint", "pretrained",
            or "pretrained_finetune".

    Returns:
        The loaded model in eval mode on the target device.
    """
    from easydict import EasyDict

    device = device or torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    if strategy == "pretrained":
        # ---- Kinetics-400 pretrained backbone ----
        model = models.video.r2plus1d_18(
            weights=models.video.R2Plus1D_18_Weights.DEFAULT,
        )
        # Replace 400-class fc with num_classes head
        model.fc = nn.Linear(model.fc.in_features, settings.num_classes, bias=True)
        # fc is randomly initialised — backbone has meaningful features
        logger.info(
            "Built R(2+1)D with Kinetics-400 pretrained backbone, "
            "random fc head (num_classes=%d)", settings.num_classes,
        )

    else:
        # ---- Original checkpoint path (may contain empty-trained weights) ----
        model = models.video.r2plus1d_18(weights=None, progress=False)
        model.fc = nn.Linear(model.fc.in_features, settings.num_classes, bias=True)

        checkpoint_args = EasyDict({
            "base_model_name": settings.base_model_name,
            "start_epoch": settings.start_epoch,
            "lr": settings.lr,
            "model_path": settings.model_path,
        })
        model = load_weights(model, checkpoint_args)

        if strategy == "pretrained_finetune":
            # Overlay pretrained backbone on top of the checkpoint fc
            pretrained_sd = models.video.r2plus1d_18(
                weights=models.video.R2Plus1D_18_Weights.DEFAULT,
            ).state_dict()
            own_sd = model.state_dict()
            loaded = 0
            for k, v in pretrained_sd.items():
                if k.startswith("fc."):
                    continue
                if k in own_sd and v.shape == own_sd[k].shape:
                    own_sd[k] = v
                    loaded += 1
            model.load_state_dict(own_sd)
            logger.info(
                "Overlaid %d pretrained backbone layers on checkpoint fc", loaded,
            )

    model = model.to(device)
    model.eval()
    return model
