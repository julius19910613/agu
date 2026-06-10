from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
from torchvision import models

from app.config import Settings

logger = logging.getLogger(__name__)


def build_r2plus1d_model(
    settings: Settings,
    device: torch.device | None = None,
) -> nn.Module:
    """Build and return a R(2+1)D-18 model with checkpoint weights loaded.

    Supports two loading strategies:
    - ``best.pt`` convention: When ``base_model_name`` is ``"best"``, loads
      ``{model_path}/best.pt`` directly (train_mac.py saves best models as
      ``best.pt`` with ``{"state_dict": ..., "optimizer": ..., "epoch": N}``).
    - Legacy convention: Otherwise falls back to ``utils.checkpoints.load_weights``
      which assembles the path as ``{model_path}/{base_model_name}_{epoch}_{lr}.pt``.

    Args:
        settings: Application settings containing model_path, base_model_name,
            start_epoch, lr, and num_classes.
        device: Target device. Auto-detected if None.

    Returns:
        The loaded model in eval mode on the target device.
    """
    device = device or torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    model = models.video.r2plus1d_18(weights=None, progress=False)
    model.fc = nn.Linear(model.fc.in_features, settings.num_classes, bias=True)

    if settings.base_model_name == "best":
        # Load directly from best.pt (train_mac.py convention)
        best_path = os.path.join(settings.model_path, "best.pt")
        logger.info("Loading best checkpoint from %s", best_path)
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        # Filter to only keys that exist in the current model (skip mismatches)
        model_dict = model.state_dict()
        loaded = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(loaded)
        model.load_state_dict(model_dict)
        logger.info(
            "Loaded %d/%d parameters from best.pt (epoch %d)",
            len(loaded), len(model_dict), checkpoint.get("epoch", -1),
        )
    else:
        # Legacy loading via utils.checkpoints.load_weights
        from easydict import EasyDict
        from utils.checkpoints import load_weights

        checkpoint_args = EasyDict(
            {
                "base_model_name": settings.base_model_name,
                "start_epoch": settings.start_epoch,
                "lr": settings.lr,
                "model_path": settings.model_path,
            }
        )
        model = load_weights(model, checkpoint_args)

    model = model.to(device)
    model.eval()
    return model
