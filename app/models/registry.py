from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from app.config import Settings


ModelLoader = Callable[["Settings", "torch.device | None"], Any]

_MODEL_LOADERS: dict[str, ModelLoader] = {}


def normalize_model_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def register_model_loader(name: str, loader: ModelLoader, *, replace: bool = False) -> None:
    """Register a model loader by name.

    External integrations can use this to add alternatives such as VideoMAE,
    SlowFast, or an MMAction2-backed model without changing AGU's API layer.
    """
    normalized = normalize_model_name(name)
    if not replace and normalized in _MODEL_LOADERS:
        raise ValueError(f"Model loader already registered: {normalized}")
    _MODEL_LOADERS[normalized] = loader


def get_model_loader(name: str) -> ModelLoader:
    normalized = normalize_model_name(name)
    try:
        return _MODEL_LOADERS[normalized]
    except KeyError as exc:
        available = ", ".join(list_model_loaders())
        raise ValueError(f"Unknown model loader '{name}'. Available: {available}") from exc


def list_model_loaders() -> list[str]:
    return sorted(_MODEL_LOADERS)


def build_registered_model(name: str, settings: "Settings", device: "torch.device | None" = None) -> Any:
    return get_model_loader(name)(settings, device)


def _register_defaults() -> None:
    from app.models.r2plus1d import build_r2plus1d_model

    register_model_loader("r2plus1d", build_r2plus1d_model, replace=True)
    register_model_loader("r2plus1d-v3", build_r2plus1d_model, replace=True)


_register_defaults()
