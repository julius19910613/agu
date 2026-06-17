from __future__ import annotations

from collections.abc import Callable
from typing import Any


TrackerExtractor = Callable[..., Any]

_TRACKER_BACKENDS: dict[str, TrackerExtractor] = {}


def normalize_tracker_name(name: str) -> str:
    return name.strip().upper().replace("-", "_")


def register_tracker_backend(name: str, extractor: TrackerExtractor, *, replace: bool = False) -> None:
    """Register a tracker backend by name.

    The default AGU pipeline still calls `extract_tracked_frames` directly. This
    registry is the public extension point for future tracker adapters.
    """
    normalized = normalize_tracker_name(name)
    if not replace and normalized in _TRACKER_BACKENDS:
        raise ValueError(f"Tracker backend already registered: {normalized}")
    _TRACKER_BACKENDS[normalized] = extractor


def get_tracker_backend(name: str) -> TrackerExtractor:
    normalized = normalize_tracker_name(name)
    try:
        return _TRACKER_BACKENDS[normalized]
    except KeyError as exc:
        available = ", ".join(list_tracker_backends())
        raise ValueError(f"Unknown tracker backend '{name}'. Available: {available}") from exc


def list_tracker_backends() -> list[str]:
    return sorted(_TRACKER_BACKENDS)


def extract_with_registered_tracker(name: str, **kwargs: Any) -> Any:
    return get_tracker_backend(name)(tracker_type=name, **kwargs)


def _register_defaults() -> None:
    from app.analysis.tracking import TRACKER_TYPES, extract_tracked_frames

    register_tracker_backend("YOLO", extract_tracked_frames, replace=True)
    for tracker_name in TRACKER_TYPES:
        register_tracker_backend(tracker_name, extract_tracked_frames, replace=True)


_register_defaults()
