"""Dependency-light plugin discovery and capability diagnostics for AGU.

Third-party packages expose a callable in the ``agu.plugins`` entry-point
group. The callable receives this module's global registry and registers one or
more :class:`PluginSpec` instances. Importing this module never imports torch,
OpenCV, Ultralytics, or other optional runtime backends.
"""

from __future__ import annotations

import builtins
from dataclasses import asdict, dataclass, field
from importlib import metadata
from importlib.util import find_spec
from typing import Callable, Iterable, Literal

PluginKind = Literal["model", "tracker", "storage", "stage", "integration"]
AvailabilityCheck = Callable[[], tuple[bool, str]]


@dataclass(frozen=True)
class PluginSpec:
    name: str
    kind: PluginKind
    capabilities: tuple[str, ...]
    version: str = "unknown"
    description: str = ""
    requires: tuple[str, ...] = ()
    source: str = "builtin"
    availability_check: AvailabilityCheck | None = field(default=None, repr=False, compare=False)

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.name.strip().lower()}"

    def diagnose(self) -> dict[str, object]:
        missing = [module for module in self.requires if find_spec(module) is None]
        available = not missing
        reason = "available" if available else f"missing modules: {', '.join(missing)}"
        if available and self.availability_check is not None:
            try:
                available, reason = self.availability_check()
            except Exception as exc:  # diagnostics must not break CLI startup
                available, reason = False, f"availability check failed: {exc}"
        payload = asdict(self)
        payload.pop("availability_check", None)
        payload.update({"available": available, "reason": reason})
        return payload


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, PluginSpec] = {}
        self._discovery_errors: list[dict[str, str]] = []
        self._discovered = False

    def register(self, plugin: PluginSpec, *, replace: bool = False) -> None:
        key = plugin.key
        if key in self._plugins and not replace:
            raise ValueError(f"Plugin already registered: {key}")
        self._plugins[key] = plugin

    def list(self, kind: PluginKind | None = None) -> builtins.list[PluginSpec]:
        plugins: Iterable[PluginSpec] = self._plugins.values()
        if kind is not None:
            plugins = (plugin for plugin in plugins if plugin.kind == kind)
        return sorted(plugins, key=lambda plugin: (plugin.kind, plugin.name))

    def discover(self, *, force: bool = False) -> builtins.list[dict[str, str]]:
        if self._discovered and not force:
            return builtins.list(self._discovery_errors)
        self._discovered = True
        self._discovery_errors.clear()
        entry_points = metadata.entry_points()
        selected: Iterable[metadata.EntryPoint] = entry_points.select(group="agu.plugins")
        for entry_point in selected:
            try:
                register = entry_point.load()
                register(self)
            except Exception as exc:
                self._discovery_errors.append(
                    {"entry_point": entry_point.name, "value": entry_point.value, "error": str(exc)}
                )
        return builtins.list(self._discovery_errors)

    def diagnostics(self, kind: PluginKind | None = None) -> dict[str, object]:
        return {
            "plugins": [plugin.diagnose() for plugin in self.list(kind)],
            "discovery_errors": builtins.list(self._discovery_errors),
        }

    def clear(self) -> None:
        self._plugins.clear()
        self._discovery_errors.clear()
        self._discovered = False


registry = PluginRegistry()


def register_builtin_plugins() -> None:
    """Register built-ins without importing heavyweight implementation modules."""
    builtins = (
        PluginSpec(
            name="r2plus1d-v3",
            kind="model",
            capabilities=("action-recognition", "v3-preprocessing"),
            version="1",
            requires=("torch", "torchvision"),
            description="AGU R(2+1)D action classifier with the stable v3 preprocessing contract.",
        ),
        PluginSpec(
            name="opencv-legacy",
            kind="tracker",
            capabilities=("person-tracking",),
            version="1",
            requires=("cv2",),
            description="OpenCV legacy tracker adapter.",
        ),
        PluginSpec(
            name="ultralytics",
            kind="tracker",
            capabilities=("person-detection", "person-tracking", "reid"),
            version="1",
            requires=("ultralytics",),
            description="Optional Ultralytics ByteTrack/BoT-SORT adapter; review AGPL/commercial licensing.",
        ),
        PluginSpec(
            name="local",
            kind="storage",
            capabilities=("artifact-json", "artifact-bytes", "artifact-copy"),
            version="1",
            description="Local filesystem artifact storage.",
        ),
        PluginSpec(
            name="analysis-pipeline-v1",
            kind="stage",
            capabilities=("single-video", "segmented-video", "before-dispatch-hook", "after-dispatch-hook"),
            version="1",
            description="Versioned AGU pipeline with validation, dispatch, extension hooks, and finalization.",
        ),
        PluginSpec(
            name="rapidocr",
            kind="integration",
            capabilities=("scoreboard-ocr",),
            version="1",
            requires=("rapidocr_onnxruntime",),
            description="Optional offline scoreboard OCR adapter.",
        ),
    )
    for plugin in builtins:
        registry.register(plugin, replace=True)


register_builtin_plugins()
