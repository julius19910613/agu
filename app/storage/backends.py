from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class StoredArtifact:
    path: str
    url: str | None = None


class StorageBackend(Protocol):
    def write_json(self, name: str, payload: Any) -> StoredArtifact:
        ...

    def write_bytes(self, name: str, payload: bytes) -> StoredArtifact:
        ...

    def copy_file(self, source: str | Path, name: str | None = None) -> StoredArtifact:
        ...


class LocalStorageBackend:
    """Local filesystem storage backend for open-source/self-hosted AGU."""

    def __init__(self, root_dir: str | Path, public_base_url: str | None = None) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.public_base_url = public_base_url.rstrip("/") if public_base_url else None
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_name(self, name: str) -> Path:
        candidate = (self.root_dir / name).resolve()
        if self.root_dir != candidate and self.root_dir not in candidate.parents:
            raise ValueError(f"Storage path escapes root: {name}")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def _artifact(self, path: Path) -> StoredArtifact:
        url = None
        if self.public_base_url:
            rel = path.relative_to(self.root_dir).as_posix()
            url = f"{self.public_base_url}/{rel}"
        return StoredArtifact(path=str(path), url=url)

    def write_json(self, name: str, payload: Any) -> StoredArtifact:
        path = self._resolve_name(name)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._artifact(path)

    def write_bytes(self, name: str, payload: bytes) -> StoredArtifact:
        path = self._resolve_name(name)
        path.write_bytes(payload)
        return self._artifact(path)

    def copy_file(self, source: str | Path, name: str | None = None) -> StoredArtifact:
        source_path = Path(source)
        target = self._resolve_name(name or source_path.name)
        shutil.copy2(source_path, target)
        return self._artifact(target)
