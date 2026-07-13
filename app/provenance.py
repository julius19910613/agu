"""Non-secret runtime provenance helpers for reproducible analysis manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_provenance(model_path: str, base_model_name: str = "best") -> dict[str, Any]:
    root = Path(model_path).expanduser()
    preferred = root / "best.pt"
    candidates = [preferred, *sorted(root.glob(f"{base_model_name}*.pt"))] if root.is_dir() else [root]
    checkpoint = next((candidate for candidate in candidates if candidate.is_file()), None)
    if checkpoint is None:
        return {"status": "missing", "name": preferred.name}
    stat = checkpoint.stat()
    return {
        "status": "available",
        "name": checkpoint.name,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(checkpoint),
    }
