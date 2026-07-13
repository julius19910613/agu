#!/usr/bin/env python3
"""Validate lightweight open-source baseline artifacts.

This script intentionally avoids loading model weights or video files. It checks
that public examples and documentation stay aligned with AGU's Pydantic API
schemas, so contributors can validate the repository without private assets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Missing required artifact: {path.relative_to(ROOT)}")


def validate_examples() -> None:
    from app.analysis.schemas import AnalysisRequest, AnalysisTaskStatusResponse

    request_path = ROOT / "examples/sample_request.json"
    output_path = ROOT / "examples/sample_output.json"

    require_file(request_path)
    require_file(output_path)

    try:
        AnalysisRequest.model_validate(load_json(request_path))
    except ValidationError as exc:
        raise AssertionError(f"sample_request.json does not match AnalysisRequest: {exc}") from exc

    try:
        AnalysisTaskStatusResponse.model_validate(load_json(output_path))
    except ValidationError as exc:
        raise AssertionError(f"sample_output.json does not match AnalysisTaskStatusResponse: {exc}") from exc


def validate_docs() -> None:
    required_docs = [
        ROOT / "CONTRIBUTING.md",
        ROOT / "CODE_OF_CONDUCT.md",
        ROOT / "SECURITY.md",
        ROOT / "CITATION.cff",
        ROOT / "CHANGELOG.md",
        ROOT / "LICENSE",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "pyproject.toml",
        ROOT / "docs/api.md",
        ROOT / "docs/README.md",
        ROOT / "docs/checkpoints.md",
        ROOT / "docs/datasets.md",
        ROOT / "docs/extensions.md",
        ROOT / "docs/model-card.md",
        ROOT / "docs/open-source-scope-assessment.md",
        ROOT / "docs/release-notes.md",
    ]
    for path in required_docs:
        require_file(path)

    api_text = (ROOT / "docs/api.md").read_text(encoding="utf-8")
    for marker in [
        "POST /api/v1/analysis/run",
        "GET /api/v1/analysis/status/{task_id}",
        "Stable Output Fields",
    ]:
        if marker not in api_text:
            raise AssertionError(f"docs/api.md missing marker: {marker}")

    model_card = (ROOT / "docs/model-card.md").read_text(encoding="utf-8")
    for marker in [
        "OpenCV BGR",
        "112x112",
        "macro-F1",
        "balanced accuracy",
    ]:
        if marker not in model_card:
            raise AssertionError(f"docs/model-card.md missing marker: {marker}")

    extensions = (ROOT / "docs/extensions.md").read_text(encoding="utf-8")
    for marker in ["Plugin Discovery Contract", "Pipeline Stage Contract", "Model Registry", "Tracker Registry", "Storage Backend"]:
        if marker not in extensions:
            raise AssertionError(f"docs/extensions.md missing marker: {marker}")

    release_notes = (ROOT / "docs/release-notes.md").read_text(encoding="utf-8")
    for marker in ["Source Statement", "License Statement", "Dataset Policy", "Weight Distribution Policy"]:
        if marker not in release_notes:
            raise AssertionError(f"docs/release-notes.md missing marker: {marker}")

    datasets = (ROOT / "docs/datasets.md").read_text(encoding="utf-8")
    for marker in ["Supported Training Layout", "SpaceJam Notes", "Annotation Format"]:
        if marker not in datasets:
            raise AssertionError(f"docs/datasets.md missing marker: {marker}")

    for path in [
        ROOT / "examples/benchmark/ground_truth.json",
        ROOT / "examples/benchmark/golden_metrics.json",
        ROOT / ".github/workflows/ci.yml",
        ROOT / ".github/workflows/release.yml",
    ]:
        require_file(path)


def validate_extension_points() -> None:
    from app.analysis.tracker_registry import list_tracker_backends
    from app.models.registry import list_model_loaders

    if "r2plus1d" not in list_model_loaders():
        raise AssertionError("Default r2plus1d model loader is not registered")

    if "YOLO" not in list_tracker_backends():
        raise AssertionError("Default YOLO tracker backend is not registered")


def main() -> int:
    try:
        validate_examples()
        validate_docs()
        validate_extension_points()
    except AssertionError as exc:
        print(f"Open-source baseline validation failed: {exc}")
        return 1

    print("Open-source baseline validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
