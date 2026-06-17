from __future__ import annotations

import json

import pytest

from app.analysis.tracker_registry import (
    get_tracker_backend,
    list_tracker_backends,
    register_tracker_backend,
)
from app.models.registry import get_model_loader, list_model_loaders, register_model_loader
from app.storage.backends import LocalStorageBackend


def test_model_registry_defaults_and_custom_loader():
    assert "r2plus1d" in list_model_loaders()
    assert callable(get_model_loader("r2plus1d"))

    def loader(settings, device=None):
        return {"settings": settings, "device": device}

    register_model_loader("unit-test-model", loader, replace=True)
    assert get_model_loader("unit_test_model") is loader


def test_tracker_registry_defaults_and_custom_backend():
    assert "YOLO" in list_tracker_backends()
    assert callable(get_tracker_backend("yolo"))

    def extractor(**kwargs):
        return kwargs

    register_tracker_backend("unit-test-tracker", extractor, replace=True)
    assert get_tracker_backend("UNIT_TEST_TRACKER") is extractor


def test_local_storage_backend_writes_json_bytes_and_urls(tmp_path):
    storage = LocalStorageBackend(tmp_path, public_base_url="/static/outputs")

    json_artifact = storage.write_json("nested/result.json", {"ok": True})
    assert json_artifact.url == "/static/outputs/nested/result.json"
    assert json.loads((tmp_path / "nested/result.json").read_text()) == {"ok": True}

    bytes_artifact = storage.write_bytes("binary/blob.bin", b"agu")
    assert bytes_artifact.url == "/static/outputs/binary/blob.bin"
    assert (tmp_path / "binary/blob.bin").read_bytes() == b"agu"


def test_local_storage_backend_blocks_path_escape(tmp_path):
    storage = LocalStorageBackend(tmp_path)

    with pytest.raises(ValueError):
        storage.write_json("../escape.json", {"bad": True})
