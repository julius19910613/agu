# AGU Extension Points

AGU keeps its core pipeline small, but exposes lightweight extension points for models, trackers, and output storage.

## Model Registry

Default loaders:

- `r2plus1d`
- `r2plus1d-v3`

Example:

```python
from app.models.registry import register_model_loader


def build_my_model(settings, device=None):
    ...


register_model_loader("my-model", build_my_model)
```

Use:

```python
from app.models.registry import build_registered_model

model = build_registered_model("r2plus1d", settings, device)
```

## Tracker Registry

Default backends:

- `YOLO`
- OpenCV legacy tracker names such as `CSRT`, `KCF`, `MOSSE`

Example:

```python
from app.analysis.tracker_registry import register_tracker_backend


def extract_my_tracks(**kwargs):
    ...


register_tracker_backend("MY_TRACKER", extract_my_tracks)
```

## Storage Backend

The first supported backend is local filesystem storage:

```python
from app.storage.backends import LocalStorageBackend

storage = LocalStorageBackend("analysis_outputs", public_base_url="/static/outputs")
artifact = storage.write_json("result.json", {"ok": True})
```

Future adapters should follow the same `write_json`, `write_bytes`, and `copy_file` shape for S3, COS, or CloudBase Storage.

## Boundary

These extension points should not add business database writes, authentication, RBAC, or mini-program-specific behavior to AGU.
