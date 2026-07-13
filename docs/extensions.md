# AGU Extension Points

AGU exposes a typed pipeline and lightweight extension points for models,
trackers, storage, stages, and integrations. Existing direct registries remain
supported; new distributions should also publish plugin metadata.

## Plugin Discovery Contract

An external distribution declares a registration callable:

```toml
[project.entry-points."agu.plugins"]
my_plugin = "my_package.plugin:register"
```

The callable receives `app.plugins.PluginRegistry` and registers one or more
`PluginSpec` values. It may also call `register_analysis_stage()` at the
`before_dispatch` or `after_dispatch` hook. Each spec declares `kind`,
`capabilities`, `version`, optional import requirements, source, and
description. See `examples/plugins/minimal_plugin.py` for a stage that adds
non-secret metadata to the real analysis manifest.

Inspect the environment without starting the service or loading model weights:

```bash
agu plugins list
agu plugins doctor
agu plugins doctor --strict
```

`--strict` is intended for a deployment profile that expects every listed
optional integration. Normal diagnostics report missing optional adapters but
exit successfully.

## Pipeline Stage Contract

`app.analysis.pipeline` provides `PipelineContext`, `AnalysisStage`,
`CallableStage`, and `PipelineRunner`. The current service delegates through the
`analysis.validate`, `analysis.dispatch`, and `analysis.finalize` built-ins plus
before/after extension hooks, so single and segmented behavior stay unchanged
while future extraction can move tracking, inference, fusion, audit, and
persistence into separately testable stages.

Stage names are unique, every run records status and duration in
`result.pipeline_manifest`, and exceptions propagate unchanged after a failed
trace entry is recorded.

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

Detector, tracker, ReID, OCR, pose, VLM, and storage integrations should remain
optional adapters with a documented fallback. AGU owns stable schemas, task and
segment orchestration, basketball evidence reconciliation, and evaluation.
