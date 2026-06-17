# AGU API Contract

AGU exposes a minimal FastAPI surface for basketball video action analysis. It does not implement authentication, RBAC, API gateway behavior, grouping, player profile management, or business database writes.

## Health

```http
GET /health
GET /ready
```

Expected response:

```json
{"status": "ok"}
```

`/ready` returns:

```json
{"status": "ready"}
```

## Submit Analysis

```http
POST /api/v1/analysis/run
```

Alias for external BFF integration:

```http
POST /api/v1/analysis/tasks
```

Request body:

```json
{
  "video_path": "examples/lebron_shoots.mp4",
  "vlm_mode": "off",
  "max_frames": 120,
  "generate_video": false,
  "tracker_conf_thres": 0.3,
  "tracker_iou_thres": 0.6,
  "tracker_min_appear_ratio": 0.02,
  "tracker_min_appear_abs": 5
}
```

Response body:

```json
{
  "task_id": "b1f8...",
  "status": "pending",
  "message": "Analysis started asynchronously. Please poll the status endpoint to query progress."
}
```

## Query Analysis

```http
GET /api/v1/analysis/status/{task_id}
```

Aliases:

```http
GET /api/v1/analysis/tasks/{task_id}
GET /api/v1/analysis/tasks/{task_id}/result
```

Response body:

```json
{
  "task_id": "b1f8...",
  "status": "completed",
  "progress": 100,
  "error": null,
  "result": {
    "video": "examples/lebron_shoots.mp4",
    "created_at_unix": 1765890000.0,
    "runtime_seconds": 12.34,
    "frame_size": {"width": 1280, "height": 720},
    "seq_length": 16,
    "vid_stride": 8,
    "vlm_mode": "off",
    "ollama_model": null,
    "records": [],
    "summary": {
      "clip_count": 0,
      "action_counts": {},
      "needs_review_count": 0,
      "source_counts": {}
    }
  }
}
```

## Stable Output Fields

`result.records[]` contains one row per player clip.

| Field | Type | Meaning |
| --- | --- | --- |
| `player` | integer | Player index in the tracked video |
| `clip_index` | integer | Clip sequence index for that player |
| `start_frame` | integer | First frame covered by the clip |
| `end_frame` | integer | Last frame covered by the clip |
| `r2plus1d` | object | Raw model prediction |
| `motion` | object | Motion feature summary |
| `vlm` | object or null | Optional VLM review result |
| `final` | object | Fused final action decision |

`result.summary` provides aggregate counts:

| Field | Type | Meaning |
| --- | --- | --- |
| `clip_count` | integer | Total analyzed clips |
| `action_counts` | object | Final action histogram |
| `needs_review_count` | integer | Number of clips marked for review |
| `source_counts` | object | Decision source histogram |

## Error Behavior

- Invalid or missing `video_path`: `400`.
- Missing task: `404`.
- Background analysis failure: task status becomes `failed`, with `error` populated.

## Reproducibility Notes

- Task state is in memory and is lost when the process restarts.
- `video_path` must be visible to the AGU process/container.
- AGU writes JSON output to `BASKETBALL_OUTPUT_DIR`.
- AGU writes annotated video output to `BASKETBALL_VIDEO_OUTPUT_DIR` when `generate_video=true`.
