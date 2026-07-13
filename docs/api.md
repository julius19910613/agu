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
  "tracker_min_appear_abs": 5,
  "segmented_analysis": true,
  "action_vid_stride": 24,
  "tracking_fps": 8.0,
  "yolo_imgsz": 320,
  "max_players_per_segment": 12,
  "yolo_device": "cpu",
  "tracker_backend": "botsort",
  "yolo_tracker_config": "botsort.yaml",
  "yolo_reid_enabled": false,
  "yolo_reid_model": "auto",
  "r2plus1d_device": "mps_if_available",
  "long_video_mode": false,
  "segment_duration_sec": 15.0,
  "segment_overlap_sec": 2.0,
  "segment_start_sec": 0.0,
  "segment_end_sec": null,
  "max_segments": null,
  "vlm_audit": true,
  "vlm_audit_frames": 6
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
POST /api/v1/analysis/tasks/{task_id}/cancel
POST /api/v1/analysis/tasks/{task_id}/retry
```

Response body:

```json
{
  "task_id": "b1f8...",
  "status": "completed",
  "progress": 100,
  "error": null,
  "result": {
    "schema_version": "1.0",
    "pipeline_manifest": {
      "pipeline_version": "1",
      "stages": [
        {"stage": "analysis.validate", "status": "completed", "duration_ms": 0.01},
        {"stage": "analysis.dispatch", "status": "completed", "duration_ms": 12.3},
        {"stage": "analysis.finalize", "status": "completed", "duration_ms": 0.01}
      ],
      "metadata": {"schema_version": "1.0", "segmented_analysis": false}
    },
    "video": "examples/lebron_shoots.mp4",
    "created_at_unix": 1765890000.0,
    "runtime_seconds": 12.34,
    "frame_size": {"width": 1280, "height": 720},
    "seq_length": 16,
    "vid_stride": 8,
    "tracker_backend": "bytetrack",
    "tracker_config": "bytetrack.yaml",
    "reid_enabled": false,
    "vlm_mode": "off",
    "ollama_model": null,
    "records": [],
    "summary": {
      "clip_count": 0,
      "action_counts": {},
      "needs_review_count": 0,
      "source_counts": {}
    },
    "long_video": null
  }
}
```

Task status also includes `request_id`, `created_at_unix`, and
`updated_at_unix`. Cancellation is cooperative: AGU stops at the next pipeline
progress boundary and preserves `status=cancelled`; a completed/failed task is
not rewritten. `retry` creates a new task from the stored request and is allowed
only for failed or cancelled in-memory tasks. Task requests/results remain
in-memory and are lost after process restart.

`max_runtime_sec` sets a per-request cooperative deadline. When omitted,
`BASKETBALL_ANALYSIS_TIMEOUT_SEC` is used; `0` disables it. `max_frames`,
`max_segments`, segment bounds, and this deadline form the current resource
budget controls. Long detector/model calls can finish before cancellation or a
deadline is observed.

For running tasks, `progress` is a best-effort integer from 0 to 100. AGU now
updates it at major pipeline stages such as tracker setup, window cropping,
R(2+1)D inference, fusion/VLM verification, identity extraction, segment audit,
and post-processing. It is intended for CLI/UI polling feedback, not as a
durable timing contract; completed and failed tasks still finish at `100`.

## Stable Output Fields

`result.schema_version` versions the stable JSON contract. Additive fields do
not change this value; an incompatible removal or semantic change requires a
new major schema version. `result.pipeline_manifest` records the stage names,
status, elapsed milliseconds, and non-secret execution metadata. Durations are
diagnostic and are not a performance SLA.

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

`result.player_identity_features[]` contains local-track identity evidence used
by long-video stitching. `appearance_embedding` is generated by the configured
identity embedding backend. The default backend is
`torchvision_mobilenet_v3_small`; `torchreid_osnet_x0_25` can be enabled when
the optional local `torchreid` package and weights are available, with
`sidecar_hsv_hist` retained as an explicit lightweight fallback.

| Field | Type | Meaning |
| --- | --- | --- |
| `player` | integer | Segment-local player index |
| `segment_id` | integer or null | Segment ID when available |
| `local_player_id` | string or null | Segment-local player key |
| `start_frame` / `end_frame` | integer | Feature frame range |
| `first_center` / `last_center` | array | First/last observed bbox centers |
| `appearance_signature` | object | Mean HSV/RGB plus torso luminance and jersey-dark-ratio signature retained for explainability/fallback |
| `appearance_embedding` | array | Model/fallback appearance embedding used for cosine similarity |
| `embedding_model` | string | Embedding source/model identifier |
| `embedding_dim` | integer | Embedding vector length |
| `face_embedding` | array | Optional embedding of OpenCV-detected frontal-face crops; empty when no usable face is visible |
| `face_embedding_model` | string or null | Face detector plus configured embedding backend identifier |
| `face_sample_count` | integer | Number of sampled player crops with a usable frontal face |
| `track_coverage` | number | Fraction of sampled frames with usable boxes |
| `method` | string | Feature extraction method |
| `sampled_boxes` | array | Sampled frame-level boxes used for duplicate-ID conflict and overlap checks |
| `jersey_number_candidates` | array | Optional VLM/OCR jersey-number candidates for this local track |

## Long Video Mode

AGU analyzes videos through overlapped segments by default
(`segmented_analysis=true`). The overlap prevents actions near a segment boundary
from being missed, and global summaries count only the owned time range for each
segment so overlap does not double-count clips. The standard async task envelope
is unchanged, but completed task results include `result.long_video`.

Important request fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `segmented_analysis` | boolean | Enables unified overlapped segment analysis for any video |
| `long_video_mode` | boolean | Backward-compatible alias that also enables segmented analysis |
| `segment_duration_sec` | number | Segment length in seconds |
| `segment_overlap_sec` | number | Overlap between adjacent segments |
| `segment_start_sec` | number | Optional start offset for targeted smoke checks |
| `segment_end_sec` | number or null | Optional end offset |
| `max_segments` | integer or null | Optional segment cap for validation |
| `vlm_audit` | boolean | Enables VLM contact-sheet audit per segment |
| `vlm_audit_frames` | integer | Frames sampled into each segment contact sheet |
| `action_vid_stride` | integer or null | Optional accelerated action clip stride used when `vid_stride` is not set |
| `tracking_fps` | number or null | Optional YOLO tracking frame-rate cap |
| `yolo_imgsz` | integer or null | Optional YOLO inference image size |
| `max_players_per_segment` | integer or null | Optional cap for strongest active YOLO player tracks |
| `yolo_device` | string or null | Optional YOLO device override; current Mac default is `cpu` |
| `tracker_backend` | string or null | Optional open-source tracker adapter: `bytetrack`, `botsort`, or `custom` |
| `yolo_tracker_config` | string or null | Optional Ultralytics tracker YAML, for example `bytetrack.yaml`, `botsort.yaml`, or a custom YAML path |
| `yolo_reid_enabled` | boolean or null | Enables generated BoT-SORT ReID config when supported |
| `yolo_reid_model` | string or null | ReID model value for BoT-SORT, for example `auto` or a classifier model path |
| `identity_embedding_backend` | string or null | Optional identity embedding backend: `torchvision_mobilenet_v3_small`, `torchreid_osnet_x0_25`, or `sidecar_hsv_hist` |
| `identity_embedding_weights` | string or null | Optional identity embedding weights: `default`, `imagenet1k_v1`, or `none` |
| `identity_embedding_device` | string or null | Optional identity embedding device: `auto`, `cpu`, `cuda`, `mps`, or `mps_if_available` |
| `jersey_number_vlm_enabled` | boolean or null | Enables optional VLM jersey-number reading from sampled player crops |
| `jersey_number_vlm_frames` | integer or null | Number of player crops sent to VLM for jersey-number reading |
| `confirmed_identity_merges` | array | Optional confirmed global-player merge instructions used to emit `long_video.merged_players[]` |
| `vlm_identity_merge_enabled` | boolean or null | Enables optional VLM post-processing over duplicate identity candidates |
| `vlm_identity_merge_max_candidates` | integer or null | Maximum duplicate candidates sent to VLM for merge review |
| `vlm_identity_merge_confidence` | number or null | Minimum VLM same-player confidence required to emit a confirmed merge |
| `r2plus1d_device` | string or null | Optional R(2+1)D device override: `auto`, `cpu`, `cuda`, `mps`, or `mps_if_available` |

Face identity is configured at service level with `BASKETBALL_FACE_IDENTITY_BACKEND`, `BASKETBALL_FACE_DETECTION_MODEL_PATH`, `BASKETBALL_FACE_RECOGNITION_MODEL_PATH`, and `BASKETBALL_FACE_DETECTION_SCORE_THRESHOLD`. The default `opencv_sface_if_available` backend uses local OpenCV YuNet and SFace ONNX models and falls back to Haar sampling when either model is unavailable.

SFace output is quality-gated per local track: at least two face samples must form a majority cluster with internal cosine similarity of at least `0.50`. Unstable detections from track-ID switches, back-facing heads, or background people are omitted instead of being averaged into identity evidence. The resulting quality is exposed as `player_identity_features[].face_embedding_quality`.

`result.long_video.segments[]` contains segment-level summary and VLM audit status:

| Field | Type | Meaning |
| --- | --- | --- |
| `segment_id` | integer | Segment sequence index |
| `start_sec` / `end_sec` | number | Segment time range |
| `start_frame` / `end_frame` | integer | Segment frame range |
| `player_count` | integer | AGU tracked player count in the segment |
| `summary` | object | Segment-level action/source/review counts |
| `vlm_audit` | object or null | VLM estimate of visible player count and actions |
| `audit_status` | string | `pass`, warning, or failure status |
| `audit_notes` | array | Human-readable mismatch notes |

`result.long_video.players[]` contains segment-local player summaries:

| Field | Type | Meaning |
| --- | --- | --- |
| `player_id` | string | Segment-local player key, for example `segment_0:player_3` |
| `global_player_id` | string or null | Conservative cross-segment identity candidate |
| `identity_confidence` | number | Confidence for the global identity candidate |
| `identity_method` | string | Identity stitching method, currently `appearance_continuity_stitch_v2` |
| `identity_evidence` | array | Evidence used for the identity stitch |
| `segments_seen` | integer | Number of segments in which this local track appears |
| `clip_count` | integer | Owned clips counted for this player |
| `action_counts` | object | Final action histogram for this player |
| `needs_review_count` | integer | Player clips marked for review |
| `average_confidence` | number | Average final action confidence |
| `statistics` | object | Estimated points, assists, rebounds, blocks, and steals |

`statistics` includes `status`, `estimated_fields`, and `candidate_fields`.
The current `action_proxy_v1` implementation marks assists as an estimated
field. `shoot` clips are exposed as `shot_attempts` and
`point_candidate_count`; points remain candidate evidence until a made-shot,
free-throw, or scoreboard-linked scoring event is confirmed. Blocks, rebounds,
and steals remain candidate fields until event confirmation is available.

`result.long_video.scoreboard_summary` is populated when
`scoreboard_audit=true` (enabled by CLI `accurate` and `vlm-full` presets). It
contains sampled scoreboard checkpoints plus the latest readable final score:

| Field | Type | Meaning |
| --- | --- | --- |
| `enabled` | boolean | Whether scoreboard audit was requested |
| `status` | string | `ok`, `disabled`, `no_readable_scoreboard`, `inconsistent_scoreboard`, or VLM/error status |
| `final_left_score` / `final_right_score` | integer or null | Latest readable left/right score |
| `final_total_points` | integer or null | Sum of latest readable left/right score |
| `final_time_sec` | number or null | Timestamp of the checkpoint used as final score |
| `checkpoints[]` | array | Independent VLM reads from OpenCV-ranked burst frames and temporal LED fusion crops |

`result.long_video.identity_duplicate_candidates[]` contains conservative
post-stitch duplicate-ID review candidates. These candidates do not rewrite
statistics automatically; they expose evidence for VLM or human confirmation.

| Field | Type | Meaning |
| --- | --- | --- |
| `left_global_player_id` / `right_global_player_id` | string | Possible duplicate global IDs |
| `confidence` | number | Merge-review confidence from appearance, color, action, and temporal evidence |
| `status` | string | Candidate status, currently requires VLM or human confirmation |
| `recommended_action` | string | Suggested action, for example `review_merge` |
| `left_local_player_ids` / `right_local_player_ids` | array | Segment-local tracks behind each global ID |
| `evidence` | array | Positive duplicate evidence |
| `conflict_evidence` | array | Hard or soft conflict evidence |

`result.long_video.confirmed_identity_merges[]` echoes confirmed merge
instructions supplied in the request. `result.long_video.merged_players[]`
contains the aggregate player summaries produced from those confirmed merges.
This is a separate audit-safe view: original `players[]` and `records[]` remain
unchanged.

`result.long_video.identity_graph_summary` provides an overview of the review
graph: node count, duplicate candidate count, confirmed merge count, VLM merge
decision count, method, and notes. It is informational and does not mutate
player summaries.

When `vlm_identity_merge_enabled=true`, AGU sends the top
`identity_duplicate_candidates[]` to the configured VLM using labeled LEFT/RIGHT
player crop contact sheets. Each VLM response is recorded in
`result.long_video.identity_merge_decisions[]`. Only available decisions with
`is_same_player=true` and confidence above `vlm_identity_merge_confidence` are
converted into additional `confirmed_identity_merges[]`.

| Field | Type | Meaning |
| --- | --- | --- |
| `left_global_player_id` / `right_global_player_id` | string | Reviewed duplicate candidate pair |
| `is_same_player` | boolean | VLM same-player decision |
| `confidence` | number | VLM confidence for the decision |
| `canonical_global_player_id` | string or null | Canonical ID chosen by VLM, constrained to the reviewed pair |
| `merged_global_player_ids` | array | IDs to merge into the canonical ID |
| `reason` | string | VLM rationale |
| `evidence` | array | VLM visual evidence |
| `available` | boolean | Whether VLM returned a usable decision |

| Field | Type | Meaning |
| --- | --- | --- |
| `player_id` | string | Synthetic merged player key, for example `merged:player_004` |
| `global_player_id` | string | Canonical confirmed global player ID |
| `merged_from_global_player_ids` | array | Canonical plus merged global IDs included in the aggregate |
| `merge_confidence` | number | Confidence supplied by the confirmed merge source |
| `merge_evidence` | array | Confirmation and identity evidence retained for audit |
| `action_counts` | object | Aggregated final action histogram |
| `statistics` | object | Aggregated estimated points, assists, rebounds, blocks, and steals |

`statistics` is an action-proxy estimate, not official box-score truth. In the
current model, `shoot` actions are shot-attempt/point candidates rather than
made scores; assists are estimated from `pass` actions. Points, blocks,
rebounds, and steals require event confirmation and should be read from
`result.long_video.event_candidates` or `result.long_video.scoreboard_summary`
until ball/rim/possession/scoreboard confirmation is available.

`result.long_video.event_candidates[]` contains low/medium-confidence event
evidence:

| Field | Type | Meaning |
| --- | --- | --- |
| `event_type` | string | `block_candidate`, `rebound_candidate`, or `steal_candidate` |
| `player_id` | string or null | Candidate global player ID when available |
| `segment_id` | integer or null | Segment where the candidate was observed |
| `start_frame` / `end_frame` | integer | Candidate frame range |
| `confidence` | number | Candidate confidence, not official-stat certainty |
| `method` | string | Candidate generation method |
| `status` | string | Confirmation requirement |
| `evidence` | array | Human-readable evidence notes |
| `owner_candidates` | array | Ranked nearby player candidates for actor selection or review |

`block_candidate` is intentionally conservative: a single block-classified clip
is promoted only when its average confidence is high enough; otherwise AGU keeps
the action record but does not expose it as an event candidate. This avoids
turning isolated R(2+1)D block noise into a box-score event.

When segment VLM audit is available with confident action evidence, AGU also uses
it as negative evidence for `block_candidate`: if the audit does not include a
visible block, block candidates from that segment are suppressed. The underlying
clip-level `records[]` and `summary.action_counts` remain unchanged so reviewers
can still inspect model disagreements.

Each `owner_candidates[]` item contains:

| Field | Type | Meaning |
| --- | --- | --- |
| `global_player_id` | string | Candidate event-owner player ID |
| `local_player_ids` | array | Segment-local tracks supporting the candidate |
| `rank` | integer | Rank within the event candidate |
| `score` | number | Deterministic owner score from confidence, temporal proximity, action hints, identity confidence, and support |
| `clip_count` | integer | Nearby clips for the candidate |
| `action_match_count` | integer | Nearby clips whose action matches the event hint set |
| `avg_confidence` | number | Average final action confidence for relevant clips |
| `nearest_frame_gap` | integer | Nearest supporting clip distance from the event center |
| `evidence` | array | Human-readable owner-scoring evidence |

Known MVP boundary: `global_player_id` is a conservative candidate generated
from adjacent segment action, body appearance, torso/jersey luminance, optional
frontal-face evidence, and track continuity. OpenCV Haar face detection is a
local optional sidecar: profiles, occluded faces, and small crops fall back to
clothing/body evidence instead of fabricating face evidence. Duplicate identity
candidates remain review-only until jersey number, VLM, or human confirmation.

Scoreboard audit uses the optional `rapidocr_if_available` backend before VLM
fallback. Install `requirements-ocr.txt` for local offline OCR. Configure
`BASKETBALL_SCOREBOARD_OCR_BACKEND=off|rapidocr|rapidocr_if_available` and
`BASKETBALL_SCOREBOARD_OCR_CONFIDENCE`; OCR candidates still pass the same
burst, cross-anchor, monotonic-score, and clock consistency reconciliation.

## Error Behavior

- Invalid or missing `video_path`: `400`.
- Missing task: `404`.
- Background analysis failure: task status becomes `failed`, with `error` populated.

## Reproducibility Notes

- Task state is in memory and is lost when the process restarts.
- `video_path` must be visible to the AGU process/container.
- By default `video_path` must be under the AGU working directory. Set `BASKETBALL_ALLOWED_VIDEO_ROOTS` to a comma-separated list of additional trusted video directories.
- AGU writes JSON output to `BASKETBALL_OUTPUT_DIR`.
- AGU writes annotated video output to `BASKETBALL_VIDEO_OUTPUT_DIR` when `generate_video=true`.
