# Local Service Curl Hook

This hook is the AGU post-development runtime gate. Use it after development
tasks that change Python service code, API behavior, configuration, task
orchestration, inference/tracking/VLM behavior, output schemas, or README/API
contract content.

Documentation-only changes that do not affect runtime behavior may record this
hook as not applicable, but README/API contract edits should still run the
README/code consistency check below.

## Steps

1. Start AGU locally on an available loopback port:

```bash
/Users/ppt/projects/agu/venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

If port `8765` is busy, choose another local port and use the same port for all
curl commands.

2. Verify health with curl:

```bash
curl -sS http://127.0.0.1:8765/health
curl -sS http://127.0.0.1:8765/ready
```

Expected responses:

```json
{"status":"ok"}
{"status":"ready"}
```

3. Submit a lightweight analysis task with curl:

```bash
curl -sS -X POST http://127.0.0.1:8765/api/v1/analysis/run \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "examples/lebron_shoots.mp4",
    "vlm_mode": "off",
    "generate_video": false,
    "segmented_analysis": false,
    "max_frames": 60
  }'
```

Expected response shape:

```json
{
  "task_id": "...",
  "status": "pending",
  "message": "Analysis started asynchronously. Please poll the status endpoint to query progress."
}
```

4. Poll the returned task:

```bash
curl -sS http://127.0.0.1:8765/api/v1/analysis/status/<task_id>
```

Expected response shape:

```json
{
  "task_id": "...",
  "status": "pending|processing|completed|failed",
  "progress": 0,
  "error": null,
  "result": null
}
```

For user-visible analysis behavior changes, poll until `completed` or record the
failure/error. For documentation-only or harness-only changes, health plus task
submission is sufficient unless the docs claim a completed result shape changed.

5. Check README/code consistency:

- Compare `README.md` quick-start commands and request parameters with
  `app/analysis/schemas.py`, `app/config.py`, and `docs/api.md`.
- If public request/response fields changed, update both `README.md` and
  `docs/api.md`.
- If startup, environment variables, output directories, or model/checkpoint
  assumptions changed, update `README.md`, `.env.example`, and the relevant
  harness docs.

6. Record the hook result:

- Add the curl commands and observed status/result shape to the final response.
- For non-trivial tasks, record the runtime check in
  `docs/harness/TASK-BOARD.md` or the relevant `docs/specs/TASK-*` artifact.
- If local service startup or curl cannot run because of environment limits,
  record the exact blocker and run the closest available focused test/harness
  check.
