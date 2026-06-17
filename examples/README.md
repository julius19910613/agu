# AGU Examples

This directory contains small request and response examples for the AGU analysis API.

Run the API first:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Submit the sample request:

```bash
curl -sS -X POST http://127.0.0.1:8765/api/v1/analysis/run \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

Or use the CLI client:

```bash
python -m app.cli analyze --video examples/lebron_shoots.mp4 --vlm-mode off --max-frames 120 --no-generate-video
```
