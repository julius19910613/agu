from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


TERMINAL_STATES = {"completed", "failed"}


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urlrequest.Request(url, data=data, headers=headers, method=method)
    try:
        with urlrequest.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw or "{}")
    except urlerror.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            body = json.loads(raw or "{}")
        except json.JSONDecodeError:
            body = {"raw": raw}
        raise SystemExit(f"HTTP {exc.code}: {json.dumps(body, ensure_ascii=False)}") from exc
    except urlerror.URLError as exc:
        raise SystemExit(f"Request failed: {exc}") from exc


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def analyze(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "video_path": args.video,
        "vlm_mode": args.vlm_mode,
        "max_frames": args.max_frames,
        "generate_video": args.generate_video,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    response = _request_json("POST", f"{args.api_url.rstrip('/')}/api/v1/analysis/run", payload)
    _print_json(response)

    if not args.poll:
        return 0

    task_id = response.get("task_id")
    if not task_id:
        raise SystemExit("Response did not contain task_id")

    while True:
        time.sleep(args.poll_interval)
        status = _request_json("GET", f"{args.api_url.rstrip('/')}/api/v1/analysis/status/{task_id}")
        _print_json(status)
        if status.get("status") in TERMINAL_STATES:
            return 0 if status.get("status") == "completed" else 1


def status(args: argparse.Namespace) -> int:
    response = _request_json("GET", f"{args.api_url.rstrip('/')}/api/v1/analysis/status/{args.task_id}")
    _print_json(response)
    return 0 if response.get("status") != "failed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="AGU command line client for the running FastAPI analysis service.",
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8765", help="AGU service base URL")

    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Submit an analysis task")
    analyze_parser.add_argument("--video", required=True, help="Video path visible to the AGU service")
    analyze_parser.add_argument("--vlm-mode", default="off", choices=["off", "low-confidence", "always"])
    analyze_parser.add_argument("--max-frames", type=int, default=120)
    analyze_parser.add_argument("--generate-video", action=argparse.BooleanOptionalAction, default=False)
    analyze_parser.add_argument("--poll", action="store_true", help="Poll until task completion or failure")
    analyze_parser.add_argument("--poll-interval", type=float, default=2.0)
    analyze_parser.set_defaults(func=analyze)

    status_parser = subparsers.add_parser("status", help="Fetch analysis task status")
    status_parser.add_argument("task_id")
    status_parser.set_defaults(func=status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
