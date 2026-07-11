from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


TERMINAL_STATES = {"completed", "failed"}

PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "vlm_mode": "off",
        "generate_video": False,
        "segmented_analysis": True,
        "vlm_audit": False,
        "tracking_fps": 6.0,
        "yolo_imgsz": 320,
        "action_vid_stride": 32,
        "max_players_per_segment": 10,
    },
    "accurate": {
        "vlm_mode": "low-confidence",
        "generate_video": False,
        "segmented_analysis": True,
        "segment_duration_sec": 30.0,
        "segment_overlap_sec": 3.0,
        "vlm_audit": True,
        "vlm_audit_frames": 4,
        "scoreboard_audit": True,
        "scoreboard_audit_max_frames": 4,
        "tracking_fps": 12.0,
        "yolo_imgsz": 640,
        "action_vid_stride": 12,
        "max_players_per_segment": 18,
        "tracker_backend": "botsort",
        "yolo_reid_enabled": True,
    },
    "vlm-full": {
        "vlm_mode": "always",
        "generate_video": False,
        "segmented_analysis": True,
        "segment_duration_sec": 30.0,
        "segment_overlap_sec": 3.0,
        "vlm_audit": True,
        "vlm_audit_frames": 4,
        "scoreboard_audit": True,
        "scoreboard_audit_max_frames": 4,
        "tracking_fps": 12.0,
        "yolo_imgsz": 640,
        "action_vid_stride": 12,
        "max_players_per_segment": 18,
        "tracker_backend": "botsort",
        "yolo_reid_enabled": True,
        "vlm_identity_merge_enabled": True,
    },
}


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


def _print_status_summary(payload: dict[str, Any]) -> None:
    result = payload.get("result") or {}
    summary = result.get("summary") or {}
    long_video = result.get("long_video") or {}
    compact = {
        "task_id": payload.get("task_id"),
        "status": payload.get("status"),
        "progress": payload.get("progress"),
        "error": payload.get("error"),
        "runtime_seconds": result.get("runtime_seconds"),
        "clip_count": summary.get("clip_count"),
        "source_counts": summary.get("source_counts"),
        "segment_count": len(long_video.get("segments") or []),
        "player_count": len(long_video.get("players") or []),
        "audit_summary": long_video.get("audit_summary"),
        "scoreboard_summary": long_video.get("scoreboard_summary"),
    }
    _print_json({key: value for key, value in compact.items() if value is not None})


def _apply_preset(args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(PRESETS.get(args.preset or "", {}))
    explicit_values = {
        "video_path": args.video,
        "vlm_mode": args.vlm_mode,
        "boxes_file": args.boxes_file,
        "max_frames": args.max_frames,
        "generate_video": args.generate_video,
        "segmented_analysis": args.segmented_analysis,
        "segment_duration_sec": args.segment_duration_sec,
        "segment_overlap_sec": args.segment_overlap_sec,
        "segment_start_sec": args.segment_start_sec,
        "segment_end_sec": args.segment_end_sec,
        "max_segments": args.max_segments,
        "vlm_audit": args.vlm_audit,
        "vlm_audit_frames": args.vlm_audit_frames,
        "scoreboard_audit": args.scoreboard_audit,
        "scoreboard_audit_interval_sec": args.scoreboard_audit_interval_sec,
        "scoreboard_audit_max_frames": args.scoreboard_audit_max_frames,
        "vlm_identity_merge_enabled": args.vlm_identity_merge_enabled,
        "tracker_backend": args.tracker_backend,
        "tracker_conf_thres": args.tracker_conf_thres,
        "tracker_iou_thres": args.tracker_iou_thres,
        "tracker_min_appear_ratio": args.tracker_min_appear_ratio,
        "tracker_min_appear_abs": args.tracker_min_appear_abs,
        "tracking_fps": args.tracking_fps,
        "yolo_imgsz": args.yolo_imgsz,
        "max_players_per_segment": args.max_players_per_segment,
        "yolo_device": args.yolo_device,
        "yolo_reid_enabled": args.yolo_reid_enabled,
        "identity_embedding_backend": args.identity_embedding_backend,
        "jersey_number_vlm_enabled": args.jersey_number_vlm_enabled,
        "jersey_number_vlm_frames": args.jersey_number_vlm_frames,
        "vid_stride": args.vid_stride,
        "action_vid_stride": args.action_vid_stride,
        "low_confidence": args.low_confidence,
        "high_confidence": args.high_confidence,
    }
    payload.update({key: value for key, value in explicit_values.items() if value is not None})
    return payload


def analyze(args: argparse.Namespace) -> int:
    payload = _apply_preset(args)
    response = _request_json("POST", f"{args.api_url.rstrip('/')}/api/v1/analysis/run", payload)
    _print_status_summary(response) if args.summary else _print_json(response)

    if not args.poll:
        return 0

    task_id = response.get("task_id")
    if not task_id:
        raise SystemExit("Response did not contain task_id")

    while True:
        time.sleep(args.poll_interval)
        status = _request_json("GET", f"{args.api_url.rstrip('/')}/api/v1/analysis/status/{task_id}")
        _print_status_summary(status) if args.summary else _print_json(status)
        if status.get("status") in TERMINAL_STATES:
            if args.save_result and status.get("result"):
                Path(args.save_result).write_text(
                    json.dumps(status["result"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            return 0 if status.get("status") == "completed" else 1


def status(args: argparse.Namespace) -> int:
    response = _request_json("GET", f"{args.api_url.rstrip('/')}/api/v1/analysis/status/{args.task_id}")
    _print_json(response) if args.json else _print_status_summary(response)
    return 0 if response.get("status") != "failed" else 1


def report(args: argparse.Namespace) -> int:
    from scripts.build_player_markdown_reports import build_player_markdown_reports

    analysis = json.loads(Path(args.analysis_json).read_text(encoding="utf-8"))
    summary = build_player_markdown_reports(
        analysis=analysis,
        video_path=args.video,
        output_dir=args.output_dir,
        max_players=args.max_players,
        min_roster_score=args.min_roster_score,
        crops_per_player=args.crops_per_player,
        video_fps=args.video_fps,
        vlm_player_filter=args.vlm_player_filter,
        vlm_model=args.vlm_model,
        vlm_timeout_sec=args.vlm_timeout_sec,
        vlm_concurrency=args.vlm_concurrency,
        vlm_cache_path=args.vlm_cache_path,
        vlm_progress=args.vlm_progress,
        require_vlm_player=args.require_vlm_player,
        dedupe_players=args.dedupe_players,
    )
    _print_json(summary)
    return 0


def evaluate(args: argparse.Namespace) -> int:
    from app.analysis.evaluation import (
        evaluate_events,
        load_analysis_json,
        load_ground_truth_events,
        prediction_events_from_analysis,
        render_evaluation_markdown,
    )

    analysis = load_analysis_json(args.analysis_json)
    fps = float(args.fps or ((analysis.get("long_video") or {}).get("fps") or 30.0))
    ground_truth = load_ground_truth_events(args.events_csv, fps=fps)
    predictions = prediction_events_from_analysis(
        analysis,
        min_confidence=args.min_confidence,
        include_event_candidates=not args.no_event_candidates,
    )
    result = evaluate_events(
        ground_truth,
        predictions,
        tolerance_frames=int(round(args.tolerance_sec * fps)),
        require_player=args.require_player,
    )
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).write_text(render_evaluation_markdown(result), encoding="utf-8")
    _print_json(result if args.json else {"metrics": result["metrics"]})
    return 0 if (not args.fail_below_f1 or result["metrics"]["f1"] >= args.fail_below_f1) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="AGU command line client for the running FastAPI analysis service.",
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8765", help="AGU service base URL")

    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Submit an analysis task")
    analyze_parser.add_argument("--video", required=True, help="Video path visible to the AGU service")
    analyze_parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    analyze_parser.add_argument("--vlm-mode", choices=["off", "low-confidence", "always"])
    analyze_parser.add_argument("--boxes-file")
    analyze_parser.add_argument("--max-frames", type=int)
    analyze_parser.add_argument("--generate-video", action=argparse.BooleanOptionalAction, default=None)
    analyze_parser.add_argument("--segmented-analysis", action=argparse.BooleanOptionalAction, default=None)
    analyze_parser.add_argument("--segment-duration-sec", type=float)
    analyze_parser.add_argument("--segment-overlap-sec", type=float)
    analyze_parser.add_argument("--segment-start-sec", type=float)
    analyze_parser.add_argument("--segment-end-sec", type=float)
    analyze_parser.add_argument("--max-segments", type=int)
    analyze_parser.add_argument("--vlm-audit", action=argparse.BooleanOptionalAction, default=None)
    analyze_parser.add_argument("--vlm-audit-frames", type=int)
    analyze_parser.add_argument("--scoreboard-audit", action=argparse.BooleanOptionalAction, default=None)
    analyze_parser.add_argument("--scoreboard-audit-interval-sec", type=float)
    analyze_parser.add_argument("--scoreboard-audit-max-frames", type=int)
    analyze_parser.add_argument("--vlm-identity-merge-enabled", action=argparse.BooleanOptionalAction, default=None)
    analyze_parser.add_argument("--tracker-backend", choices=["bytetrack", "botsort", "custom"])
    analyze_parser.add_argument("--tracker-conf-thres", type=float)
    analyze_parser.add_argument("--tracker-iou-thres", type=float)
    analyze_parser.add_argument("--tracker-min-appear-ratio", type=float)
    analyze_parser.add_argument("--tracker-min-appear-abs", type=int)
    analyze_parser.add_argument("--tracking-fps", type=float)
    analyze_parser.add_argument("--yolo-imgsz", type=int)
    analyze_parser.add_argument("--max-players-per-segment", type=int)
    analyze_parser.add_argument("--yolo-device")
    analyze_parser.add_argument("--yolo-reid-enabled", action=argparse.BooleanOptionalAction, default=None)
    analyze_parser.add_argument("--identity-embedding-backend")
    analyze_parser.add_argument("--jersey-number-vlm-enabled", action=argparse.BooleanOptionalAction, default=None)
    analyze_parser.add_argument("--jersey-number-vlm-frames", type=int)
    analyze_parser.add_argument("--vid-stride", type=int)
    analyze_parser.add_argument("--action-vid-stride", type=int)
    analyze_parser.add_argument("--low-confidence", type=float)
    analyze_parser.add_argument("--high-confidence", type=float)
    analyze_parser.add_argument("--poll", action="store_true", help="Poll until task completion or failure")
    analyze_parser.add_argument("--poll-interval", type=float, default=2.0)
    analyze_parser.add_argument("--save-result", help="Write completed result JSON to this path when used with --poll")
    analyze_parser.add_argument("--summary", action="store_true", help="Print compact task summaries instead of full JSON")
    analyze_parser.set_defaults(func=analyze)

    status_parser = subparsers.add_parser("status", help="Fetch analysis task status")
    status_parser.add_argument("task_id")
    status_parser.add_argument("--json", action="store_true", help="Print full task JSON")
    status_parser.set_defaults(func=status)

    report_parser = subparsers.add_parser("report", help="Build per-player Markdown reports from an analysis JSON")
    report_parser.add_argument("--analysis-json", required=True)
    report_parser.add_argument("--video", required=True)
    report_parser.add_argument("--output-dir", required=True)
    report_parser.add_argument("--max-players", type=int)
    report_parser.add_argument("--min-roster-score", type=float, default=0.0)
    report_parser.add_argument("--crops-per-player", type=int, default=8)
    report_parser.add_argument("--video-fps", type=float, default=2.0)
    report_parser.add_argument("--dedupe-players", action="store_true")
    report_parser.add_argument("--vlm-player-filter", action="store_true")
    report_parser.add_argument("--require-vlm-player", action="store_true")
    report_parser.add_argument("--vlm-model", default="qwen3-vl:4b")
    report_parser.add_argument("--vlm-timeout-sec", type=float, default=45.0)
    report_parser.add_argument("--vlm-concurrency", type=int, default=1)
    report_parser.add_argument("--vlm-cache-path")
    report_parser.add_argument("--vlm-progress", action="store_true")
    report_parser.set_defaults(func=report)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate AGU analysis JSON against event labels")
    evaluate_parser.add_argument("--analysis-json", required=True)
    evaluate_parser.add_argument("--events-csv", required=True)
    evaluate_parser.add_argument("--fps", type=float)
    evaluate_parser.add_argument("--tolerance-sec", type=float, default=8.0)
    evaluate_parser.add_argument("--min-confidence", type=float, default=0.0)
    evaluate_parser.add_argument("--require-player", action="store_true")
    evaluate_parser.add_argument("--no-event-candidates", action="store_true")
    evaluate_parser.add_argument("--output-json")
    evaluate_parser.add_argument("--output-md")
    evaluate_parser.add_argument("--fail-below-f1", type=float)
    evaluate_parser.add_argument("--json", action="store_true", help="Print full match details")
    evaluate_parser.set_defaults(func=evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
