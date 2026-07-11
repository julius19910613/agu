from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ACTION_ALIASES = {
    "shot": "shoot",
    "shot_attempt": "shoot",
    "shoot": "shoot",
    "made_shot": "shoot",
    "missed_shot": "shoot",
    "pass": "pass",
    "rebound": "rebound_candidate",
    "rebound_candidate": "rebound_candidate",
    "steal": "steal_candidate",
    "steal_candidate": "steal_candidate",
    "block": "block_candidate",
    "block_candidate": "block_candidate",
}


@dataclass(frozen=True)
class GroundTruthEvent:
    event_id: str
    event_type: str
    center_frame: int
    player_id: Optional[str] = None


@dataclass(frozen=True)
class PredictionEvent:
    event_type: str
    center_frame: int
    confidence: float
    player_id: Optional[str] = None
    source: str = "record"


def load_analysis_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_ground_truth_events(path: str | Path, fps: float = 30.0) -> List[GroundTruthEvent]:
    """Load a minimal event CSV for repeatable CLI evaluation.

    Supported columns are intentionally permissive so exported spreadsheets can
    be used without a one-off converter: event_id/id, event_type/action/type,
    center_frame/frame/start_frame/end_frame/time_sec/start_sec/end_sec, and
    player_id/global_player_id/manual_player_id.
    """
    csv_path = Path(path)
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
    last_error: Optional[Exception] = None
    rows: List[Dict[str, str]] = []
    for encoding in encodings:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as fp:
                rows = list(csv.DictReader(fp))
            last_error = None
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error

    events: List[GroundTruthEvent] = []
    for index, row in enumerate(rows, start=1):
        event_type = _normalize_action(_first_value(row, "event_type", "action", "type", "事件类型", "动作"))
        if not event_type:
            continue
        center_frame = _event_center_frame(row, fps=fps)
        player_id = _first_value(row, "global_player_id", "player_id", "manual_player_id", "球员ID", "球员")
        events.append(
            GroundTruthEvent(
                event_id=_first_value(row, "event_id", "id", "事件ID") or f"event_{index:04d}",
                event_type=event_type,
                center_frame=center_frame,
                player_id=player_id or None,
            )
        )
    return events


def prediction_events_from_analysis(
    analysis: Dict[str, Any],
    *,
    min_confidence: float = 0.0,
    include_event_candidates: bool = True,
) -> List[PredictionEvent]:
    predictions: List[PredictionEvent] = []
    for record in analysis.get("records") or []:
        final = record.get("final") or {}
        action = _normalize_action(str(final.get("action") or ""))
        confidence = float(final.get("confidence") or 0.0)
        if not action or confidence < min_confidence:
            continue
        predictions.append(
            PredictionEvent(
                event_type=action,
                center_frame=(int(record.get("start_frame") or 0) + int(record.get("end_frame") or 0)) // 2,
                confidence=confidence,
                player_id=record.get("global_player_id") or record.get("local_player_id"),
                source=str(final.get("source") or "record"),
            )
        )

    if include_event_candidates:
        long_video = analysis.get("long_video") or {}
        for event in long_video.get("event_candidates") or []:
            event_type = _normalize_action(str(event.get("event_type") or ""))
            confidence = float(event.get("confidence") or 0.0)
            if not event_type or confidence < min_confidence:
                continue
            predictions.append(
                PredictionEvent(
                    event_type=event_type,
                    center_frame=(int(event.get("start_frame") or 0) + int(event.get("end_frame") or 0)) // 2,
                    confidence=confidence,
                    player_id=event.get("player_id"),
                    source=str(event.get("method") or "event_candidate"),
                )
            )
    return predictions


def evaluate_events(
    ground_truth: Iterable[GroundTruthEvent],
    predictions: Iterable[PredictionEvent],
    *,
    tolerance_frames: int,
    require_player: bool = False,
) -> Dict[str, Any]:
    gt_events = list(ground_truth)
    pred_events = sorted(list(predictions), key=lambda item: item.confidence, reverse=True)
    used_predictions: set[int] = set()
    matches: List[Dict[str, Any]] = []

    for gt in gt_events:
        best_index: Optional[int] = None
        best_distance: Optional[int] = None
        for index, pred in enumerate(pred_events):
            if index in used_predictions or pred.event_type != gt.event_type:
                continue
            distance = abs(int(pred.center_frame) - int(gt.center_frame))
            if distance > tolerance_frames:
                continue
            if require_player and gt.player_id and pred.player_id != gt.player_id:
                continue
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is None:
            matches.append({"event_id": gt.event_id, "status": "fn", "gt": _gt_dict(gt)})
            continue
        used_predictions.add(best_index)
        pred = pred_events[best_index]
        player_match = (not gt.player_id) or pred.player_id == gt.player_id
        matches.append(
            {
                "event_id": gt.event_id,
                "status": "tp" if player_match or not require_player else "fp_player",
                "gt": _gt_dict(gt),
                "prediction": _prediction_dict(pred),
                "frame_error": best_distance,
                "player_match": player_match,
            }
        )

    tp = sum(1 for match in matches if match["status"] == "tp")
    fn = sum(1 for match in matches if match["status"] == "fn")
    unmatched_predictions = len(pred_events) - len(used_predictions)
    player_mismatches = sum(1 for match in matches if match["status"] == "fp_player")
    fp = unmatched_predictions + player_mismatches

    precision, recall, f1 = _prf(tp=tp, fp=fp, fn=fn)
    return {
        "metrics": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "require_player": require_player,
            "tolerance_frames": tolerance_frames,
        },
        "matches": matches,
        "unmatched_prediction_count": unmatched_predictions,
    }


def render_evaluation_markdown(result: Dict[str, Any]) -> str:
    metrics = result.get("metrics") or {}
    lines = [
        "# AGU Evaluation Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Precision | {float(metrics.get('precision') or 0.0):.4f} |",
        f"| Recall | {float(metrics.get('recall') or 0.0):.4f} |",
        f"| F1 | {float(metrics.get('f1') or 0.0):.4f} |",
        f"| TP | {int(metrics.get('true_positive') or 0)} |",
        f"| FP | {int(metrics.get('false_positive') or 0)} |",
        f"| FN | {int(metrics.get('false_negative') or 0)} |",
        "",
        "## Matches",
        "",
        "| Event | Status | GT Type | GT Player | Prediction Type | Prediction Player | Frame Error |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for match in result.get("matches") or []:
        gt = match.get("gt") or {}
        pred = match.get("prediction") or {}
        lines.append(
            f"| {match.get('event_id')} | {match.get('status')} | {gt.get('event_type')} | "
            f"{gt.get('player_id') or ''} | {pred.get('event_type') or ''} | "
            f"{pred.get('player_id') or ''} | {match.get('frame_error') if match.get('frame_error') is not None else ''} |"
        )
    return "\n".join(lines) + "\n"


def _normalize_action(value: str) -> str:
    key = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return ACTION_ALIASES.get(key, key)


def _first_value(row: Dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _event_center_frame(row: Dict[str, str], fps: float) -> int:
    direct = _first_value(row, "center_frame", "frame", "帧")
    if direct:
        return int(round(float(direct)))
    start_frame = _first_value(row, "start_frame", "开始帧")
    end_frame = _first_value(row, "end_frame", "结束帧")
    if start_frame and end_frame:
        return (int(round(float(start_frame))) + int(round(float(end_frame)))) // 2
    time_sec = _first_value(row, "time_sec", "center_sec", "时间", "时间秒")
    if time_sec:
        return int(round(float(time_sec) * fps))
    start_sec = _first_value(row, "start_sec", "开始秒")
    end_sec = _first_value(row, "end_sec", "结束秒")
    if start_sec and end_sec:
        return int(round(((float(start_sec) + float(end_sec)) / 2.0) * fps))
    return 0


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _gt_dict(event: GroundTruthEvent) -> Dict[str, Any]:
    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "center_frame": event.center_frame,
        "player_id": event.player_id,
    }


def _prediction_dict(event: PredictionEvent) -> Dict[str, Any]:
    return {
        "event_type": event.event_type,
        "center_frame": event.center_frame,
        "confidence": event.confidence,
        "player_id": event.player_id,
        "source": event.source,
    }
