import csv
from pathlib import Path

from app.analysis.evaluation import (
    evaluate_events,
    load_ground_truth_events,
    prediction_events_from_analysis,
)


def test_load_ground_truth_events_accepts_utf8_sig_csv(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["event_id", "event_type", "time_sec", "global_player_id"])
        writer.writeheader()
        writer.writerow(
            {
                "event_id": "e1",
                "event_type": "shot_attempt",
                "time_sec": "2.0",
                "global_player_id": "player_001",
            }
        )

    events = load_ground_truth_events(path, fps=30.0)

    assert len(events) == 1
    assert events[0].event_type == "shoot"
    assert events[0].center_frame == 60
    assert events[0].player_id == "player_001"


def test_evaluate_events_can_score_strict_player_matches(tmp_path: Path) -> None:
    analysis = {
        "records": [
            {
                "start_frame": 50,
                "end_frame": 70,
                "global_player_id": "player_001",
                "final": {"action": "shoot", "confidence": 0.9, "source": "r2plus1d"},
            },
            {
                "start_frame": 80,
                "end_frame": 95,
                "global_player_id": "player_002",
                "final": {"action": "pass", "confidence": 0.7, "source": "r2plus1d"},
            },
        ]
    }
    predictions = prediction_events_from_analysis(analysis)
    ground_truth = load_ground_truth_events(_write_events_csv(tmp_path / "events.csv"), fps=30.0)
    result = evaluate_events(ground_truth, predictions, tolerance_frames=12, require_player=True)

    assert result["metrics"]["true_positive"] == 1
    assert result["metrics"]["false_positive"] == 1
    assert result["metrics"]["false_negative"] == 0
    assert result["metrics"]["f1"] > 0.65


def test_prediction_events_include_long_video_event_candidates() -> None:
    analysis = {
        "records": [],
        "long_video": {
            "event_candidates": [
                {
                    "event_type": "rebound_candidate",
                    "start_frame": 100,
                    "end_frame": 120,
                    "confidence": 0.4,
                    "player_id": "player_003",
                    "method": "candidate",
                }
            ]
        },
    }

    predictions = prediction_events_from_analysis(analysis)

    assert len(predictions) == 1
    assert predictions[0].event_type == "rebound_candidate"
    assert predictions[0].player_id == "player_003"


def _write_events_csv(path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["event_id", "event_type", "center_frame", "global_player_id"])
        writer.writeheader()
        writer.writerow(
            {
                "event_id": "e1",
                "event_type": "shoot",
                "center_frame": "60",
                "global_player_id": "player_001",
            }
        )
    return path
