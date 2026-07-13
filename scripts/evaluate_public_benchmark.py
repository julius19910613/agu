#!/usr/bin/env python3
"""Evaluate AGU's public contract fixture without private data or models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "examples/benchmark"


def _load(name: str) -> dict:
    return json.loads((FIXTURE / name).read_text(encoding="utf-8"))


def evaluate() -> dict[str, float | str]:
    from app.analysis.evaluation import (
        evaluate_events,
        load_analysis_json,
        load_ground_truth_events,
        prediction_events_from_analysis,
    )

    truth = _load("ground_truth.json")
    analysis = load_analysis_json(FIXTURE / "analysis_prediction.json")
    scoreboard = (analysis.get("long_video") or {}).get("scoreboard_summary") or {}
    expected_scoreboard = truth["scoreboard"]
    scoreboard_accuracy = float(
        scoreboard.get("final_left_score") == expected_scoreboard["left_score"]
        and scoreboard.get("final_right_score") == expected_scoreboard["right_score"]
    )

    identity_prediction = _load("identity_prediction.json")
    predicted_pairs = {
        (item["left"], item["right"]): bool(item["same_player"])
        for item in identity_prediction["identity_pairs"]
    }
    expected_pairs = truth["identity_pairs"]
    correct_pairs = sum(
        predicted_pairs.get((item["left"], item["right"])) == bool(item["same_player"])
        for item in expected_pairs
    )
    identity_pair_accuracy = correct_pairs / len(expected_pairs) if expected_pairs else 1.0

    events = evaluate_events(
        load_ground_truth_events(FIXTURE / "ground_truth_events.csv", fps=30.0),
        prediction_events_from_analysis(analysis),
        tolerance_frames=5,
        require_player=True,
    )["metrics"]
    return {
        "benchmark_version": "1",
        "scoreboard_accuracy": scoreboard_accuracy,
        "identity_pair_accuracy": identity_pair_accuracy,
        "event_precision": events["precision"],
        "event_recall": events["recall"],
        "event_f1": events["f1"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Compare against checked-in golden metrics")
    args = parser.parse_args(argv)
    metrics = evaluate()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.strict and metrics != _load("golden_metrics.json"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
