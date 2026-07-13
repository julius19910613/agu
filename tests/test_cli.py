from pathlib import Path

from app import cli


def test_analyze_preset_builds_accurate_payload(monkeypatch, capsys) -> None:
    captured = {}

    def fake_request(method, url, payload=None):
        captured["method"] = method
        captured["url"] = url
        captured["payload"] = payload
        return {"task_id": "task-1", "status": "pending", "message": "ok"}

    monkeypatch.setattr(cli, "_request_json", fake_request)

    code = cli.main(["analyze", "--video", "examples/lebron_shoots.mp4", "--preset", "accurate", "--summary"])

    assert code == 0
    assert captured["method"] == "POST"
    assert captured["payload"]["video_path"] == "examples/lebron_shoots.mp4"
    assert captured["payload"]["vlm_mode"] == "low-confidence"
    assert captured["payload"]["tracker_backend"] == "botsort"
    assert captured["payload"]["yolo_reid_enabled"] is True
    assert captured["payload"]["segment_duration_sec"] == 30.0
    assert captured["payload"]["segment_overlap_sec"] == 3.0
    assert captured["payload"]["vlm_audit_frames"] == 4
    assert captured["payload"]["scoreboard_audit"] is True
    assert captured["payload"]["scoreboard_audit_max_frames"] == 6
    assert "task-1" in capsys.readouterr().out


def test_analyze_explicit_args_override_preset(monkeypatch) -> None:
    captured = {}

    def fake_request(method, url, payload=None):
        captured["payload"] = payload
        return {"task_id": "task-1", "status": "pending", "message": "ok"}

    monkeypatch.setattr(cli, "_request_json", fake_request)

    code = cli.main(
        [
            "analyze",
            "--video",
            "examples/lebron_shoots.mp4",
            "--preset",
            "vlm-full",
            "--vlm-mode",
            "off",
            "--no-vlm-audit",
            "--max-segments",
            "1",
        ]
    )

    assert code == 0
    assert captured["payload"]["vlm_mode"] == "off"
    assert captured["payload"]["vlm_audit"] is False
    assert captured["payload"]["max_segments"] == 1


def test_status_defaults_to_compact_summary(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "_request_json",
        lambda method, url, payload=None: {
            "task_id": "task-1",
            "status": "completed",
            "progress": 100,
            "result": {"summary": {"clip_count": 2}, "long_video": {"segments": []}},
        },
    )

    code = cli.main(["status", "task-1"])

    assert code == 0
    output = capsys.readouterr().out
    assert "clip_count" in output
    assert "records" not in output


def test_evaluate_cli_writes_outputs(tmp_path: Path, capsys) -> None:
    analysis_json = tmp_path / "analysis.json"
    analysis_json.write_text(
        '{"records":[{"start_frame":10,"end_frame":20,"global_player_id":"player_001",'
        '"final":{"action":"shoot","confidence":0.9,"source":"r2plus1d"}}]}',
        encoding="utf-8",
    )
    events_csv = tmp_path / "events.csv"
    events_csv.write_text(
        "event_id,event_type,center_frame,global_player_id\n"
        "e1,shot_attempt,15,player_001\n",
        encoding="utf-8",
    )
    output_json = tmp_path / "eval.json"
    output_md = tmp_path / "eval.md"

    code = cli.main(
        [
            "evaluate",
            "--analysis-json",
            str(analysis_json),
            "--events-csv",
            str(events_csv),
            "--require-player",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert code == 0
    assert output_json.exists()
    assert output_md.exists()
    assert "f1" in capsys.readouterr().out
