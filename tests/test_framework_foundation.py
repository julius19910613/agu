from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import __version__, cli
from app.analysis.pipeline import CallableStage, PipelineContext, PipelineRunner
from app.analysis.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisSummaryResponse,
    Size2D,
)
from app.plugins import PluginRegistry, PluginSpec


def test_distribution_version_and_cli_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_pipeline_runner_records_order_and_duration() -> None:
    context: PipelineContext[str, str] = PipelineContext(request="video")
    runner = PipelineRunner(
        [
            CallableStage("prepare", lambda ctx: ctx.metadata.update({"prepared": True})),
            CallableStage("analyze", lambda ctx: setattr(ctx, "result", "ok")),
        ]
    )
    completed = runner.run(context)
    assert completed.result == "ok"
    assert [item["stage"] for item in completed.trace] == ["prepare", "analyze"]
    assert all(item["status"] == "completed" for item in completed.trace)
    assert all(item["duration_ms"] >= 0 for item in completed.trace)


def test_pipeline_rejects_duplicate_stage_names() -> None:
    stage = CallableStage("duplicate", lambda ctx: None)
    with pytest.raises(ValueError, match="unique"):
        PipelineRunner([stage, stage])


def test_plugin_registry_reports_optional_dependency() -> None:
    registry = PluginRegistry()
    registry.register(
        PluginSpec(
            name="missing-example",
            kind="integration",
            capabilities=("example",),
            requires=("agu_module_that_does_not_exist",),
            source="test",
        )
    )
    diagnosis = registry.diagnostics()["plugins"][0]
    assert diagnosis["available"] is False
    assert "agu_module_that_does_not_exist" in diagnosis["reason"]


def test_plugins_cli_lists_builtin_capabilities(capsys) -> None:
    assert cli.main(["plugins", "list", "--kind", "stage"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plugins"][0]["name"] == "analysis-pipeline-v1"
    assert "segmented-video" in payload["plugins"][0]["capabilities"]


def test_profile_values_and_explicit_flags(monkeypatch, tmp_path: Path) -> None:
    profile = tmp_path / "profile.toml"
    profile.write_text('[analysis]\nvlm_mode = "always"\nmax_frames = 12\n', encoding="utf-8")
    captured = {}

    def fake_request(method, url, payload=None):
        captured["payload"] = payload
        return {"task_id": "task-profile", "status": "pending", "message": "ok"}

    monkeypatch.setattr(cli, "_request_json", fake_request)
    assert cli.main(
        [
            "analyze",
            "--video",
            "example.mp4",
            "--profile",
            str(profile),
            "--vlm-mode",
            "off",
        ]
    ) == 0
    assert captured["payload"]["vlm_mode"] == "off"
    assert captured["payload"]["max_frames"] == 12


def test_analysis_response_additive_contract_defaults() -> None:
    response = AnalysisResponse(
        video="fixture.mp4",
        created_at_unix=0.0,
        runtime_seconds=0.0,
        frame_size=Size2D(width=1, height=1),
        seq_length=16,
        vid_stride=8,
        vlm_mode="off",
        ollama_model=None,
        records=[],
        summary=AnalysisSummaryResponse(
            clip_count=0,
            action_counts={},
            needs_review_count=0,
            source_counts={},
        ),
    )
    assert response.schema_version == "1.0"
    assert response.pipeline_manifest == {}


def test_analysis_service_dispatches_through_pipeline(monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from app.analysis.service import AnalysisService

    service = AnalysisService.__new__(AnalysisService)
    service.settings = SimpleNamespace(
        tracker_backend="bytetrack",
        identity_embedding_backend="sidecar_hsv_hist",
        face_identity_backend="disabled",
        scoreboard_ocr_backend="disabled",
    )
    expected = AnalysisResponse(
        video="fixture.mp4",
        created_at_unix=0.0,
        runtime_seconds=0.0,
        frame_size=Size2D(width=1, height=1),
        seq_length=16,
        vid_stride=8,
        vlm_mode="off",
        ollama_model=None,
        records=[],
        summary=AnalysisSummaryResponse(clip_count=0, action_counts={}, needs_review_count=0, source_counts={}),
    )
    monkeypatch.setattr(service, "_run_single_analysis", lambda request, progress_callback=None: expected)
    result = service.run_analysis(AnalysisRequest(video_path="fixture.mp4", segmented_analysis=False))
    assert result is expected
    assert [item["stage"] for item in result.pipeline_manifest["stages"]] == [
        "analysis.validate",
        "analysis.dispatch",
        "analysis.finalize",
    ]
    assert all(item["status"] == "completed" for item in result.pipeline_manifest["stages"])


def test_registered_extension_stage_runs_in_real_service_pipeline(monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from app.analysis.pipeline import CallableStage, clear_analysis_stages, register_analysis_stage
    from app.analysis.service import AnalysisService

    service = AnalysisService.__new__(AnalysisService)
    service.settings = SimpleNamespace(
        tracker_backend="bytetrack",
        identity_embedding_backend="sidecar_hsv_hist",
        face_identity_backend="disabled",
        scoreboard_ocr_backend="disabled",
    )
    expected = AnalysisResponse(
        video="fixture.mp4",
        created_at_unix=0.0,
        runtime_seconds=0.0,
        frame_size=Size2D(width=1, height=1),
        seq_length=16,
        vid_stride=8,
        vlm_mode="off",
        ollama_model=None,
        records=[],
        summary=AnalysisSummaryResponse(clip_count=0, action_counts={}, needs_review_count=0, source_counts={}),
    )
    monkeypatch.setattr(service, "_run_single_analysis", lambda request, progress_callback=None: expected)
    clear_analysis_stages()
    try:
        register_analysis_stage(
            CallableStage("test.annotate", lambda context: context.metadata.update({"extension": "ran"})),
            position="after_dispatch",
        )
        result = service.run_analysis(AnalysisRequest(video_path="fixture.mp4", segmented_analysis=False))
        assert result.pipeline_manifest["metadata"]["extension"] == "ran"
        assert "test.annotate" in [item["stage"] for item in result.pipeline_manifest["stages"]]
    finally:
        clear_analysis_stages()


def test_public_benchmark_matches_golden() -> None:
    from scripts.evaluate_public_benchmark import evaluate

    root = Path(__file__).resolve().parents[1]
    golden = json.loads((root / "examples/benchmark/golden_metrics.json").read_text(encoding="utf-8"))
    assert evaluate() == golden


def test_cancelled_task_cannot_be_overwritten_by_late_result() -> None:
    from app.analysis.task_manager import TaskManager

    manager = TaskManager()
    task_id = manager.create_task(request=AnalysisRequest(video_path="fixture.mp4"))
    state = manager.request_cancel(task_id)
    assert state is not None
    assert state.status == "cancelled"
    manager.set_result(task_id, AnalysisResponse(
        video="fixture.mp4",
        created_at_unix=0.0,
        runtime_seconds=0.0,
        frame_size=Size2D(width=1, height=1),
        seq_length=16,
        vid_stride=8,
        vlm_mode="off",
        ollama_model=None,
        records=[],
        summary=AnalysisSummaryResponse(clip_count=0, action_counts={}, needs_review_count=0, source_counts={}),
    ))
    assert manager.get_task(task_id).status == "cancelled"
    assert manager.get_task(task_id).result is None


def test_cli_cancel_uses_compatible_task_endpoint(monkeypatch, capsys) -> None:
    captured = {}

    def fake_request(method, url, payload=None):
        captured.update({"method": method, "url": url})
        return {"task_id": "task-1", "status": "cancelled", "progress": 100}

    monkeypatch.setattr(cli, "_request_json", fake_request)
    assert cli.main(["cancel", "task-1"]) == 0
    assert captured == {
        "method": "POST",
        "url": "http://127.0.0.1:8765/api/v1/analysis/tasks/task-1/cancel",
    }
    assert "cancelled" in capsys.readouterr().out


def test_retry_creates_new_task_from_cancelled_request() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from fastapi import BackgroundTasks

    from app.analysis.router import retry_task
    from app.analysis.task_manager import TaskManager

    manager = TaskManager()
    request = AnalysisRequest(video_path="fixture.mp4")
    old_task_id = manager.create_task(request=request)
    manager.request_cancel(old_task_id)
    response = retry_task(old_task_id, BackgroundTasks(), object(), manager)
    assert response.task_id != old_task_id
    new_state = manager.get_task(response.task_id)
    assert new_state is not None
    assert new_state.status == "pending"
    assert new_state.request.video_path == "fixture.mp4"


def test_runtime_budget_marks_task_failed_at_progress_boundary(monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from app.analysis import router
    from app.analysis.task_manager import TaskManager

    class Service:
        settings = SimpleNamespace(analysis_timeout_sec=0.0)

        def run_analysis(self, request, progress_callback=None):
            raise AssertionError("deadline should be checked before dispatch")

    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(router.time, "monotonic", lambda: next(ticks))
    manager = TaskManager()
    request = AnalysisRequest(video_path="fixture.mp4", max_runtime_sec=1.0)
    task_id = manager.create_task(request=request)
    router.bg_run_analysis(task_id, request, Service(), manager)
    state = manager.get_task(task_id)
    assert state.status == "failed"
    assert "max_runtime_sec=1" in state.error
