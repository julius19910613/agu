from __future__ import annotations

import logging
import os
import time
import traceback
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.analysis.schemas import AnalysisRequest, AnalysisRunAsyncResponse, AnalysisTaskStatusResponse
from app.analysis.service import AnalysisService
from app.analysis.task_manager import TaskManager
from app.config import Settings
from app.dependencies import get_analysis_service, get_settings, get_task_manager_dep

router = APIRouter(prefix="/analysis", tags=["analysis"])


class AnalysisTaskCancelled(RuntimeError):
    """Raised at a cooperative stage boundary after cancellation."""


class AnalysisDeadlineExceeded(RuntimeError):
    """Raised at a cooperative stage boundary after the runtime budget."""


def _video_path_is_allowed(video_path: str, settings: Settings) -> bool:
    configured_roots = [
        value.strip()
        for value in str(settings.allowed_video_roots or "").replace("\n", ",").split(",")
        if value.strip()
    ]
    roots = [Path.cwd(), *(Path(value).expanduser() for value in configured_roots)]
    real_video_path = Path(video_path).expanduser().resolve()
    for root in roots:
        real_root = root.resolve()
        try:
            if os.path.commonpath([str(real_root), str(real_video_path)]) == str(real_root):
                return True
        except ValueError:
            continue
    return False


def bg_run_analysis(
    task_id: str,
    request: AnalysisRequest,
    service: AnalysisService,
    task_manager: TaskManager
) -> None:
    """Run analysis in a background thread and update TaskManager states."""
    try:
        started_at = time.monotonic()
        task_manager.update_status(task_id, status="processing", progress=10)
        configured_timeout = float(
            getattr(getattr(service, "settings", None), "analysis_timeout_sec", 0.0) or 0.0
        )
        timeout_sec = request.max_runtime_sec if request.max_runtime_sec is not None else configured_timeout

        def update_progress(progress: int) -> None:
            if task_manager.should_cancel(task_id):
                raise AnalysisTaskCancelled("Analysis cancelled by request.")
            if timeout_sec > 0 and time.monotonic() - started_at > timeout_sec:
                raise AnalysisDeadlineExceeded(f"Analysis exceeded max_runtime_sec={timeout_sec:g}.")
            task_manager.update_status(task_id, status="processing", progress=progress)

        update_progress(10)
        result = service.run_analysis(request, progress_callback=update_progress)
        update_progress(99)
        task_manager.set_result(task_id, result)
    except AnalysisTaskCancelled:
        task_manager.update_status(
            task_id,
            status="cancelled",
            progress=100,
            error="Analysis cancelled by request.",
        )
    except AnalysisDeadlineExceeded as exc:
        task_manager.update_status(task_id, status="failed", progress=100, error=str(exc))
    except Exception as e:
        err_msg = "".join(traceback.format_exception(None, e, e.__traceback__))
        logging.getLogger("app.analysis.router").error(
            "Background analysis task %s failed: %s\n%s", 
            task_id, str(e), err_msg
        )
        task_manager.update_status(task_id, status="failed", progress=100, error=str(e))


@router.post("/run", response_model=AnalysisRunAsyncResponse)
def run_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    service: AnalysisService = Depends(get_analysis_service),
    settings: Settings = Depends(get_settings),
    task_manager: TaskManager = Depends(get_task_manager_dep),
) -> AnalysisRunAsyncResponse:
    """Start a hybrid video analysis asynchronously in the background."""
    video_path = request.video_path
    if not _video_path_is_allowed(video_path, settings):
        raise HTTPException(status_code=400, detail="Access denied: Invalid video path.")

    if not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail=f"Video file not found: {video_path}")
    
    if request.boxes_file and not os.path.exists(request.boxes_file):
        raise HTTPException(status_code=400, detail=f"Boxes file not found: {request.boxes_file}")

    # Create background task and dispatch
    task_id = task_manager.create_task(request=request.model_copy(deep=True))
    background_tasks.add_task(bg_run_analysis, task_id, request, service, task_manager)

    return AnalysisRunAsyncResponse(
        task_id=task_id,
        status="pending",
        message="Analysis started asynchronously. Please poll the status endpoint to query progress."
    )


@router.get("/status/{task_id}", response_model=AnalysisTaskStatusResponse)
def get_task_status(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager_dep)
) -> AnalysisTaskStatusResponse:
    """Retrieve the progress and results of a background analysis task."""
    state = task_manager.get_task(task_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found.")
    
    return AnalysisTaskStatusResponse(
        task_id=state.task_id,
        request_id=state.request_id,
        status=state.status,
        progress=state.progress,
        error=state.error,
        result=state.result,
        created_at_unix=state.created_at_unix,
        updated_at_unix=state.updated_at_unix,
    )


@router.post("/tasks", response_model=AnalysisRunAsyncResponse)
def create_task_alias(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    service: AnalysisService = Depends(get_analysis_service),
    settings: Settings = Depends(get_settings),
    task_manager: TaskManager = Depends(get_task_manager_dep),
) -> AnalysisRunAsyncResponse:
    """Backward-compatible alias for external BFF analysis task submission."""
    return run_analysis(request, background_tasks, service, settings, task_manager)


@router.get("/tasks/{task_id}", response_model=AnalysisTaskStatusResponse)
def get_task_status_alias(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager_dep)
) -> AnalysisTaskStatusResponse:
    """Backward-compatible alias for /analysis/tasks/{task_id}."""
    return get_task_status(task_id, task_manager)


@router.get("/tasks/{task_id}/result", response_model=AnalysisTaskStatusResponse)
def get_task_result_alias(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager_dep)
) -> AnalysisTaskStatusResponse:
    """Backward-compatible alias for /analysis/tasks/{task_id}/result."""
    return get_task_status(task_id, task_manager)


@router.post("/tasks/{task_id}/cancel", response_model=AnalysisTaskStatusResponse)
def cancel_task(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager_dep),
) -> AnalysisTaskStatusResponse:
    state = task_manager.request_cancel(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found.")
    return get_task_status(task_id, task_manager)


@router.post("/tasks/{task_id}/retry", response_model=AnalysisRunAsyncResponse)
def retry_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    service: AnalysisService = Depends(get_analysis_service),
    task_manager: TaskManager = Depends(get_task_manager_dep),
) -> AnalysisRunAsyncResponse:
    previous = task_manager.get_task(task_id)
    if previous is None:
        raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found.")
    if previous.status not in {"failed", "cancelled"}:
        raise HTTPException(status_code=409, detail="Only failed or cancelled tasks can be retried.")
    if previous.request is None:
        raise HTTPException(status_code=409, detail="Original task request is unavailable.")
    request = previous.request.model_copy(deep=True)
    new_task_id = task_manager.create_task(request=request)
    background_tasks.add_task(bg_run_analysis, new_task_id, request, service, task_manager)
    return AnalysisRunAsyncResponse(
        task_id=new_task_id,
        status="pending",
        message=f"Retry created from task {task_id}.",
    )
