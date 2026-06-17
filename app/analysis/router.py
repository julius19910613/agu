from __future__ import annotations

import os
import logging
import traceback

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from app.config import Settings
from app.analysis.schemas import (
    AnalysisRequest, 
    AnalysisResponse, 
    AnalysisRunAsyncResponse, 
    AnalysisTaskStatusResponse
)
from app.analysis.service import AnalysisService
from app.analysis.task_manager import TaskManager
from app.dependencies import get_settings, get_analysis_service, get_task_manager_dep

router = APIRouter(prefix="/analysis", tags=["analysis"])


def bg_run_analysis(
    task_id: str,
    request: AnalysisRequest,
    service: AnalysisService,
    task_manager: TaskManager
) -> None:
    """Run analysis in a background thread and update TaskManager states."""
    try:
        task_manager.update_status(task_id, status="processing", progress=10)
        result = service.run_analysis(request)
        task_manager.set_result(task_id, result)
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
    real_cwd = os.path.realpath(os.getcwd())
    real_video_path = os.path.realpath(video_path)
    try:
        if os.path.commonpath([real_cwd, real_video_path]) != real_cwd:
            raise HTTPException(status_code=400, detail="Access denied: Invalid video path.")
    except ValueError:
        raise HTTPException(status_code=400, detail="Access denied: Invalid video path.")

    if not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail=f"Video file not found: {video_path}")
    
    if request.boxes_file and not os.path.exists(request.boxes_file):
        raise HTTPException(status_code=400, detail=f"Boxes file not found: {request.boxes_file}")

    # Create background task and dispatch
    task_id = task_manager.create_task()
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
        status=state.status,
        progress=state.progress,
        error=state.error,
        result=state.result
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
