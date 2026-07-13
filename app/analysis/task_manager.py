from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional
from uuid import uuid4

from app.analysis.schemas import AnalysisResponse


class TaskState:
    """Represents the in-memory state of a background analysis task."""

    def __init__(self, task_id: str, request: Any = None):
        self.task_id = task_id
        self.request_id = task_id
        self.status = "pending"  # pending | processing | completed | failed | cancelled
        self.progress = 0
        self.result: Optional[AnalysisResponse] = None
        self.error: Optional[str] = None
        self.request = request
        self.cancel_requested = False
        self.created_at_unix = time.time()
        self.updated_at_unix = self.created_at_unix


class TaskManager:
    """Thread-safe manager for background analysis task states."""

    def __init__(self):
        self._tasks: Dict[str, TaskState] = {}
        self._lock = threading.Lock()

    def create_task(self, request: Any = None) -> str:
        """Create a new task in pending state and return its task_id."""
        task_id = str(uuid4().hex)
        state = TaskState(task_id, request=request)
        with self._lock:
            self._tasks[task_id] = state
        return task_id

    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Retrieve the state of a task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def update_status(self, task_id: str, status: str, progress: int = None, error: str = None) -> None:
        """Update the status, progress, or error of a task."""
        with self._lock:
            state = self._tasks.get(task_id)
            if state:
                if state.status == "cancelled" and status != "cancelled":
                    return
                state.status = status
                if progress is not None:
                    state.progress = progress
                if error is not None:
                    state.error = error
                state.updated_at_unix = time.time()

    def set_result(self, task_id: str, result: AnalysisResponse) -> None:
        """Complete a task by storing its final result payload."""
        with self._lock:
            state = self._tasks.get(task_id)
            if state and state.status != "cancelled":
                state.status = "completed"
                state.progress = 100
                state.result = result
                state.updated_at_unix = time.time()

    def request_cancel(self, task_id: str) -> Optional[TaskState]:
        with self._lock:
            state = self._tasks.get(task_id)
            if state and state.status in {"pending", "processing"}:
                state.cancel_requested = True
                state.status = "cancelled"
                state.progress = 100
                state.error = "Analysis cancelled by request."
                state.updated_at_unix = time.time()
            return state

    def should_cancel(self, task_id: str) -> bool:
        with self._lock:
            state = self._tasks.get(task_id)
            return bool(state and state.cancel_requested)


# Global singleton instance of TaskManager
_global_task_manager = TaskManager()

def get_task_manager() -> TaskManager:
    """Return the global TaskManager instance."""
    return _global_task_manager
