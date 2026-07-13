"""Typed, dependency-light pipeline primitives for incremental AGU modularity."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Generic, Literal, Protocol, TypeVar

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")
StagePosition = Literal["before_dispatch", "after_dispatch"]


@dataclass
class PipelineContext(Generic[RequestT, ResultT]):
    request: RequestT
    result: ResultT | None = None
    services: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)


class AnalysisStage(Protocol[RequestT, ResultT]):
    name: str

    def run(self, context: PipelineContext[RequestT, ResultT]) -> None:
        ...


@dataclass(frozen=True)
class CallableStage(Generic[RequestT, ResultT]):
    name: str
    handler: Callable[[PipelineContext[RequestT, ResultT]], None]

    def run(self, context: PipelineContext[RequestT, ResultT]) -> None:
        self.handler(context)


class PipelineRunner(Generic[RequestT, ResultT]):
    def __init__(self, stages: list[AnalysisStage[RequestT, ResultT]]) -> None:
        if not stages:
            raise ValueError("PipelineRunner requires at least one stage")
        names = [stage.name for stage in stages]
        if len(names) != len(set(names)):
            raise ValueError("Pipeline stage names must be unique")
        self.stages = list(stages)

    def run(self, context: PipelineContext[RequestT, ResultT]) -> PipelineContext[RequestT, ResultT]:
        for stage in self.stages:
            started = perf_counter()
            status = "completed"
            try:
                stage.run(context)
            except Exception:
                status = "failed"
                raise
            finally:
                context.trace.append(
                    {
                        "stage": stage.name,
                        "status": status,
                        "duration_ms": round((perf_counter() - started) * 1000.0, 3),
                    }
                )
        return context


_EXTENSION_STAGES: dict[StagePosition, dict[str, AnalysisStage[Any, Any]]] = {
    "before_dispatch": {},
    "after_dispatch": {},
}


def register_analysis_stage(
    stage: AnalysisStage[Any, Any],
    *,
    position: StagePosition = "after_dispatch",
    replace: bool = False,
) -> None:
    """Register a process-wide extension stage at a stable hook boundary."""
    stages = _EXTENSION_STAGES[position]
    if stage.name in stages and not replace:
        raise ValueError(f"Pipeline stage already registered at {position}: {stage.name}")
    stages[stage.name] = stage


def list_analysis_stages(position: StagePosition) -> list[AnalysisStage[Any, Any]]:
    return [stage for _, stage in sorted(_EXTENSION_STAGES[position].items())]


def clear_analysis_stages() -> None:
    """Clear extension stages; intended for isolated tests and plugin reloads."""
    for stages in _EXTENSION_STAGES.values():
        stages.clear()


def pipeline_manifest(context: PipelineContext[Any, Any]) -> dict[str, Any]:
    return {
        "pipeline_version": "1",
        "stages": [dict(item) for item in context.trace],
        "metadata": dict(context.metadata),
    }
