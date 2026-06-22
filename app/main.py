from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models.r2plus1d import build_r2plus1d_model
from app.dependencies import init_globals
from app.analysis.router import router as analysis_router


def resolve_model_device(preference: str) -> torch.device:
    if not isinstance(preference, str):
        preference = "auto"
    preference = (preference or "auto").lower()
    if preference in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    if preference == "mps_if_available":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if preference == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preference)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """App lifespan context manager. Loads ML models on startup."""
    settings = get_settings()
    
    try:
        torch_num_threads = int(settings.torch_num_threads)
    except (TypeError, ValueError):
        torch_num_threads = 0
    if torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)

    device = resolve_model_device(settings.r2plus1d_device)
    print(f"Loading R(2+1)D model on {device}...")
    model = build_r2plus1d_model(settings, device=device)
    init_globals(model=model, device=device)
    print("Model loaded successfully. App ready.")
    
    # Mount output directories for static file serving using absolute paths
    abs_output_dir = os.path.abspath(settings.output_dir)
    abs_video_output_dir = os.path.abspath(settings.video_output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    os.makedirs(abs_video_output_dir, exist_ok=True)
    
    app.mount("/static/outputs", StaticFiles(directory=abs_output_dir), name="outputs")
    app.mount("/static/videos", StaticFiles(directory=abs_video_output_dir), name="videos")
    
    yield
    
    print("Shutting down...")


app = FastAPI(
    title="Basketball Defense Analysis API",
    description="Hybrid Spatio-Temporal classification of basketball actions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(analysis_router, prefix="/api/v1")


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["system"])
def ready_check() -> dict[str, str]:
    return {"status": "ready"}
