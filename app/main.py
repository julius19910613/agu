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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """App lifespan context manager. Loads ML models on startup."""
    settings = get_settings()
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
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
