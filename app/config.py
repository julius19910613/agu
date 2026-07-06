from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables or .env file.

    All fields have sensible defaults matching the original hybrid_analysis.py
    arguments so the app can start with zero configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="BASKETBALL_",
        case_sensitive=False,
    )

    # --- Model checkpoint ---
    model_path: str = "model_checkpoints/r2plus1d_v3/"
    base_model_name: str = "best"
    start_epoch: int = 0
    lr: float = 0.0001
    num_classes: int = 10

    # --- Video pipeline ---
    seq_length: int = 16
    vid_stride: int = 8
    action_vid_stride: int = 24
    batch_size: int = 8
    r2plus1d_device: str = "mps_if_available"
    yolo_device: str = "cpu"
    tracking_fps: float = 8.0
    yolo_imgsz: int = 320
    max_players_per_segment: int = 12
    torch_num_threads: int = 10
    progress_log: bool = True
    tracker_type: str = "YOLO"  # CSRT | YOLO
    tracker_backend: str = "bytetrack"  # bytetrack | botsort | custom
    yolo_tracker_config: str = ""
    yolo_reid_enabled: bool = False
    yolo_reid_model: str = "auto"
    identity_embedding_backend: str = "torchvision_mobilenet_v3_small"
    identity_embedding_weights: str = "default"
    identity_embedding_device: str = "mps_if_available"
    identity_embedding_batch_size: int = 16
    identity_embedding_allow_fallback: bool = True
    yolo_model_name: str = "yolov8n.pt"
    default_video: str = "examples/lebron_shoots.mp4"

    # --- VLM (Ollama) ---
    vlm_mode: str = "low-confidence"  # off | low-confidence | always
    ollama_model: str = "qwen3-vl:4b"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_timeout: float = 45.0
    vlm_frames: int = 1
    vlm_image_width: int = 224
    max_vlm_clips: int = 8
    jersey_number_vlm_enabled: bool = False
    jersey_number_vlm_frames: int = 2
    vlm_identity_merge_enabled: bool = False
    vlm_identity_merge_max_candidates: int = 8
    vlm_identity_merge_confidence: float = 0.78
    vlm_identity_merge_crops_per_side: int = 3

    # --- Confidence thresholds ---
    low_confidence: float = 0.45
    high_confidence: float = 0.70
    smoothing_confidence: float = 0.6

    # --- Output directories ---
    output_dir: str = "analysis_outputs"
    video_output_dir: str = "output_videos"

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8765


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
