from __future__ import annotations

from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch

from app.analysis.schemas import ModelPrediction

# Label configuration from hybrid_analysis.py
LABELS: Dict[int, str] = {
    0: "block",
    1: "pass",
    2: "run",
    3: "dribble",
    4: "shoot",
    5: "ball in hand",
    6: "defense",
    7: "pick",
    8: "no_action",
    9: "walk",
}

LABEL_TO_ID = {value: key for key, value in LABELS.items()}

# Target spatial size matching v3 training (VideoToNumpy + VideoTransform)
_TARGET_SIZE = 112


def _preprocess_clip_v3(clip_frames: np.ndarray) -> np.ndarray:
    """Preprocess a clip to match v3 training pipeline (VideoToNumpy format).

    v3 was trained with `VideoToNumpy` which produces BGR [0,255] arrays,
    followed by `VideoTransform(val)` which resizes to 112x112.
    No color conversion, no /255, no Kinetics normalization.

    Args:
        clip_frames: Array of shape `(T, H, W, C)` in BGR uint8 or float [0,255].

    Returns:
        Float32 array of shape `(C, T, 112, 112)` in BGR [0,255].
    """
    num_frames = clip_frames.shape[0]
    result = np.empty((3, num_frames, _TARGET_SIZE, _TARGET_SIZE), dtype=np.float32)

    for i in range(num_frames):
        frame = clip_frames[i]
        # Ensure uint8 for resize
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame_resized = cv2.resize(frame, (_TARGET_SIZE, _TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
        # (H, W, C) → (C, H, W)
        result[:, i, :, :] = frame_resized.transpose(2, 0, 1).astype(np.float32)

    return result


def inference_batch(batch: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Prepare a batch of clips for R(2+1)D model inference.

    Applies the same preprocessing as the v3 training pipeline:
    - Keeps BGR color order (matching VideoToNumpy)
    - Resizes frames to 112x112
    - Values remain in [0, 255] (no /255, no Kinetics normalization)

    Args:
        batch: Batch of clips with shape `(B, T, H, W, C)` in BGR `[0, 255]`.

    Returns:
        Tensor of shape `(B, C, T, 112, 112)` in BGR float32 [0, 255].
    """
    batch_np = batch.detach().cpu().numpy() if isinstance(batch, torch.Tensor) else np.asarray(batch)
    processed_clips = [
        _preprocess_clip_v3(clip_frames)
        for clip_frames in batch_np
    ]
    return torch.from_numpy(np.stack(processed_clips, axis=0)).float()


def predict_player_clips(
    model: torch.nn.Module,
    player_clips: Dict[int, Sequence[np.ndarray]],
    device: torch.device | None = None,
    batch_size: int = 8,
) -> Dict[int, List[ModelPrediction]]:
    """Run model inference on pre-cropped video windows for all players.

    Args:
        model: Loaded PyTorch R(2+1)D model.
        player_clips: Dictionary mapping player index to list of clip arrays.
        device: Target compute device. Auto-detected if None.
        batch_size: Inference batch size.

    Returns:
        Dictionary mapping player index to list of ModelPrediction objects.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_predictions: Dict[int, List[ModelPrediction]] = {}

    for player, clips in player_clips.items():
        predictions: List[ModelPrediction] = []
        for start in range(0, len(clips), batch_size):
            batch_np = np.asarray(clips[start : start + batch_size])
            batch = inference_batch(batch_np).to(device)
            with torch.no_grad():
                outputs = model(batch)
                softmax = torch.softmax(outputs, dim=1).detach().cpu().numpy()

            for row in softmax:
                action_id = int(np.argmax(row))
                probabilities = {
                    LABELS[idx]: float(prob)
                    for idx, prob in enumerate(row[: len(LABELS)])
                }
                predictions.append(
                    ModelPrediction(
                        action_id=action_id,
                        action=LABELS[action_id],
                        confidence=float(row[action_id]),
                        probabilities=probabilities,
                    )
                )
        all_predictions[player] = predictions

    return all_predictions
