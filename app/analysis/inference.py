from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from app.analysis.schemas import ModelPrediction
from app.models.preprocessing import preprocess_clip_frames

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


def inference_batch(batch: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Prepare a batch of clips for R(2+1)D model inference.

    Applies the same unified preprocessing as the training pipeline:
    - Converts BGR frames to RGB
    - Resizes frames to 112x112
    - Divides by 255
    - Normalizes to Kinetics-400 mean and standard deviation

    Args:
        batch: Batch of clips with shape `(B, T, H, W, C)` in BGR `[0, 255]`.

    Returns:
        Tensor of shape `(B, C, T, 112, 112)` in normalized RGB float32.
    """
    batch_np = batch.detach().cpu().numpy() if isinstance(batch, torch.Tensor) else np.asarray(batch)
    processed_clips = [
        preprocess_clip_frames(list(clip_frames))
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
