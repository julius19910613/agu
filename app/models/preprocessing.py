"""Unified preprocessing for R(2+1)D basketball action recognition.

All three pipelines (train, val, inference) use these functions to ensure a
consistent input distribution matching Kinetics-400 pretraining.
"""

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Lambda

# Kinetics-400 normalization constants (RGB)
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

TARGET_SIZE = 112  # R(2+1)D expects 112x112 spatial input


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to RGB.

    Args:
        frame: Frame array with shape `(H, W, 3)`.

    Returns:
        RGB frame with the same shape as the input.
    """
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_frame(frame: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """Resize a frame to a square spatial size.

    Args:
        frame: Frame array with shape `(H, W, 3)`.
        size: Target height and width.

    Returns:
        Resized frame with shape `(size, size, 3)`.
    """
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)


def preprocess_frame(frame: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """Apply full preprocessing to a single frame.

    Args:
        frame: BGR frame with shape `(H, W, 3)` and values in `[0, 255]`.
        size: Target height and width.

    Returns:
        Float32 RGB frame with shape `(size, size, 3)` normalized to the
        Kinetics-400 distribution.
    """
    frame = bgr_to_rgb(frame)
    frame = resize_frame(frame, size)
    frame = frame.astype(np.float32) / 255.0
    for channel in range(3):
        frame[:, :, channel] = (
            frame[:, :, channel] - KINETICS_MEAN[channel]
        ) / KINETICS_STD[channel]
    return frame


def preprocess_clip_numpy(clip: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """Preprocess a clip stored as a numpy array.

    Args:
        clip: Clip array with shape `(C, T, H, W)` in BGR `[0, 255]` format.
        size: Target height and width.

    Returns:
        Float32 clip with shape `(C, T, size, size)` in RGB and normalized to
        the Kinetics-400 distribution.
    """
    channels, num_frames, _, _ = clip.shape
    result = np.empty((channels, num_frames, size, size), dtype=np.float32)

    for frame_idx in range(num_frames):
        frame_bgr = clip[:, frame_idx, :, :].transpose(1, 2, 0)
        frame_processed = preprocess_frame(frame_bgr, size)
        result[:, frame_idx, :, :] = frame_processed.transpose(2, 0, 1)

    return result


def preprocess_clip_frames(
    frames: list[np.ndarray],
    size: int = TARGET_SIZE,
) -> np.ndarray:
    """Preprocess a list of BGR frames from OpenCV.

    Args:
        frames: List of BGR uint8 frames, each with shape `(H, W, 3)`.
        size: Target height and width.

    Returns:
        Float32 clip with shape `(3, T, size, size)` normalized to the
        Kinetics-400 distribution.
    """
    num_frames = len(frames)
    result = np.empty((3, num_frames, size, size), dtype=np.float32)

    for frame_idx, frame in enumerate(frames):
        processed = preprocess_frame(frame, size)
        result[:, frame_idx, :, :] = processed.transpose(2, 0, 1)

    return result


def _resize_video_tensor(x: torch.Tensor, size: int) -> torch.Tensor:
    """Resize a `(C, T, H, W)` video tensor to the target spatial size."""
    x = x.unsqueeze(0)
    x = torch.nn.functional.interpolate(
        x,
        size=(x.shape[2], size, size),
        mode="trilinear",
        align_corners=False,
    )
    return x.squeeze(0)


def _normalize_video_tensor(x: torch.Tensor) -> torch.Tensor:
    """Normalize a `(C, T, H, W)` RGB video tensor to Kinetics statistics."""
    if x.max() > 1.0:
        x = x / 255.0
    mean = torch.tensor(KINETICS_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1, 1)
    std = torch.tensor(KINETICS_STD, dtype=x.dtype, device=x.device).view(3, 1, 1, 1)
    return (x - mean) / std


def get_val_transforms(size: int = TARGET_SIZE) -> Compose:
    """Build validation transforms for `(C, T, H, W)` video tensors.

    Args:
        size: Target height and width.

    Returns:
        A torchvision `Compose` that resizes and normalizes RGB video tensors.
    """
    return Compose(
        [
            Lambda(lambda x: _resize_video_tensor(x, size)),
            Lambda(_normalize_video_tensor),
        ]
    )
