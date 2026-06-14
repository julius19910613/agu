"""Preprocessing helpers kept for the deployed v3 inference/training path.

The v3 deployment intentionally keeps video data in BGR order with value range
`[0, 255]`, resizes spatially to `112x112`, and does no `/255` normalization or
Kinetics mean/std normalization.
"""

import cv2
import numpy as np
import torch

TARGET_SIZE = 112  # R(2+1)D expects 112x112 spatial input


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Return a BGR frame as-is while keeping a safe numeric range."""
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


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
    """Resize one BGR frame for v3 without color change or normalization."""
    frame = resize_frame(frame, size)
    return frame.astype(np.float32)


def preprocess_clip_numpy(clip: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """Preprocess a clip stored as a numpy array.

    Args:
        clip: Clip array with shape `(C, T, H, W)` in BGR `[0, 255]` format.
        size: Target height and width.

    Returns:
        Float32 clip with shape `(C, T, size, size)` in BGR and range `[0, 255]`.
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
    """Preprocess a list of BGR frames from OpenCV for v3 inference."""
    num_frames = len(frames)
    result = np.empty((3, num_frames, size, size), dtype=np.float32)

    for frame_idx, frame in enumerate(frames):
        processed = preprocess_frame(bgr_to_rgb(frame), size)
        result[:, frame_idx, :, :] = processed.transpose(2, 0, 1)

    return result


def _resize_video_tensor(x: torch.Tensor, size: int) -> torch.Tensor:
    """Resize `(C, T, H, W)` while keeping v3 range and channel order."""
    x = x.unsqueeze(0)
    x = torch.nn.functional.interpolate(
        x,
        size=(x.shape[2], size, size),
        mode="trilinear",
        align_corners=False,
    )
    return x.squeeze(0)


def _normalize_video_tensor(x: torch.Tensor) -> torch.Tensor:
    """Keep v3 clip values unchanged; no mean/std normalization."""
    return x


def get_val_transforms(size: int = TARGET_SIZE):
    """Torchvision-compatible resize + no-op normalization transform."""
    from torchvision.transforms import Compose, Lambda

    return Compose(
        [
            Lambda(lambda x: _resize_video_tensor(x, size)),
            Lambda(_normalize_video_tensor),
        ]
    )
