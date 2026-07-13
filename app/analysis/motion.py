from __future__ import annotations

from typing import Sequence

import numpy as np

from app.analysis.schemas import MotionFeatures


def compute_motion_features(
    player_boxes: Sequence[Sequence[Sequence[float]]],
    player: int,
    clip_index: int,
    seq_length: int,
    vid_stride: int,
) -> MotionFeatures:
    """Compute physical motion features for a player over a clip window.

    Args:
        player_boxes: Global sequence of bounding boxes per frame, per player.
        player: Target player index.
        clip_index: Index of the current clip window.
        seq_length: Number of frames in the window.
        vid_stride: Stride between windows.

    Returns:
        A MotionFeatures object containing computed speeds and areas.
    """
    start = clip_index * vid_stride
    window = player_boxes[start : start + seq_length]
    for frame_boxes in window:
        if player < 0 or player >= len(frame_boxes):
            raise IndexError(f"Player index {player} out of bounds for frame boxes of length {len(frame_boxes)}")
    boxes = [
        np.asarray(frame_boxes[player], dtype=float)
        for frame_boxes in window
        if float(frame_boxes[player][2]) > 1.0 and float(frame_boxes[player][3]) > 1.0
    ]
    if not boxes:
        return MotionFeatures(
            avg_center_speed=0.0,
            max_center_speed=0.0,
            avg_box_area=0.0,
            area_change_ratio=0.0,
        )

    centers = np.asarray([(box[0] + box[2] / 2, box[1] + box[3] / 2) for box in boxes])
    areas = np.asarray([max(box[2], 0.0) * max(box[3], 0.0) for box in boxes])
    
    if len(centers) > 1:
        speeds = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    else:
        speeds = np.asarray([0.0])

    first_area = float(areas[0]) if float(areas[0]) > 0 else 1.0
    
    return MotionFeatures(
        avg_center_speed=float(np.mean(speeds)),
        max_center_speed=float(np.max(speeds)),
        avg_box_area=float(np.mean(areas)),
        area_change_ratio=float((areas[-1] - areas[0]) / first_area),
    )
