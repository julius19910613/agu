#!/usr/bin/env python3
"""Generate a deterministic, license-free synthetic basketball MP4 fixture."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis_outputs/public_benchmark/synthetic_basketball.mp4"


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    width, height, fps, frame_count = 640, 360, 30.0, 120
    writer = cv2.VideoWriter(str(OUTPUT), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the synthetic MP4")
    for frame_index in range(frame_count):
        frame = np.full((height, width, 3), (52, 92, 52), dtype=np.uint8)
        cv2.line(frame, (width // 2, 0), (width // 2, height), (230, 230, 230), 2)
        left_x = 80 + frame_index
        right_x = 500 - frame_index
        cv2.rectangle(frame, (left_x, 170), (left_x + 36, 260), (35, 35, 35), -1)
        cv2.rectangle(frame, (right_x, 150), (right_x + 36, 240), (220, 220, 220), -1)
        cv2.rectangle(frame, (250, 15), (390, 70), (8, 8, 8), -1)
        cv2.putText(frame, "21  19", (270, 53), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 40, 255), 2)
        writer.write(frame)
    writer.release()
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
