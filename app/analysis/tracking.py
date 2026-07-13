from __future__ import annotations

import math
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import json
import numpy as np


TRACKER_TYPES = {
    "BOOSTING": lambda: cv2.legacy.TrackerBoosting_create(),
    "MIL": lambda: cv2.legacy.TrackerMIL_create(),
    "KCF": lambda: cv2.legacy.TrackerKCF_create(),
    "TLD": lambda: cv2.legacy.TrackerTLD_create(),
    "MEDIANFLOW": lambda: cv2.legacy.TrackerMedianFlow_create(),
    "GOTURN": lambda: cv2.legacy.TrackerGOTURN_create(),
    "MOSSE": lambda: cv2.legacy.TrackerMOSSE_create(),
    "CSRT": lambda: cv2.legacy.TrackerCSRT_create(),
}


def create_tracker_by_name(tracker_type: str) -> cv2.legacy.Tracker:
    """Create an OpenCV legacy tracker by name.

    Args:
        tracker_type: One of BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT.

    Returns:
        An initialized tracker instance.

    Raises:
        ValueError: If the tracker type is unknown.
    """
    tracker_key = tracker_type.upper()
    if tracker_key not in TRACKER_TYPES:
        available = ", ".join(TRACKER_TYPES)
        raise ValueError(f"Unknown tracker '{tracker_type}'. Available: {available}")
    return TRACKER_TYPES[tracker_key]()


def default_headless_boxes(width: int, height: int) -> List[Tuple[int, int, int, int]]:
    """Return default bounding boxes scaled to the given video dimensions.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        List of (x, y, w, h) tuples for two default player boxes.
    """
    scale_x = width / 1280.0
    scale_y = height / 720.0
    return [
        (int(350 * scale_x), int(100 * scale_y), int(150 * scale_x), int(400 * scale_y)),
        (int(600 * scale_x), int(120 * scale_y), int(150 * scale_x), int(400 * scale_y)),
    ]


def read_boxes_file(path: str) -> List[Tuple[int, int, int, int]]:
    """Read bounding boxes from a JSON file.

    Accepts either ``{"boxes": [[x, y, w, h], ...]}`` or ``[[x, y, w, h], ...]``.

    Args:
        path: Path to a JSON file.

    Returns:
        List of (x, y, w, h) tuples.
    """
    with open(path, "r") as fp:
        payload = json.load(fp)
    boxes = payload["boxes"] if isinstance(payload, dict) else payload
    return [tuple(int(v) for v in box) for box in boxes]


def select_active_track_ids(
    appearance_counts: Dict[int, int],
    frame_count: int,
    min_appear_ratio: float = 0.02,
    min_appear_abs: int = 5,
    max_players: Optional[int] = None,
) -> List[int]:
    """Select stable player track IDs, optionally capped by strongest tracks."""
    min_appearances = max(min_appear_abs, int(frame_count * min_appear_ratio))
    active_track_ids = [
        tid
        for tid, count in appearance_counts.items()
        if count >= min_appearances
    ]
    if not active_track_ids and appearance_counts:
        active_track_ids = [max(appearance_counts, key=appearance_counts.get)]

    active_track_ids.sort(key=lambda tid: (-appearance_counts.get(tid, 0), tid))
    if max_players is not None and max_players > 0:
        active_track_ids = active_track_ids[:max_players]
    return sorted(active_track_ids)


def densify_track_boxes(
    raw_track_data: Sequence[Dict[int, Tuple[float, float, float, float]]],
    active_track_ids: Sequence[int],
    max_gap_frames: int = 2,
) -> List[Tuple[Tuple[float, float, float, float], ...]]:
    """Build dense box rows while only interpolating brief detector dropouts."""
    missing_box = (0.0, 0.0, 0.0, 0.0)
    tracks: Dict[int, List[Optional[Tuple[float, float, float, float]]]] = {
        track_id: [frame_boxes.get(track_id) for frame_boxes in raw_track_data]
        for track_id in active_track_ids
    }
    for track_id, boxes in tracks.items():
        known_indices = [index for index, box in enumerate(boxes) if box is not None]
        for left_index, right_index in zip(known_indices, known_indices[1:]):
            missing_count = right_index - left_index - 1
            if missing_count <= 0 or missing_count > max_gap_frames:
                continue
            left_box = np.asarray(boxes[left_index], dtype=np.float32)
            right_box = np.asarray(boxes[right_index], dtype=np.float32)
            for offset in range(1, missing_count + 1):
                alpha = offset / float(missing_count + 1)
                interpolated = left_box * (1.0 - alpha) + right_box * alpha
                boxes[left_index + offset] = tuple(float(value) for value in interpolated)

    return [
        tuple(tracks[track_id][frame_index] or missing_box for track_id in active_track_ids)
        for frame_index in range(len(raw_track_data))
    ]


def resolve_yolo_tracker_config(
    tracker_backend: str = "bytetrack",
    yolo_tracker_config: str = "",
    reid_enabled: bool = False,
    reid_model: str = "auto",
) -> str:
    """Resolve the Ultralytics tracker adapter config used by AGU."""
    backend = (tracker_backend or "bytetrack").strip().lower()
    tracker_config = (yolo_tracker_config or "").strip()
    if not tracker_config:
        tracker_config = "botsort.yaml" if backend == "botsort" else "bytetrack.yaml"

    if not reid_enabled:
        return tracker_config

    if backend != "botsort" and "botsort" not in tracker_config.lower():
        return tracker_config

    model_value = (reid_model or "auto").strip() or "auto"
    temp = tempfile.NamedTemporaryFile(prefix="agu-botsort-reid-", suffix=".yaml", delete=False, mode="w")
    temp.write(
        "\n".join(
            [
                "tracker_type: botsort",
                "track_high_thresh: 0.25",
                "track_low_thresh: 0.1",
                "new_track_thresh: 0.25",
                "track_buffer: 30",
                "match_thresh: 0.8",
                "fuse_score: True",
                "gmc_method: sparseOptFlow",
                "proximity_thresh: 0.5",
                "appearance_thresh: 0.25",
                "with_reid: True",
                f"model: {model_value}",
                "",
            ]
        )
    )
    temp.close()
    return temp.name


def extract_tracked_frames(
    video_path: str,
    tracker_type: str = "CSRT",
    headless: bool = True,
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    boxes_file: Optional[str] = None,
    max_frames: Optional[int] = None,
    conf_thres: float = 0.3,
    iou_thres: float = 0.6,
    min_appear_ratio: float = 0.02,
    min_appear_abs: int = 5,
    device: Optional[str] = None,
    yolo_model_name: str = "yolov8n.pt",
    tracker_backend: str = "bytetrack",
    yolo_tracker_config: str = "",
    reid_enabled: bool = False,
    reid_model: str = "auto",
    tracking_fps: Optional[float] = None,
    yolo_imgsz: int = 640,
    max_players: Optional[int] = None,
) -> Tuple[
    List[np.ndarray],
    List[Tuple[Tuple[float, float, float, float], ...]],
    int,
    int,
    List[Tuple[int, int, int]],
]:
    """Extract video frames with multi-object tracking.

    Args:
        video_path: Path to the input video file.
        tracker_type: OpenCV tracker algorithm name or 'YOLO'.
        headless: If True, skip GUI-based ROI selection.
        boxes: Explicit initial bounding boxes. Takes priority over boxes_file.
        boxes_file: Path to a JSON file with initial bounding boxes.
        max_frames: Optional cap on the number of frames to process.
        device: Torch compute device (used for YOLO).
        tracker_backend: Open-source tracker adapter name, such as bytetrack or botsort.
        yolo_tracker_config: Ultralytics tracker YAML name/path.
        reid_enabled: Enables generated BoT-SORT ReID config when supported.
        reid_model: ReID model setting for BoT-SORT.

    Returns:
        Tuple of (video_frames, player_boxes, width, height, colors).
    """
    if tracker_type.upper() == "YOLO":
        from ultralytics import YOLO
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        source_fps = 0.0
        cap = cv2.VideoCapture(video_path)
        try:
            if cap.isOpened():
                source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            cap.release()

        frame_stride = 1
        if tracking_fps is not None and tracking_fps > 0 and source_fps > tracking_fps:
            frame_stride = max(1, int(round(source_fps / tracking_fps)))

        # Load YOLOv8 Nano model
        yolo_model = YOLO(yolo_model_name)
        tracker_config = resolve_yolo_tracker_config(
            tracker_backend=tracker_backend,
            yolo_tracker_config=yolo_tracker_config,
            reid_enabled=reid_enabled,
            reid_model=reid_model,
        )

        # Run tracking through the configured Ultralytics tracker adapter.
        results = yolo_model.track(
            source=video_path,
            tracker=tracker_config,
            device=device,
            classes=[0],  # Person only
            persist=True,
            stream=True,
            verbose=False,
            conf=conf_thres,
            iou=iou_thres,
            imgsz=yolo_imgsz,
            vid_stride=frame_stride,
        )

        video_frames: List[np.ndarray] = []
        raw_track_data: List[Dict[int, Tuple[float, float, float, float]]] = []
        appearance_counts: Dict[int, int] = {}

        for idx, r in enumerate(results):
            if max_frames is not None and idx >= max_frames:
                break
            
            video_frames.append(r.orig_img.copy())

            frame_boxes = {}
            if r.boxes is not None and r.boxes.id is not None:
                xywh = r.boxes.xywh.cpu().numpy()
                ids = r.boxes.id.int().cpu().numpy()
                for box, track_id in zip(xywh, ids):
                    track_id = int(track_id)
                    appearance_counts[track_id] = appearance_counts.get(track_id, 0) + 1
                    
                    x_center, y_center, w, h = box
                    x = x_center - w / 2
                    y = y_center - h / 2
                    frame_boxes[track_id] = (float(x), float(y), float(w), float(h))
            raw_track_data.append(frame_boxes)

        if not video_frames:
            raise RuntimeError("No frames extracted from video.")

        height, width = video_frames[0].shape[:2]

        active_track_ids = select_active_track_ids(
            appearance_counts=appearance_counts,
            frame_count=len(video_frames),
            min_appear_ratio=min_appear_ratio,
            min_appear_abs=min_appear_abs,
            max_players=max_players,
        )
        if not active_track_ids:
            active_track_ids = [0]

        player_boxes = densify_track_boxes(
            raw_track_data,
            active_track_ids,
            max_gap_frames=2,
        )

        colors: List[Tuple[int, int, int]] = [
            (255, 0, 0),    # Red
            (0, 0, 255),    # Blue
            (0, 180, 0),    # Green
            (255, 160, 0),  # Orange
            (180, 0, 180),  # Purple
            (0, 180, 180),  # Cyan
        ]
        while len(colors) < len(active_track_ids):
            colors.append(tuple(int(x) for x in np.random.randint(0, 256, size=3).tolist()))

        return video_frames, player_boxes, width, height, colors[: len(active_track_ids)]

    # Original OpenCV multi-tracker path
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        success, first_frame = cap.read()
        if not success:
            raise RuntimeError(f"Failed to read video: {video_path}")

        height, width = first_frame.shape[:2]

        if boxes is not None:
            init_boxes = boxes
        elif boxes_file:
            init_boxes = read_boxes_file(boxes_file)
        elif headless:
            init_boxes = default_headless_boxes(width, height)
        else:
            init_boxes = []
            while True:
                box = cv2.selectROI("HybridMultiTracker", first_frame, fromCenter=False, showCrosshair=True)
                init_boxes.append(tuple(int(v) for v in box))
                print("Press q to quit selecting boxes and start tracking")
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    break

        colors: List[Tuple[int, int, int]] = [(255, 0, 0), (0, 0, 255), (0, 180, 0), (255, 160, 0)]
        while len(colors) < len(init_boxes):
            colors.append(tuple(int(x) for x in np.random.randint(0, 256, size=3).tolist()))

        trackers = cv2.legacy.MultiTracker_create()
        for box in init_boxes:
            trackers.add(create_tracker_by_name(tracker_type), first_frame, box)

        video_frames: List[np.ndarray] = []
        player_boxes: List[Tuple[Tuple[float, float, float, float], ...]] = []

        frame = first_frame
        is_first = True
        while True:
            raw_frame = frame.copy()
            if is_first:
                player_boxes.append(tuple(tuple(float(v) for v in box) for box in init_boxes))
                is_first = False
            else:
                success, tracked_boxes = trackers.update(frame)
                if not success:
                    fallback = player_boxes[-1] if player_boxes else tuple(tuple(float(v) for v in box) for box in init_boxes)
                    player_boxes.append(fallback)
                else:
                    player_boxes.append(tuple(tuple(float(v) for v in box) for box in tracked_boxes))
            video_frames.append(raw_frame)

            if max_frames is not None and len(video_frames) >= max_frames:
                break

            success, frame = cap.read()
            if not success:
                break
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

    return video_frames, player_boxes, width, height, colors[: len(init_boxes)]


def crop_video(
    clip: Sequence[np.ndarray],
    crop_window: Sequence[Sequence[Sequence[float]]],
    player: int = 0,
    output_size: Tuple[int, int] = (128, 176),
) -> List[np.ndarray]:
    """Crop and resize player regions from a clip of frames.

    Args:
        clip: Sequence of video frames (H, W, 3).
        crop_window: Per-frame per-player bounding boxes.
        player: Player index to crop.
        output_size: (width, height) of the output crop.

    Returns:
        List of resized cropped frames with shape (height, width, 3).
    """
    video: List[np.ndarray] = []
    w_out, h_out = output_size
    for idx, frame in enumerate(clip):
        x, y, w, h = [int(v) for v in crop_window[idx][player]]
        if w <= 1 or h <= 1:
            video.append(np.zeros((h_out, w_out, 3), dtype=np.uint8))
            continue
        cropped = frame[max(y, 0): max(y + h, 0), max(x, 0): max(x + w, 0)]
        try:
            resized = cv2.resize(cropped, dsize=(w_out, h_out), interpolation=cv2.INTER_NEAREST)
        except cv2.error:
            resized = np.zeros((h_out, w_out, 3), dtype=np.uint8)
        video.append(resized)
    return video


def crop_windows(
    video_frames: Sequence[np.ndarray],
    player_boxes: Sequence[Sequence[Sequence[float]]],
    seq_length: int = 16,
    vid_stride: int = 8,
    min_visible_ratio: float = 0.0,
    return_clip_indices: bool = False,
) -> Dict[int, List[np.ndarray]] | tuple[Dict[int, List[np.ndarray]], Dict[int, List[int]]]:
    """Split video into overlapping windows and crop each player.

    Args:
        video_frames: Full list of video frames.
        player_boxes: Per-frame per-player bounding boxes.
        seq_length: Number of frames per clip window.
        vid_stride: Stride between clip windows.

    Returns:
        Dict mapping player index to list of clip arrays, each with shape
        (seq_length, H, W, 3).

    Raises:
        ValueError: If inputs are empty.
    """
    if not video_frames or not player_boxes:
        raise ValueError("Cannot crop windows from empty video or empty player boxes")

    player_count = len(player_boxes[0])
    player_frames: Dict[int, List[np.ndarray]] = {p: [] for p in range(player_count)}
    player_clip_indices: Dict[int, List[int]] = {p: [] for p in range(player_count)}
    n_clips = max(1, math.ceil((len(video_frames) - seq_length) / vid_stride) + 1)

    for clip_idx in range(n_clips):
        start = clip_idx * vid_stride
        end = start + seq_length
        clip = list(video_frames[start:end])
        crop_win = list(player_boxes[start:end])

        if len(clip) < seq_length:
            remaining = seq_length - len(clip)
            clip.extend([np.zeros_like(video_frames[0]) for _ in range(remaining)])
            last_boxes = crop_win[-1] if crop_win else player_boxes[-1]
            crop_win.extend([last_boxes for _ in range(remaining)])

        for player in range(player_count):
            visible_count = sum(
                1
                for frame_boxes in crop_win[: len(clip)]
                if player < len(frame_boxes)
                and float(frame_boxes[player][2]) > 1.0
                and float(frame_boxes[player][3]) > 1.0
            )
            if visible_count / max(1, seq_length) < max(0.0, min(1.0, min_visible_ratio)):
                continue
            player_frames[player].append(np.asarray(crop_video(clip, crop_win, player)))
            player_clip_indices[player].append(clip_idx)

    if return_clip_indices:
        return player_frames, player_clip_indices
    return player_frames
