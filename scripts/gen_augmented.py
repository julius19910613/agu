"""Generate augmented samples for minority classes.

Usage:
    python scripts/gen_augmented.py --multiplier 3 --minority-only
    python scripts/gen_augmented.py --multiplier 5
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from augment_videos import MINORITY_AUGMENTATIONS, MINORITY_CLASSES


LABELS = {
    0: "block",
    1: "pass",
    2: "run",
    3: "dribble",
    4: "shoot",
    5: "ball_in_hand",
    6: "defense",
    7: "pick",
    8: "no_action",
    9: "walk",
}


def get_video_frames(video_path):
    """Read all frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def apply_random_augmentations(frames, n_augs=2):
    """Apply n_augs random augmentations from the pool."""
    if not frames:
        return frames
    augmentations = random.sample(MINORITY_AUGMENTATIONS, min(n_augs, len(MINORITY_AUGMENTATIONS)))
    augmented_frames = frames
    for aug_fn in augmentations:
        augmented_frames = aug_fn(augmented_frames)
    return augmented_frames


def save_video(frames, output_path, fps=30):
    """Save frames as video."""
    if not frames:
        raise ValueError("Cannot save an empty frame list")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _load_annotations(annotation_path):
    with open(annotation_path, encoding="utf-8") as handle:
        annotations = json.load(handle)

    if isinstance(annotations, dict):
        items = []
        for video_key, label in annotations.items():
            items.append({"video": video_key, "label": int(label)})
        return items

    if isinstance(annotations, list):
        return annotations

    raise TypeError(f"Unsupported annotation format: {type(annotations).__name__}")


def _extract_label(annotation):
    if "label" in annotation:
        return int(annotation["label"])
    if "action" in annotation:
        action = annotation["action"]
        if isinstance(action, int):
            return action
        return int(np.argmax(action))
    raise KeyError("Annotation entry is missing both 'label' and 'action'")


def _resolve_video_path(video_dir, annotation):
    video_key = annotation.get("video", annotation.get("path", ""))
    if not video_key:
        return None, None

    basename = os.path.basename(video_key)
    stem, ext = os.path.splitext(basename)
    if ext:
        filename = basename
        video_id = stem
    else:
        filename = f"{basename}.mp4"
        video_id = basename
    return os.path.join(video_dir, filename), video_id


def _load_existing_outputs(output_annotation):
    if not os.path.exists(output_annotation):
        return {}
    with open(output_annotation, encoding="utf-8") as handle:
        existing = json.load(handle)
    return existing if isinstance(existing, dict) else {}


def main():
    parser = argparse.ArgumentParser(description="Generate augmented samples for minority classes")
    parser.add_argument("--annotation-path", default="dataset/annotation_dict.json")
    parser.add_argument("--video-dir", default="dataset/examples/")
    parser.add_argument("--output-dir", default="dataset/augmented-examples/")
    parser.add_argument("--output-annotation", default="dataset/augmented_annotation_dict.json")
    parser.add_argument(
        "--multiplier",
        type=int,
        default=3,
        help="Number of augmented copies per original sample",
    )
    parser.add_argument(
        "--minority-only",
        action="store_true",
        help="Only augment minority classes (shoot, pick, block, pass)",
    )
    parser.add_argument(
        "--n-augs",
        type=int,
        default=2,
        help="Number of random augmentations to apply per sample",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    annotations = _load_annotations(args.annotation_path)
    existing_outputs = _load_existing_outputs(args.output_annotation)
    augmented_annotations = {}
    total_original = 0
    total_augmented = 0

    for idx, ann in enumerate(tqdm(annotations, desc="Processing samples")):
        try:
            label = _extract_label(ann)
        except (KeyError, TypeError, ValueError):
            continue

        if args.minority_only and label not in MINORITY_CLASSES:
            continue

        video_path, video_id = _resolve_video_path(args.video_dir, ann)
        if not video_path or not os.path.exists(video_path):
            continue

        frames = get_video_frames(video_path)
        if len(frames) < 5:
            continue

        capture = cv2.VideoCapture(video_path)
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 30
        capture.release()

        total_original += 1

        for copy_idx in range(args.multiplier):
            aug_frames = apply_random_augmentations(frames, args.n_augs)
            aug_video_id = f"aug_{idx}_{copy_idx}_{video_id}"
            aug_filename = f"{aug_video_id}.mp4"
            aug_path = os.path.join(args.output_dir, aug_filename)
            save_video(aug_frames, aug_path, fps=source_fps)
            augmented_annotations[aug_video_id] = label
            total_augmented += 1

    stale_keys = set(existing_outputs) - set(augmented_annotations)
    for stale_key in stale_keys:
        if not stale_key.startswith("aug_"):
            continue
        stale_path = os.path.join(args.output_dir, f"{stale_key}.mp4")
        if os.path.exists(stale_path):
            os.remove(stale_path)

    with open(args.output_annotation, "w", encoding="utf-8") as handle:
        json.dump(augmented_annotations, handle, indent=2, sort_keys=True)

    print("\nAugmentation complete.")
    print(f"  Original samples processed: {total_original}")
    print(f"  Augmented samples generated: {total_augmented}")
    print(f"  Annotations saved to: {args.output_annotation}")

    label_counts = Counter(augmented_annotations.values())
    print("\n  Augmented class distribution:")
    for label_id in sorted(label_counts):
        print(f"    {LABELS.get(label_id, '???')}: +{label_counts[label_id]}")


if __name__ == "__main__":
    main()
