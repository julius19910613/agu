import json
import os
import random

import cv2
import numpy as np
from tqdm import tqdm


def augment_horizontal_flip(frames):
    """Flip frames horizontally. Natural for basketball (same court side)."""
    return [cv2.flip(frame, 1) for frame in frames]


def augment_scale_jitter(frames, scale_range=(0.85, 1.15)):
    """Scale jitter — slight zoom in/out. Preserves geometry."""
    if not frames:
        return frames

    scale = random.uniform(*scale_range)
    h, w = frames[0].shape[:2]
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))

    result = []
    for frame in frames:
        resized = cv2.resize(frame, (new_w, new_h))
        if scale > 1.0:
            start_y = max((new_h - h) // 2, 0)
            start_x = max((new_w - w) // 2, 0)
            resized = resized[start_y:start_y + h, start_x:start_x + w]
        else:
            pad_y = max((h - new_h) // 2, 0)
            pad_x = max((w - new_w) // 2, 0)
            canvas = np.zeros_like(frames[0])
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
            resized = canvas
        result.append(resized)
    return result


def augment_brightness_contrast(frames, brightness_range=(0.85, 1.15), contrast_range=(0.85, 1.15)):
    """Adjust brightness and contrast. Simulates lighting variations."""
    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)
    result = []
    for frame in frames:
        adjusted = np.clip(frame.astype(np.float32) * brightness * contrast, 0, 255).astype(np.uint8)
        result.append(adjusted)
    return result


def augment_gaussian_blur(frames, kernel_range=(3, 5)):
    """Mild Gaussian blur. Simulates motion/defocus."""
    if not frames:
        return frames
    kernel = random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))
    sigma = random.uniform(0.5, 1.5)
    return [cv2.GaussianBlur(frame, (kernel, kernel), sigma) for frame in frames]


def augment_temporal_shift(frames, shift_range=(-2, 2)):
    """Shift clip start/end by a few frames."""
    if not frames:
        return frames

    shift = random.randint(*shift_range)
    if shift == 0:
        return frames
    if shift > 0:
        return [frames[0]] * shift + frames[:-shift]
    return frames[-shift:] + [frames[-1]] * (-shift)


def augment_mild_rotation(frames, max_angle=10):
    """Mild rotation (±10° max). Much safer than 30° defaults."""
    if not frames:
        return frames

    angle = random.uniform(-max_angle, max_angle)
    h, w = frames[0].shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return [
        cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        for frame in frames
    ]


def augment_mild_translate(frames, max_pixels=15):
    """Mild translation (±15px max). Safer than large shifts."""
    if not frames:
        return frames

    tx = random.randint(-max_pixels, max_pixels)
    ty = random.randint(-max_pixels, max_pixels)
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = frames[0].shape[:2]
    return [
        cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        for frame in frames
    ]


MINORITY_AUGMENTATIONS = [
    augment_horizontal_flip,
    augment_scale_jitter,
    augment_brightness_contrast,
    augment_temporal_shift,
    augment_gaussian_blur,
    augment_mild_rotation,
    augment_mild_translate,
]


MINORITY_CLASSES = {
    4: "shoot",
    7: "pick",
    0: "block",
    1: "pass",
}


def get_video_frames(video_path):
    """Read all frames from a video file."""
    capture = cv2.VideoCapture(video_path)
    frames = []
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)
    capture.release()
    return frames


def save_video_frames(frames, output_path, fps=30):
    """Save a list of frames to an mp4 file."""
    if not frames:
        raise ValueError("Cannot save an empty frame list")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def keystoint(mapping):
    return {int(key): value for key, value in mapping.items()}


def rotateVideo(path, output_dir, video_id, degree):
    """Legacy helper kept for compatibility."""
    frames = get_video_frames(path)
    if not frames:
        return

    h, w = frames[0].shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1.0)
    rotated = [cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE) for frame in frames]
    save_video_frames(rotated, os.path.join(output_dir, f"{video_id}_rotate_{degree}.mp4"))


def translateVideo(path, output_dir, video_id, translate=(0, 0)):
    """Legacy helper kept for compatibility."""
    frames = get_video_frames(path)
    if not frames:
        return

    h, w = frames[0].shape[:2]
    matrix = np.float32([[1, 0, translate[0]], [0, 1, translate[1]]])
    translated = [cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE) for frame in frames]
    save_video_frames(
        translated,
        os.path.join(output_dir, f"{video_id}_translate_{translate[0]}_{translate[1]}.mp4"),
    )


def augmentVideo(annotation_dict, labels_dict, data_dir="dataset/examples/", output_dir="dataset/augmented-examples/"):
    """
    Legacy augmentation entry point.

    This now uses basketball-safe transforms for the minority classes and writes
    `dataset/augmented_annotation_dict.json` as a plain `{video_id: label}` dict
    so `BasketballDataset` can consume it directly.
    """
    with open(annotation_dict) as handle:
        annotation_data = json.load(handle)
        video_list = list(annotation_data.items())

    with open(labels_dict) as handle:
        labels_data = json.load(handle, object_hook=keystoint)

    count_dict = {}
    for video_id, action in annotation_data.items():
        label_name = labels_data[action]
        count_dict[label_name] = count_dict.get(label_name, 0) + 1

    sorted_dict = {key: value for key, value in sorted(count_dict.items(), key=lambda item: item[1])}
    filtered_actions = {name for name, count in sorted_dict.items() if count <= 2000}

    os.makedirs(output_dir, exist_ok=True)
    augmented_annotation = {}
    pbar = tqdm(video_list)

    for index, (video_id, action) in enumerate(pbar, start=1):
        if labels_data[action] not in filtered_actions:
            pbar.set_description(f"Percentage {index / max(len(video_list), 1):.4f}")
            continue

        video_path = os.path.join(data_dir, f"{video_id}.mp4")
        frames = get_video_frames(video_path)
        if not frames:
            pbar.set_description(f"Percentage {index / max(len(video_list), 1):.4f}")
            continue

        outputs = {
            f"{video_id}_flip": augment_horizontal_flip(frames),
            f"{video_id}_scale": augment_scale_jitter(frames),
            f"{video_id}_bright": augment_brightness_contrast(frames),
            f"{video_id}_shift": augment_temporal_shift(frames),
            f"{video_id}_blur": augment_gaussian_blur(frames),
            f"{video_id}_rotate_mild": augment_mild_rotation(frames),
            f"{video_id}_translate_mild": augment_mild_translate(frames),
        }
        for aug_video_id, aug_frames in outputs.items():
            save_video_frames(aug_frames, os.path.join(output_dir, f"{aug_video_id}.mp4"))
            augmented_annotation[aug_video_id] = action

        pbar.set_description(f"Percentage {index / max(len(video_list), 1):.4f}")

    with open("dataset/augmented_annotation_dict.json", "w", encoding="utf-8") as handle:
        json.dump(augmented_annotation, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    augmentVideo("dataset/annotation_dict.json", "dataset/labels_dict.json")
