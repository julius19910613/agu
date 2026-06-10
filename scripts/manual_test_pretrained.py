"""Validate pretrained-backbone R(2+1)D model with adam.mp4 (max 150 frames).

Runs the full pipeline locally (no API server needed) and prints the
action distribution per player so we can compare against the all-walk
result from the empty-trained checkpoint.
"""
import sys, os, time, json, math
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from app.config import Settings
from app.models.r2plus1d_v2 import build_r2plus1d_model
from app.analysis.tracking import extract_tracked_frames, crop_windows
from app.analysis.inference import predict_player_clips

LABELS = {
    0: "block", 1: "pass", 2: "run", 3: "dribble", 4: "shoot",
    5: "ball in hand", 6: "defense", 7: "pick", 8: "no_action", 9: "walk",
}

def main():
    settings = Settings()
    device = torch.device("cpu")  # CPU to avoid MPS memory pressure
    print(f"[Device] {device}")

    # 1. Build model with pretrained backbone
    print("[Model] Loading Kinetics-400 pretrained backbone...")
    t0 = time.time()
    model = build_r2plus1d_model(settings, device, strategy="pretrained")
    print(f"[Model] Loaded in {time.time()-t0:.1f}s")

    # 2. Track players
    print("[Tracking] Running YOLO + ByteTrack on adam.mp4 (max 150 frames)...")
    t0 = time.time()
    video_frames, player_boxes, width, height, colors = extract_tracked_frames(
        video_path="examples/adam.mp4",
        tracker_type="YOLO",
        headless=True,
        boxes_file=None,
        max_frames=150,
        conf_thres=0.3,
        iou_thres=0.6,
        min_appear_ratio=0.02,
        min_appear_abs=5,
        device=device,
    )
    print(f"[Tracking] Done in {time.time()-t0:.1f}s — {len(video_frames)} frames, "
          f"{len(colors)} players, {width}x{height}")

    # 3. Crop windows
    print("[Cropping] Extracting player clips...")
    t0 = time.time()
    player_clips = crop_windows(
        video_frames, player_boxes,
        seq_length=settings.seq_length,
        vid_stride=settings.vid_stride,
    )
    print(f"[Cropping] Done in {time.time()-t0:.1f}s")
    for pid, clips in player_clips.items():
        print(f"  Player {pid}: {len(clips)} clips")

    # 4. Inference
    print("[Inference] Running R(2+1)D on all clips...")
    t0 = time.time()
    predictions = predict_player_clips(model, player_clips, device, settings.batch_size)
    print(f"[Inference] Done in {time.time()-t0:.1f}s")

    # 5. Summarise
    print("\n" + "="*55)
    print("  RESULTS — Pretrained Backbone (random fc head)")
    print("="*55)

    total_clips = 0
    global_actions = Counter()
    for pid in sorted(predictions.keys()):
        preds = predictions[pid]
        actions = Counter(p.action for p in preds)
        global_actions.update(actions)
        total_clips += len(preds)
        top_action = actions.most_common(1)[0][0]
        top_count = actions.most_common(1)[0][1]
        detail = ", ".join(f"{a}:{c}" for a, c in actions.most_common(3))
        print(f"  Player {pid:>2}: {len(preds):>3} clips | top={top_action}({top_count}) | {detail}")

    print(f"\n  Total clips: {total_clips}")
    print(f"  Global action distribution:")
    for action, count in global_actions.most_common():
        pct = count / total_clips * 100
        print(f"    {action:>14}: {count:>4} ({pct:5.1f}%)")

    # 6. Save JSON
    output = {
        "model_strategy": "pretrained",
        "total_clips": total_clips,
        "players": len(predictions),
        "global_actions": dict(global_actions),
        "per_player": {
            str(pid): {p.action: p.confidence for p in preds}
            for pid, preds in predictions.items()
        },
    }
    out_path = "/tmp/defense_pretrained_result.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}")

if __name__ == "__main__":
    main()
