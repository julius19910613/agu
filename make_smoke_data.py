#!/usr/bin/env python3
"""Quick smoke test for train_mac.py using synthetic data.

Creates a tiny dataset (100 videos) in /tmp to verify the training
loop works end-to-end before waiting for the real SpaceJam download.
"""
import os
import sys
import json
import numpy as np
import cv2

# Create tiny dataset
OUT_DIR = "/tmp/spacejam_smoke"
EXAMPLES_DIR = os.path.join(OUT_DIR, "examples")
AUG_DIR = os.path.join(OUT_DIR, "augmented-examples")
os.makedirs(EXAMPLES_DIR, exist_ok=True)
os.makedirs(AUG_DIR, exist_ok=True)

N_ORIG = 80
N_AUG = 20
LABELS = {0:"block",1:"pass",2:"run",3:"dribble",4:"shoot",5:"ball in hand",6:"defense",7:"pick",8:"no_action",9:"walk"}

# Generate synthetic 16-frame videos at 128x176 (matching SpaceJam)
print(f"Generating {N_ORIG} original + {N_AUG} augmented clips...")
annotation = {}
for i in range(N_ORIG):
    vid_id = f"{i:07d}"
    label = i % 10
    annotation[vid_id] = label
    # Write 16-frame mp4
    out_path = os.path.join(EXAMPLES_DIR, f"{vid_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, 10, (176, 128))
    for _ in range(16):
        frame = np.random.randint(0, 256, (128, 176, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

aug_annotation = {}
for i in range(N_AUG):
    vid_id = f"aug_{i:07d}"
    label = i % 10
    aug_annotation[vid_id] = label
    out_path = os.path.join(AUG_DIR, f"{vid_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, 10, (176, 128))
    for _ in range(16):
        frame = np.random.randint(0, 256, (128, 176, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

with open(os.path.join(OUT_DIR, "annotation_dict.json"), "w") as f:
    json.dump(annotation, f)
with open(os.path.join(OUT_DIR, "augmented_annotation_dict.json"), "w") as f:
    json.dump(aug_annotation, f)

print(f"Done — {N_ORIG + N_AUG} clips in {OUT_DIR}")
print(f"annotation_dict.json: {len(annotation)} entries")
print(f"augmented_annotation_dict.json: {len(aug_annotation)} entries")
