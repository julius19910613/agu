"""Generate stratified train/val/test splits and persist to JSON.

Usage:
    python scripts/gen_splits.py --annotation-path dataset/annotation_dict.json --output-dir dataset/splits
"""
import argparse
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


LABELS = {
    0: "block", 1: "pass", 2: "run", 3: "dribble", 4: "shoot",
    5: "ball_in_hand", 6: "defense", 7: "pick", 8: "no_action", 9: "walk",
}


def _extract_label(entry):
    if isinstance(entry, dict):
        if "label" in entry:
            return int(entry["label"])
        if "action" in entry:
            action = entry["action"]
            if isinstance(action, list) and len(action) == 10:
                return int(np.argmax(action))
            return int(action)
        if "class" in entry:
            return int(entry["class"])
        for value in entry.values():
            if isinstance(value, list) and len(value) == 10:
                return int(np.argmax(value))
    return int(entry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-path", default="dataset/annotation_dict.json")
    parser.add_argument("--augmented-path", default="dataset/augmented_annotation_dict.json")
    parser.add_argument("--output-dir", default="dataset/splits")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.annotation_path) as f:
        annotations = json.load(f)

    if isinstance(annotations, dict):
        annotations = list(annotations.values())

    indices = list(range(len(annotations)))
    labels = np.array([_extract_label(annotation) for annotation in annotations])
    indices = np.array(indices)

    os.makedirs(args.output_dir, exist_ok=True)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=args.test_ratio, random_state=args.seed)
    train_val_idx, test_idx = next(sss1.split(indices, labels))

    remaining_labels = labels[train_val_idx]
    relative_val_ratio = args.val_ratio / (1 - args.test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_ratio, random_state=args.seed)
    train_idx_rel, val_idx_rel = next(sss2.split(train_val_idx, remaining_labels))
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    splits = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
        "label_counts": {
            split_name: {LABELS[i]: int(np.sum(labels[idx_arr] == i)) for i in range(10)}
            for split_name, idx_arr in [("train", train_idx), ("val", val_idx), ("test", test_idx)]
        },
        "seed": args.seed,
    }

    output_path = os.path.join(args.output_dir, "stratified_split.json")
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Split saved to {output_path}")
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    for split_name in ["train", "val", "test"]:
        counts = splits["label_counts"][split_name]
        print(f"  {split_name} class distribution: {counts}")


if __name__ == "__main__":
    main()
