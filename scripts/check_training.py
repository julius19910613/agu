#!/usr/bin/env python3
"""Training monitor: parse history file and output epoch val accuracy results.

Usage:
    python3 scripts/check_training.py --history-path histories/history_r2plus1d_v3.txt

Output (JSON):
    {"completed_epochs": [1,2,3], "val_accuracies": [0.683,0.747,0.775], "total_completed": 3}
"""
import argparse
import json
import re


def parse_history(path: str) -> dict:
    """Parse the training history file for epoch-level val accuracy."""
    completed = []
    val_accs = []

    try:
        with open(path, "r") as f:
            text = f.read()
    except FileNotFoundError:
        return {"error": f"History file not found: {path}"}

    epoch_counter = 0
    for line in text.splitlines():
        m = re.search(r"val accuracy:\s+([\d.]+)", line)
        if m:
            epoch_counter += 1
            completed.append(epoch_counter)
            val_accs.append(float(m.group(1)))

    return {
        "completed_epochs": completed,
        "val_accuracies": val_accs,
        "total_completed": epoch_counter,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-path", required=True)
    args = parser.parse_args()
    result = parse_history(args.history_path)
    print(json.dumps(result, indent=2))
