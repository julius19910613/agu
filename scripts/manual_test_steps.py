"""Quick integration test: run analysis on adam.mp4 with CPU only (avoid MPS OOM)."""
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))

import torch
from app.config import Settings
from app.models.r2plus1d import build_r2plus1d_model
from app.analysis.service import AnalysisService
from app.analysis.schemas import AnalysisRequest

def main():
    settings = Settings()
    print(f"[Config] tracker_type={settings.tracker_type}")
    print(f"[Config] low_confidence={settings.low_confidence}, high_confidence={settings.high_confidence}")
    print(f"[Config] vid_stride={settings.vid_stride}, seq_length={settings.seq_length}")

    # Force CPU to avoid MPS memory pressure
    device = torch.device("cpu")
    print(f"[Device] {device} (forced CPU to conserve memory)")
    model = build_r2plus1d_model(settings=settings, device=device)
    print("[Model] R(2+1)D loaded")

    service = AnalysisService(settings=settings, model=model, device=device)

    request = AnalysisRequest(
        video_path="examples/adam.mp4",
        vlm_mode="off",
        generate_video=False,
    )
    print(f"\n[Request] video_path={request.video_path}")
    print(f"[Request] tracker_conf_thres={request.tracker_conf_thres}")
    print(f"[Request] tracker_iou_thres={request.tracker_iou_thres}")
    print(f"[Request] tracker_min_appear_ratio={request.tracker_min_appear_ratio}")
    print(f"[Request] tracker_min_appear_abs={request.tracker_min_appear_abs}")

    print("\n--- Starting analysis ---")
    started = time.time()
    try:
        result = service.run_analysis(request)
        elapsed = time.time() - started
        print(f"\n--- Analysis completed in {elapsed:.1f}s ---")
        print(f"Players detected: {len(set(r.player for r in result.records))}")
        print(f"Total records: {len(result.records)}")
        print(f"Frame size: {result.frame_size}")
        print(f"Vid stride: {result.vid_stride}")
        print(f"Seq length: {result.seq_length}")
        print(f"Runtime: {result.runtime_seconds:.2f}s")

        # Action distribution
        from collections import Counter
        action_counts = Counter(r.r2plus1d.action for r in result.records)
        print(f"\nAction distribution:")
        for action, count in action_counts.most_common():
            print(f"  {action}: {count}")

        # Player summary
        player_actions = {}
        for r in result.records:
            player_actions.setdefault(r.player, []).append(r.r2plus1d.action)
        print(f"\nPer-player actions:")
        for player, actions in sorted(player_actions.items()):
            counts = Counter(actions)
            print(f"  Player {player}: {dict(counts.most_common())}")

    except Exception as e:
        elapsed = time.time() - started
        print(f"\n--- Analysis FAILED after {elapsed:.1f}s ---")
        traceback.print_exc()

if __name__ == "__main__":
    main()
