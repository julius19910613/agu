"""Step-by-step test: test YOLO tracking only, then inference separately."""
import sys, os, time, traceback
sys.path.insert(0, os.path.dirname(__file__))

def test_tracking():
    """Test step 1: YOLO tracking."""
    from app.analysis.tracking import extract_tracked_frames
    
    print("=== Step 1: YOLO Tracking ===")
    started = time.time()
    try:
        video_frames, player_boxes, width, height, colors = extract_tracked_frames(
            video_path="examples/adam.mp4",
            tracker_type="YOLO",
            headless=True,
            boxes_file=None,
            max_frames=None,
            conf_thres=0.3,
            iou_thres=0.6,
            min_appear_ratio=0.02,
            min_appear_abs=5,
            device=None,  # auto-detect
        )
        elapsed = time.time() - started
        print(f"Tracking completed in {elapsed:.1f}s")
        print(f"Total frames: {len(video_frames)}")
        print(f"Frame size: {width}x{height}")
        print(f"Players detected: {len(colors)}")
        print(f"Boxes per frame (first 5): {[len(b) for b in player_boxes[:5]]}")
        return video_frames, player_boxes, width, height, colors
    except Exception as e:
        elapsed = time.time() - started
        print(f"Tracking FAILED after {elapsed:.1f}s")
        traceback.print_exc()
        return None, None, None, None, None

def test_cropping(video_frames, player_boxes, width, height):
    """Test step 2: Window cropping."""
    from app.analysis.tracking import crop_windows
    from app.config import Settings
    
    print("\n=== Step 2: Window Cropping ===")
    settings = Settings()
    started = time.time()
    try:
        player_clips = crop_windows(
            video_frames, player_boxes,
            seq_length=settings.seq_length,
            vid_stride=settings.vid_stride,
        )
        elapsed = time.time() - started
        print(f"Cropping completed in {elapsed:.1f}s")
        for player, clips in player_clips.items():
            print(f"  Player {player}: {len(clips)} clips")
        return player_clips
    except Exception as e:
        elapsed = time.time() - started
        print(f"Cropping FAILED after {elapsed:.1f}s")
        traceback.print_exc()
        return None

def test_inference(player_clips):
    """Test step 3: Model inference."""
    import torch
    from app.config import Settings
    from app.models.r2plus1d import build_r2plus1d_model
    from app.analysis.inference import predict_player_clips
    
    print("\n=== Step 3: Model Inference ===")
    settings = Settings()
    device = torch.device("cpu")
    model = build_r2plus1d_model(settings=settings, device=device)
    
    started = time.time()
    try:
        predictions = predict_player_clips(model, player_clips, device, settings.batch_size)
        elapsed = time.time() - started
        print(f"Inference completed in {elapsed:.1f}s")
        for player, clips in predictions.items():
            actions = [c.action for c in clips]
            from collections import Counter
            print(f"  Player {player}: {dict(Counter(actions))}")
        return predictions
    except Exception as e:
        elapsed = time.time() - started
        print(f"Inference FAILED after {elapsed:.1f}s")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    video_frames, player_boxes, w, h, colors = test_tracking()
    if video_frames is not None:
        player_clips = test_cropping(video_frames, player_boxes, w, h)
        if player_clips is not None:
            test_inference(player_clips)
