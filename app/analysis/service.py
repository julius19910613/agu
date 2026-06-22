from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional
from uuid import uuid4

import cv2
import numpy as np
import torch

from app.config import Settings
from app.analysis.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisRecordResponse,
    AnalysisSummaryResponse,
    EventCandidateResponse,
    LongVideoAnalysisResponse,
    LongVideoAuditSummaryResponse,
    LongVideoPlayerSummaryResponse,
    LongVideoSegmentResponse,
    IdentityDuplicateCandidateResponse,
    PlayerIdentityFeatureResponse,
    PlayerBoxScoreEstimateResponse,
    Size2D,
    VLMVideoAuditResponse,
)
from app.analysis.tracking import extract_tracked_frames, crop_windows
from app.analysis.identity_embedding import BaseIdentityEmbedder, build_identity_embedder
from app.analysis.inference import predict_player_clips
from app.analysis.motion import compute_motion_features
from app.analysis.vlm import OllamaVLMVerifier
from app.analysis.fusion import fuse_decision, should_call_vlm, apply_temporal_smoothing, summarize_records
from app.video.writer import write_annotated_video


LOGGER = logging.getLogger(__name__)


class AnalysisService:
    """Orchestrates the hybrid analysis pipeline."""

    def __init__(self, settings: Settings, model: torch.nn.Module, device: torch.device):
        self.settings = settings
        self.model = model
        self.device = device
        self._identity_embedder_key: Optional[tuple[str, str, str, int, bool]] = None
        self._identity_embedder: Optional[BaseIdentityEmbedder] = None
        try:
            torch_num_threads = int(self.settings.torch_num_threads)
        except (TypeError, ValueError):
            torch_num_threads = 0
        if torch_num_threads > 0:
            torch.set_num_threads(torch_num_threads)

    def _log_progress(self, message: str) -> None:
        if self.settings.progress_log:
            LOGGER.info(message)

    def _resolve_r2plus1d_device(self, request: AnalysisRequest) -> torch.device:
        preference = request.r2plus1d_device or self.settings.r2plus1d_device or "auto"
        if not isinstance(preference, str):
            preference = "auto"
        preference = preference.lower()
        if preference in {"auto", ""}:
            return self.device
        if preference == "mps_if_available":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        if preference == "mps" and not torch.backends.mps.is_available():
            return torch.device("cpu")
        if preference == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(preference)

    def _ensure_model_device(self, device: torch.device) -> torch.nn.Module:
        try:
            current_device = next(self.model.parameters()).device
        except StopIteration:
            current_device = device
        if current_device != device:
            self._log_progress(f"Moving R(2+1)D model from {current_device} to {device}.")
            self.model.to(device)
        self.model.eval()
        return self.model
        
    def run_analysis(self, request: AnalysisRequest) -> AnalysisResponse:
        """Run either the standard single-pass pipeline or long-video segmented pipeline."""
        if request.segmented_analysis or request.long_video_mode:
            return self.run_long_video_analysis(request)
        return self._run_single_analysis(request)

    def _run_single_analysis(self, request: AnalysisRequest, persist_output: bool = True) -> AnalysisResponse:
        """Run the full hybrid analysis pipeline blockingly."""
        started_at = time.time()
        effective_vid_stride = request.vid_stride if request.vid_stride is not None else self.settings.vid_stride
        effective_low_confidence = request.low_confidence if request.low_confidence is not None else self.settings.low_confidence
        effective_high_confidence = request.high_confidence if request.high_confidence is not None else self.settings.high_confidence
        effective_tracking_fps = request.tracking_fps if request.tracking_fps is not None else self.settings.tracking_fps
        effective_yolo_imgsz = request.yolo_imgsz if request.yolo_imgsz is not None else self.settings.yolo_imgsz
        effective_max_players = (
            request.max_players_per_segment
            if request.max_players_per_segment is not None
            else self.settings.max_players_per_segment
        )
        effective_yolo_device = request.yolo_device or self.settings.yolo_device
        effective_tracker_backend = request.tracker_backend or self.settings.tracker_backend
        effective_tracker_config = request.yolo_tracker_config or self.settings.yolo_tracker_config
        effective_reid_enabled = (
            request.yolo_reid_enabled
            if request.yolo_reid_enabled is not None
            else self.settings.yolo_reid_enabled
        )
        effective_reid_model = request.yolo_reid_model or self.settings.yolo_reid_model
        effective_identity_backend = request.identity_embedding_backend or self.settings.identity_embedding_backend
        effective_identity_weights = request.identity_embedding_weights or self.settings.identity_embedding_weights
        effective_identity_device = request.identity_embedding_device or self.settings.identity_embedding_device
        effective_jersey_number_vlm_enabled = (
            request.jersey_number_vlm_enabled
            if request.jersey_number_vlm_enabled is not None
            else getattr(self.settings, "jersey_number_vlm_enabled", False)
        )
        effective_jersey_number_vlm_frames = (
            request.jersey_number_vlm_frames
            if request.jersey_number_vlm_frames is not None
            else getattr(self.settings, "jersey_number_vlm_frames", 2)
        )
        inference_device = self._resolve_r2plus1d_device(request)
        model = self._ensure_model_device(inference_device)
        
        # 1. Video Tracking
        self._log_progress(
            "Starting tracking: "
            f"video={request.video_path}, yolo_device={effective_yolo_device}, "
            f"tracker_backend={effective_tracker_backend}, reid_enabled={effective_reid_enabled}, "
            f"tracking_fps={effective_tracking_fps}, yolo_imgsz={effective_yolo_imgsz}, "
            f"max_players={effective_max_players}."
        )
        video_frames, player_boxes, width, height, colors = extract_tracked_frames(
            video_path=request.video_path,
            tracker_type=self.settings.tracker_type,
            headless=True,
            boxes_file=request.boxes_file,
            max_frames=request.max_frames,
            conf_thres=request.tracker_conf_thres,
            iou_thres=request.tracker_iou_thres,
            min_appear_ratio=request.tracker_min_appear_ratio,
            min_appear_abs=request.tracker_min_appear_abs,
            device=effective_yolo_device,
            yolo_model_name=self.settings.yolo_model_name,
            tracker_backend=effective_tracker_backend,
            yolo_tracker_config=effective_tracker_config,
            reid_enabled=effective_reid_enabled,
            reid_model=effective_reid_model,
            tracking_fps=effective_tracking_fps,
            yolo_imgsz=effective_yolo_imgsz,
            max_players=effective_max_players,
        )

        # 2. Window Cropping
        self._log_progress(
            f"Cropping action windows: frames={len(video_frames)}, players={len(player_boxes[0]) if player_boxes else 0}, "
            f"seq_length={self.settings.seq_length}, vid_stride={effective_vid_stride}."
        )
        player_clips = crop_windows(
            video_frames,
            player_boxes,
            seq_length=self.settings.seq_length,
            vid_stride=effective_vid_stride,
        )
        
        # 3. Model Inference
        self._log_progress(f"Running R(2+1)D inference on {inference_device} with batch_size={self.settings.batch_size}.")
        predictions = predict_player_clips(
            model=model,
            player_clips=player_clips,
            device=inference_device,
            batch_size=self.settings.batch_size,
        )
        
        # 4. VLM Initialization
        verifier: Optional[OllamaVLMVerifier] = None
        if request.vlm_mode != "off":
            verifier = OllamaVLMVerifier(
                model=self.settings.ollama_model,
                host=self.settings.ollama_host,
                timeout=self.settings.ollama_timeout,
                image_width=self.settings.vlm_image_width,
            )
        jersey_number_verifier: Optional[OllamaVLMVerifier] = None
        if effective_jersey_number_vlm_enabled:
            jersey_number_verifier = OllamaVLMVerifier(
                model=self.settings.ollama_model,
                host=self.settings.ollama_host,
                timeout=self.settings.ollama_timeout,
                image_width=max(384, int(self.settings.vlm_image_width)),
            )

        # 5. Fusion & Verification
        output_records: List[Dict[str, Any]] = []
        final_prediction_ids: Dict[int, Dict[int, int]] = {}
        vlm_used_count = 0

        for player, player_predictions in predictions.items():
            final_prediction_ids[player] = {}
            for clip_index, prediction in enumerate(player_predictions):
                motion = compute_motion_features(
                    player_boxes,
                    player=player,
                    clip_index=clip_index,
                    seq_length=self.settings.seq_length,
                    vid_stride=effective_vid_stride,
                )
                
                vlm_decision = None
                if verifier and should_call_vlm(
                    request.vlm_mode,
                    prediction,
                    effective_low_confidence,
                    vlm_used_count,
                    self.settings.max_vlm_clips,
                ):
                    from app.analysis.vlm import select_keyframes
                    frames = select_keyframes(
                        player_clips[player][clip_index], 
                        max_frames=self.settings.vlm_frames
                    )
                    vlm_decision = verifier.verify(frames, prediction, motion)
                    vlm_used_count += 1

                final = fuse_decision(
                    prediction,
                    vlm_decision,
                    high_confidence=effective_high_confidence,
                    low_confidence=effective_low_confidence,
                )
                final_prediction_ids[player][clip_index] = final.action_id
                
                output_records.append({
                    "player": player,
                    "clip_index": clip_index,
                    "start_frame": clip_index * effective_vid_stride,
                    "end_frame": min(clip_index * effective_vid_stride + self.settings.seq_length - 1, len(video_frames) - 1),
                    "r2plus1d": prediction,
                    "motion": motion,
                    "vlm": vlm_decision,
                    "final": final,
                })

        # 6. Temporal Smoothing
        apply_temporal_smoothing(output_records, final_prediction_ids, self.settings.smoothing_confidence)
        
        # Build Response
        summary_dict = summarize_records(output_records)
        player_identity_features = self._extract_player_identity_features(
            video_frames=video_frames,
            player_boxes=player_boxes,
            frame_offset=0,
            embedding_backend=effective_identity_backend,
            embedding_weights=effective_identity_weights,
            embedding_device=effective_identity_device,
            jersey_number_verifier=jersey_number_verifier,
            jersey_number_frames=effective_jersey_number_vlm_frames,
        )
        identity_embedding_model = (
            player_identity_features[0].embedding_model
            if player_identity_features
            else effective_identity_backend
        )
        
        response = AnalysisResponse(
            video=request.video_path,
            created_at_unix=started_at,
            runtime_seconds=time.time() - started_at,
            frame_size=Size2D(width=width, height=height),
            seq_length=self.settings.seq_length,
            vid_stride=effective_vid_stride,
            tracker_backend=effective_tracker_backend,
            tracker_config=effective_tracker_config or ("botsort.yaml" if effective_tracker_backend == "botsort" else "bytetrack.yaml"),
            reid_enabled=bool(effective_reid_enabled),
            identity_embedding_backend=effective_identity_backend,
            identity_embedding_model=identity_embedding_model,
            vlm_mode=request.vlm_mode,
            ollama_model=self.settings.ollama_model if request.vlm_mode != "off" else None,
            records=[AnalysisRecordResponse(**r) for r in output_records],
            summary=AnalysisSummaryResponse(**summary_dict),
            player_identity_features=player_identity_features,
        )

        analysis_id = str(uuid4().hex)

        # 8. Video Generation (Write video first to avoid orphan JSON on failure)
        if request.generate_video:
            import cv2
            fps = 30.0
            cap = cv2.VideoCapture(request.video_path)
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open video for FPS extraction: {request.video_path}")
                val = cap.get(cv2.CAP_PROP_FPS)
                if val is not None and val > 0:
                    fps = val
            finally:
                cap.release()

            video_name = os.path.splitext(os.path.basename(request.video_path))[0]
            video_output_path = os.path.join(self.settings.video_output_dir, f"{video_name}_{analysis_id}.mp4")
            write_annotated_video(
                video_path=video_output_path,
                video_frames=video_frames,
                player_boxes=player_boxes,
                predictions=final_prediction_ids,
                colors=colors,
                frame_width=width,
                frame_height=height,
                vid_stride=effective_vid_stride,
                fps=fps,
            )

        # 7. Persistence
        if persist_output:
            os.makedirs(self.settings.output_dir, exist_ok=True)
            json_path = os.path.join(self.settings.output_dir, f"{analysis_id}.json")
            with open(json_path, "w") as fp:
                fp.write(response.model_dump_json(indent=2))

        return response

    def run_long_video_analysis(self, request: AnalysisRequest) -> AnalysisResponse:
        """Run segmented analysis and VLM contact-sheet audit for long videos."""
        started_at = time.time()
        metadata = self._read_video_metadata(request.video_path)
        fps = metadata["fps"]
        frame_count = metadata["frame_count"]
        width = metadata["width"]
        height = metadata["height"]
        duration_sec = metadata["duration_sec"]

        segment_duration = max(1.0, float(request.segment_duration_sec))
        segment_overlap = max(0.0, min(float(request.segment_overlap_sec), segment_duration - 0.1))
        segment_ranges = self._build_segment_ranges(
            duration_sec=duration_sec,
            fps=fps,
            frame_count=frame_count,
            segment_duration_sec=segment_duration,
            segment_overlap_sec=segment_overlap,
            segment_start_sec=request.segment_start_sec,
            segment_end_sec=request.segment_end_sec,
            max_segments=request.max_segments,
        )
        effective_vid_stride = (
            request.vid_stride
            if request.vid_stride is not None
            else (
                request.action_vid_stride
                if request.action_vid_stride is not None
                else self.settings.action_vid_stride
            )
        )
        effective_tracker_backend = request.tracker_backend or self.settings.tracker_backend
        effective_tracker_config = request.yolo_tracker_config or self.settings.yolo_tracker_config
        effective_reid_enabled = (
            request.yolo_reid_enabled
            if request.yolo_reid_enabled is not None
            else self.settings.yolo_reid_enabled
        )
        self._log_progress(
            "Starting segmented analysis: "
            f"segments={len(segment_ranges)}, duration={duration_sec:.2f}s, "
            f"segment_duration={segment_duration:.2f}s, overlap={segment_overlap:.2f}s, "
            f"action_vid_stride={effective_vid_stride}."
        )

        verifier: Optional[OllamaVLMVerifier] = None
        if request.vlm_audit:
            verifier = OllamaVLMVerifier(
                model=self.settings.ollama_model,
                host=self.settings.ollama_host,
                timeout=self.settings.ollama_timeout,
                image_width=self.settings.vlm_image_width,
            )

        merged_records: List[AnalysisRecordResponse] = []
        segment_outputs: List[LongVideoSegmentResponse] = []
        player_actions: Dict[str, Counter[str]] = defaultdict(Counter)
        player_confidences: Dict[str, List[float]] = defaultdict(list)
        player_reviews: Counter[str] = Counter()
        player_segments: Dict[str, set[int]] = defaultdict(set)
        player_identity_features: Dict[str, PlayerIdentityFeatureResponse] = {}
        status_counts: Counter[str] = Counter()

        for segment_index, segment in enumerate(segment_ranges):
            self._log_progress(
                f"Segment {segment_index + 1}/{len(segment_ranges)}: "
                f"{segment['start_sec']:.2f}s-{segment['end_sec']:.2f}s."
            )
            temp_path = self._write_video_segment(
                request.video_path,
                start_frame=segment["start_frame"],
                end_frame=segment["end_frame"],
                fps=fps,
                width=width,
                height=height,
            )
            try:
                segment_request = request.model_copy(
                    update={
                        "video_path": temp_path,
                        "long_video_mode": False,
                        "segmented_analysis": False,
                        "generate_video": False,
                        "max_frames": None,
                        "vlm_mode": "off",
                        "vid_stride": effective_vid_stride,
                    }
                )
                segment_result = self._run_single_analysis(segment_request, persist_output=False)
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            adjusted_records = [
                record.model_copy(
                    update={
                        "clip_index": len(merged_records) + index,
                        "start_frame": int(record.start_frame) + segment["start_frame"],
                        "end_frame": int(record.end_frame) + segment["start_frame"],
                        "segment_id": segment["segment_id"],
                        "local_player_id": f"segment_{segment['segment_id']}:player_{record.player}",
                    }
                )
                for index, record in enumerate(segment_result.records)
            ]
            owned_records = self._owned_records_for_segment(
                records=adjusted_records,
                segment=segment,
                next_segment=segment_ranges[segment_index + 1] if segment_index + 1 < len(segment_ranges) else None,
            )
            merged_records.extend(owned_records)

            for record in owned_records:
                player_key = f"segment_{segment['segment_id']}:player_{record.player}"
                player_actions[player_key][record.final.action] += 1
                player_confidences[player_key].append(float(record.final.confidence))
                player_reviews[player_key] += int(record.final.needs_review)
                player_segments[player_key].add(segment["segment_id"])

            for feature in segment_result.player_identity_features:
                local_player_id = f"segment_{segment['segment_id']}:player_{feature.player}"
                player_identity_features[local_player_id] = feature.model_copy(
                    update={
                        "segment_id": segment["segment_id"],
                        "local_player_id": local_player_id,
                        "start_frame": int(feature.start_frame) + segment["start_frame"],
                        "end_frame": int(feature.end_frame) + segment["start_frame"],
                    }
                )

            player_count = len({record.player for record in segment_result.records})
            vlm_audit = None
            if verifier is not None:
                audit_frames = self._sample_contact_sheet_frames(
                    request.video_path,
                    start_frame=segment["start_frame"],
                    end_frame=segment["end_frame"],
                    sample_count=request.vlm_audit_frames,
                )
                contact_sheet = self._make_contact_sheet(audit_frames)
                scope = f"{segment['start_sec']:.1f}s-{segment['end_sec']:.1f}s"
                vlm_audit = verifier.audit_video_frames([contact_sheet], scope=scope)

            audit_status, audit_notes = self._compare_segment_with_vlm(
                player_count=player_count,
                summary=segment_result.summary,
                vlm_audit=vlm_audit,
            )
            status_counts[audit_status] += 1
            self._log_progress(
                f"Segment {segment_index + 1}/{len(segment_ranges)} complete: "
                f"players={player_count}, clips={segment_result.summary.clip_count}, audit={audit_status}."
            )

            segment_outputs.append(
                LongVideoSegmentResponse(
                    segment_id=segment["segment_id"],
                    start_sec=segment["start_sec"],
                    end_sec=segment["end_sec"],
                    start_frame=segment["start_frame"],
                    end_frame=segment["end_frame"],
                    player_count=player_count,
                    summary=segment_result.summary,
                    vlm_audit=vlm_audit,
                    audit_status=audit_status,
                    audit_notes=audit_notes,
                )
            )

        global_summary = self._summarize_response_records(merged_records)
        player_summaries = [
            LongVideoPlayerSummaryResponse(
                player_id=player_id,
                segments_seen=len(player_segments[player_id]),
                clip_count=sum(actions.values()),
                action_counts=dict(actions),
                needs_review_count=int(player_reviews[player_id]),
                average_confidence=(
                    sum(player_confidences[player_id]) / len(player_confidences[player_id])
                    if player_confidences[player_id]
                    else 0.0
                ),
                statistics=self._estimate_player_statistics(actions),
            )
            for player_id, actions in sorted(player_actions.items())
        ]
        identity_map, identity_confidences, identity_evidence = self._merge_segment_local_identities(
            player_summaries,
            player_identity_features,
        )
        player_summaries = [
            summary.model_copy(
                update={
                    "global_player_id": identity_map.get(summary.player_id),
                    "identity_confidence": identity_confidences.get(summary.player_id, 0.0),
                    "identity_method": "appearance_continuity_stitch_v2",
                    "identity_evidence": identity_evidence.get(summary.player_id, []),
                }
            )
            for summary in player_summaries
        ]
        identity_duplicate_candidates = self._detect_identity_duplicate_candidates(
            player_summaries,
            player_identity_features,
        )
        merged_records = [
            record.model_copy(
                update={
                    "global_player_id": identity_map.get(record.local_player_id or ""),
                    "identity_confidence": identity_confidences.get(record.local_player_id or "", 0.0),
                }
            )
            for record in merged_records
        ]
        event_candidates = self._detect_event_candidates(merged_records)

        audit_summary = LongVideoAuditSummaryResponse(
            total_segments=len(segment_outputs),
            passed=sum(count for status, count in status_counts.items() if status == "pass"),
            warnings=sum(count for status, count in status_counts.items() if status.startswith("warn")),
            failed=sum(count for status, count in status_counts.items() if status.startswith("fail")),
            status_counts=dict(status_counts),
        )

        response = AnalysisResponse(
            video=request.video_path,
            created_at_unix=started_at,
            runtime_seconds=time.time() - started_at,
            frame_size=Size2D(width=width, height=height),
            seq_length=self.settings.seq_length,
            vid_stride=effective_vid_stride,
            tracker_backend=effective_tracker_backend,
            tracker_config=effective_tracker_config or ("botsort.yaml" if effective_tracker_backend == "botsort" else "bytetrack.yaml"),
            reid_enabled=bool(effective_reid_enabled),
            identity_embedding_backend=request.identity_embedding_backend or self.settings.identity_embedding_backend,
            identity_embedding_model=(
                next(iter(player_identity_features.values())).embedding_model
                if player_identity_features
                else (request.identity_embedding_backend or self.settings.identity_embedding_backend)
            ),
            vlm_mode=request.vlm_mode,
            ollama_model=self.settings.ollama_model if request.vlm_audit else None,
            records=merged_records,
            summary=global_summary,
            player_identity_features=list(player_identity_features.values()),
            long_video=LongVideoAnalysisResponse(
                duration_sec=duration_sec,
                fps=fps,
                frame_count=frame_count,
                segment_duration_sec=segment_duration,
                segment_overlap_sec=segment_overlap,
                segments=segment_outputs,
                players=player_summaries,
                event_candidates=event_candidates,
                identity_duplicate_candidates=identity_duplicate_candidates,
                audit_summary=audit_summary,
            ),
        )

        os.makedirs(self.settings.output_dir, exist_ok=True)
        json_path = os.path.join(self.settings.output_dir, f"{uuid4().hex}.json")
        with open(json_path, "w") as fp:
            fp.write(response.model_dump_json(indent=2))

        return response

    def _owned_records_for_segment(
        self,
        records: List[AnalysisRecordResponse],
        segment: Dict[str, Any],
        next_segment: Optional[Dict[str, Any]],
    ) -> List[AnalysisRecordResponse]:
        """Keep only records owned by this segment to avoid overlap double-counting."""
        owned_start = int(segment["start_frame"])
        owned_end = int(next_segment["start_frame"]) - 1 if next_segment else int(segment["end_frame"])
        owned: List[AnalysisRecordResponse] = []
        for record in records:
            center = (int(record.start_frame) + int(record.end_frame)) // 2
            if owned_start <= center <= owned_end:
                owned.append(record)
        return owned

    def _read_video_metadata(self, video_path: str) -> Dict[str, float]:
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            cap.release()
        duration_sec = frame_count / fps if fps > 0 else 0.0
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": duration_sec,
        }

    def _build_segment_ranges(
        self,
        duration_sec: float,
        fps: float,
        frame_count: int,
        segment_duration_sec: float,
        segment_overlap_sec: float,
        segment_start_sec: float = 0.0,
        segment_end_sec: Optional[float] = None,
        max_segments: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        ranges: List[Dict[str, Any]] = []
        step = max(0.1, segment_duration_sec - segment_overlap_sec)
        start_sec = max(0.0, min(float(segment_start_sec), duration_sec))
        stop_sec = min(duration_sec, float(segment_end_sec)) if segment_end_sec is not None else duration_sec
        while start_sec < stop_sec and (max_segments is None or len(ranges) < max_segments):
            end_sec = min(stop_sec, start_sec + segment_duration_sec)
            start_frame = min(frame_count - 1, max(0, int(round(start_sec * fps))))
            end_frame = min(frame_count - 1, max(start_frame, int(round(end_sec * fps)) - 1))
            ranges.append(
                {
                    "segment_id": len(ranges),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            )
            if end_sec >= stop_sec:
                break
            start_sec += step
        return ranges

    def _extract_player_identity_features(
        self,
        video_frames: List[np.ndarray],
        player_boxes: List[tuple],
        frame_offset: int = 0,
        embedding_backend: Optional[str] = None,
        embedding_weights: Optional[str] = None,
        embedding_device: Optional[str] = None,
        jersey_number_verifier: Optional[OllamaVLMVerifier] = None,
        jersey_number_frames: int = 2,
    ) -> List[PlayerIdentityFeatureResponse]:
        """Extract lightweight appearance and continuity features for player tracks."""
        if not video_frames or not player_boxes:
            return []
        embedder = self._get_identity_embedder(
            backend=embedding_backend,
            weights=embedding_weights,
            device=embedding_device,
        )
        player_count = len(player_boxes[0]) if player_boxes[0] else 0
        if player_count == 0:
            return []

        features: List[PlayerIdentityFeatureResponse] = []
        sample_stride = max(1, len(video_frames) // 12)
        for player in range(player_count):
            means: List[np.ndarray] = []
            crops: List[np.ndarray] = []
            centers: List[List[float]] = []
            sampled_boxes: List[Dict[str, float]] = []
            valid_frames = 0
            first_frame = frame_offset
            last_frame = frame_offset + len(video_frames) - 1
            for frame_index in range(0, len(video_frames), sample_stride):
                if player >= len(player_boxes[frame_index]):
                    continue
                box = player_boxes[frame_index][player]
                x, y, w, h = [float(value) for value in box]
                if w <= 1 or h <= 1:
                    continue
                frame = video_frames[frame_index]
                height, width = frame.shape[:2]
                x1 = max(0, min(width - 1, int(round(x))))
                y1 = max(0, min(height - 1, int(round(y))))
                x2 = max(x1 + 1, min(width, int(round(x + w))))
                y2 = max(y1 + 1, min(height, int(round(y + h))))
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_summary = cv2.resize(crop, (16, 16), interpolation=cv2.INTER_AREA)
                hsv = cv2.cvtColor(crop_summary, cv2.COLOR_BGR2HSV)
                bgr_mean = crop_summary.reshape(-1, 3).mean(axis=0)
                hsv_mean = hsv.reshape(-1, 3).mean(axis=0)
                crops.append(crop)
                sampled_boxes.append(
                    {
                        "frame": float(frame_offset + frame_index),
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "cx": x + w / 2.0,
                        "cy": y + h / 2.0,
                    }
                )
                means.append(
                    np.array(
                        [
                            float(hsv_mean[0]) / 179.0,
                            float(hsv_mean[1]) / 255.0,
                            float(hsv_mean[2]) / 255.0,
                            float(bgr_mean[0]) / 255.0,
                            float(bgr_mean[1]) / 255.0,
                            float(bgr_mean[2]) / 255.0,
                        ],
                        dtype=np.float32,
                    )
                )
                centers.append([x + w / 2.0, y + h / 2.0])
                valid_frames += 1

            if not means:
                continue
            signature = np.stack(means, axis=0).mean(axis=0)
            embedding_result = embedder.embed_crops(crops)
            embedding = embedding_result.embedding
            jersey_number_candidates = []
            if jersey_number_verifier is not None:
                jersey_frames = self._select_jersey_number_frames(crops, jersey_number_frames)
                jersey_number_candidates = jersey_number_verifier.read_jersey_number(
                    jersey_frames,
                    scope=f"player_{player}",
                )
            first_center = centers[0] if centers else []
            last_center = centers[-1] if centers else []
            features.append(
                PlayerIdentityFeatureResponse(
                    player=player,
                    start_frame=first_frame,
                    end_frame=last_frame,
                    first_center=first_center,
                    last_center=last_center,
                    appearance_signature={
                        "h_mean": float(signature[0]),
                        "s_mean": float(signature[1]),
                        "v_mean": float(signature[2]),
                        "b_mean": float(signature[3]),
                        "g_mean": float(signature[4]),
                        "r_mean": float(signature[5]),
                    },
                    appearance_embedding=[float(value) for value in embedding.tolist()],
                    embedding_model=embedding_result.model_id,
                    embedding_dim=int(embedding.shape[0]),
                    track_coverage=valid_frames / max(1, (len(video_frames) + sample_stride - 1) // sample_stride),
                    method=embedding_result.method,
                    sampled_boxes=sampled_boxes,
                    jersey_number_candidates=jersey_number_candidates,
                )
            )
        return features

    def _select_jersey_number_frames(
        self,
        crops: List[np.ndarray],
        max_frames: int,
    ) -> List[np.ndarray]:
        if not crops:
            return []
        limit = max(1, int(max_frames or 1))
        sorted_crops = sorted(crops, key=lambda crop: crop.shape[0] * crop.shape[1], reverse=True)
        return sorted_crops[:limit]

    def _get_identity_embedder(
        self,
        backend: Optional[str] = None,
        weights: Optional[str] = None,
        device: Optional[str] = None,
    ) -> BaseIdentityEmbedder:
        effective_backend = backend or getattr(self.settings, "identity_embedding_backend", "torchvision_mobilenet_v3_small")
        effective_weights = weights or getattr(self.settings, "identity_embedding_weights", "default")
        effective_device = device or getattr(self.settings, "identity_embedding_device", "mps_if_available")
        effective_batch_size = int(getattr(self.settings, "identity_embedding_batch_size", 16) or 16)
        allow_fallback = bool(getattr(self.settings, "identity_embedding_allow_fallback", True))
        key = (
            str(effective_backend),
            str(effective_weights),
            str(effective_device),
            effective_batch_size,
            allow_fallback,
        )
        if self._identity_embedder is None or self._identity_embedder_key != key:
            self._identity_embedder = build_identity_embedder(
                backend=str(effective_backend),
                weights=str(effective_weights),
                device=str(effective_device),
                batch_size=effective_batch_size,
                allow_fallback=allow_fallback,
            )
            self._identity_embedder_key = key
        return self._identity_embedder

    def _crop_identity_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Generate a sidecar appearance embedding from a player crop.

        This is a local, dependency-light placeholder for true ReID embeddings.
        It uses normalized HSV histograms so the stitching path can consume a
        vector embedding today and later swap in model-generated ReID vectors.
        """
        from app.analysis.identity_embedding import SidecarHsvHistogramEmbedder

        return SidecarHsvHistogramEmbedder().embed_crops([crop_bgr]).embedding

    def _parse_segment_player_id(self, player_id: str) -> tuple[int, int]:
        try:
            segment_part, player_part = player_id.split(":")
            return int(segment_part.replace("segment_", "")), int(player_part.replace("player_", ""))
        except (ValueError, AttributeError):
            return -1, -1

    def _action_similarity(self, left: Dict[str, int], right: Dict[str, int]) -> float:
        actions = set(left) | set(right)
        if not actions:
            return 0.0
        dot = sum(float(left.get(action, 0)) * float(right.get(action, 0)) for action in actions)
        left_norm = sum(float(count) ** 2 for count in left.values()) ** 0.5
        right_norm = sum(float(count) ** 2 for count in right.values()) ** 0.5
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _merge_segment_local_identities(
        self,
        player_summaries: List[LongVideoPlayerSummaryResponse],
        player_identity_features: Optional[Dict[str, PlayerIdentityFeatureResponse]] = None,
    ) -> tuple[Dict[str, str], Dict[str, float], Dict[str, List[str]]]:
        """Create conservative global player IDs from adjacent segment-local summaries.

        This is intentionally a lightweight open-source-friendly baseline. When
        tracker features are available it uses appearance and track-continuity
        evidence; otherwise it falls back to action/local-index continuity.
        """
        player_identity_features = player_identity_features or {}
        identity_map: Dict[str, str] = {}
        identity_confidences: Dict[str, float] = {}
        identity_evidence: Dict[str, List[str]] = {}
        last_by_global_id: Dict[str, tuple[int, str, Dict[str, int]]] = {}
        next_global_index = 0

        sorted_summaries = sorted(
            player_summaries,
            key=lambda summary: self._parse_segment_player_id(summary.player_id),
        )
        for summary in sorted_summaries:
            segment_id, local_index = self._parse_segment_player_id(summary.player_id)
            best_candidate: Optional[tuple[float, str, str, float, float, float, float]] = None
            for global_player_id, (previous_segment, previous_player_id, previous_actions) in last_by_global_id.items():
                if segment_id <= previous_segment or segment_id - previous_segment > 1:
                    continue
                previous_local_index = self._parse_segment_player_id(previous_player_id)[1]
                action_similarity = self._action_similarity(previous_actions, summary.action_counts)
                appearance_similarity = self._appearance_similarity(
                    player_identity_features.get(previous_player_id),
                    player_identity_features.get(summary.player_id),
                )
                continuity_similarity = self._track_continuity_similarity(
                    player_identity_features.get(previous_player_id),
                    player_identity_features.get(summary.player_id),
                )
                local_index_bonus = 1.0 if previous_local_index == local_index else 0.0
                has_identity_features = (
                    previous_player_id in player_identity_features
                    and summary.player_id in player_identity_features
                )
                if has_identity_features:
                    score = (
                        action_similarity * 0.25
                        + appearance_similarity * 0.35
                        + continuity_similarity * 0.30
                        + local_index_bonus * 0.10
                    )
                else:
                    score = action_similarity * 0.75 + local_index_bonus * 0.25
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (
                        score,
                        global_player_id,
                        previous_player_id,
                        action_similarity,
                        appearance_similarity,
                        continuity_similarity,
                        local_index_bonus,
                    )

            if best_candidate is not None and best_candidate[0] >= 0.45:
                score, global_player_id, previous_player_id, action_similarity, appearance_similarity, continuity_similarity, local_index_bonus = best_candidate
                confidence = min(0.90, max(0.45, score))
                identity_map[summary.player_id] = global_player_id
                identity_confidences[summary.player_id] = confidence
                identity_evidence[summary.player_id] = [
                    f"stitched from {previous_player_id}",
                    f"combined identity score {score:.2f}",
                    f"embedding similarity {appearance_similarity:.2f}",
                    f"track continuity {continuity_similarity:.2f}",
                    f"action similarity {action_similarity:.2f}",
                    f"same local index bonus {local_index_bonus:.0f}",
                ]
                last_by_global_id[global_player_id] = (segment_id, summary.player_id, summary.action_counts)
                continue

            global_player_id = f"player_{next_global_index:03d}"
            next_global_index += 1
            identity_map[summary.player_id] = global_player_id
            identity_confidences[summary.player_id] = 0.25
            identity_evidence[summary.player_id] = [
                "new segment-local track; no reliable appearance/continuity stitch evidence",
            ]
            last_by_global_id[global_player_id] = (segment_id, summary.player_id, summary.action_counts)

        return identity_map, identity_confidences, identity_evidence

    def _appearance_similarity(
        self,
        left: Optional[PlayerIdentityFeatureResponse],
        right: Optional[PlayerIdentityFeatureResponse],
    ) -> float:
        if left is None or right is None:
            return 0.0
        if left.appearance_embedding and right.appearance_embedding:
            left_values = np.array(left.appearance_embedding, dtype=np.float32)
            right_values = np.array(right.appearance_embedding, dtype=np.float32)
            if left_values.shape == right_values.shape and left_values.size > 0:
                denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
                if denominator > 0.0:
                    return max(0.0, min(1.0, float(np.dot(left_values, right_values) / denominator)))
        keys = ["h_mean", "s_mean", "v_mean", "b_mean", "g_mean", "r_mean"]
        left_values = np.array([left.appearance_signature.get(key, 0.0) for key in keys], dtype=np.float32)
        right_values = np.array([right.appearance_signature.get(key, 0.0) for key in keys], dtype=np.float32)
        distance = float(np.linalg.norm(left_values - right_values))
        return max(0.0, min(1.0, 1.0 - distance / 1.75))

    def _track_continuity_similarity(
        self,
        left: Optional[PlayerIdentityFeatureResponse],
        right: Optional[PlayerIdentityFeatureResponse],
    ) -> float:
        if left is None or right is None or not left.last_center or not right.first_center:
            return 0.0
        dx = float(left.last_center[0]) - float(right.first_center[0])
        dy = float(left.last_center[1]) - float(right.first_center[1])
        distance = (dx * dx + dy * dy) ** 0.5
        return max(0.0, min(1.0, 1.0 - distance / 900.0))

    def _detect_identity_duplicate_candidates(
        self,
        player_summaries: List[LongVideoPlayerSummaryResponse],
        player_identity_features: Dict[str, PlayerIdentityFeatureResponse],
    ) -> List[IdentityDuplicateCandidateResponse]:
        """Find likely duplicate global IDs without mutating statistics.

        This conservative P0 pass suggests review-only merge candidates. It
        uses hard same-segment conflicts to avoid unsafe merges, then ranks
        remaining pairs by appearance embedding, color signature, action
        similarity, and temporal compatibility.
        """
        groups: Dict[str, Dict[str, Any]] = {}
        for summary in player_summaries:
            if not summary.global_player_id:
                continue
            segment_id, _ = self._parse_segment_player_id(summary.player_id)
            group = groups.setdefault(
                summary.global_player_id,
                {"local_ids": [], "segments": set(), "actions": Counter()},
            )
            group["local_ids"].append(summary.player_id)
            if segment_id >= 0:
                group["segments"].add(segment_id)
            group["actions"].update(summary.action_counts)

        candidates: List[IdentityDuplicateCandidateResponse] = []
        global_ids = sorted(groups)
        for index, left_gid in enumerate(global_ids):
            left_group = groups[left_gid]
            for right_gid in global_ids[index + 1 :]:
                right_group = groups[right_gid]
                conflict_evidence, overlap_similarity = self._identity_overlap_evidence(
                    left_group["local_ids"],
                    right_group["local_ids"],
                    player_identity_features,
                    left_group["segments"],
                    right_group["segments"],
                )
                if conflict_evidence:
                    continue

                appearance_similarity = self._group_appearance_similarity(
                    left_group["local_ids"],
                    right_group["local_ids"],
                    player_identity_features,
                )
                team_color_similarity = self._group_team_color_similarity(
                    left_group["local_ids"],
                    right_group["local_ids"],
                    player_identity_features,
                )
                action_similarity = self._action_similarity(left_group["actions"], right_group["actions"])
                temporal_similarity = self._segment_temporal_similarity(
                    left_group["segments"],
                    right_group["segments"],
                )
                score = (
                    appearance_similarity * 0.50
                    + team_color_similarity * 0.20
                    + action_similarity * 0.15
                    + temporal_similarity * 0.10
                    + overlap_similarity * 0.05
                )
                if score < 0.68:
                    continue

                candidates.append(
                    IdentityDuplicateCandidateResponse(
                        left_global_player_id=left_gid,
                        right_global_player_id=right_gid,
                        confidence=min(0.95, max(0.0, score)),
                        left_local_player_ids=sorted(left_group["local_ids"]),
                        right_local_player_ids=sorted(right_group["local_ids"]),
                        evidence=[
                            f"appearance embedding similarity {appearance_similarity:.2f}",
                            f"team color similarity {team_color_similarity:.2f}",
                            f"action similarity {action_similarity:.2f}",
                            f"temporal compatibility {temporal_similarity:.2f}",
                            f"bbox duplicate-overlap compatibility {overlap_similarity:.2f}",
                            "no frame-level hard conflict",
                            f"left local tracks {len(left_group['local_ids'])}",
                            f"right local tracks {len(right_group['local_ids'])}",
                        ],
                    )
                )

        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates[:100]

    def _group_appearance_similarity(
        self,
        left_local_ids: List[str],
        right_local_ids: List[str],
        player_identity_features: Dict[str, PlayerIdentityFeatureResponse],
    ) -> float:
        scores: List[float] = []
        for left_id in left_local_ids:
            left_feature = player_identity_features.get(left_id)
            if left_feature is None:
                continue
            for right_id in right_local_ids:
                right_feature = player_identity_features.get(right_id)
                if right_feature is None:
                    continue
                scores.append(self._appearance_similarity(left_feature, right_feature))
        if not scores:
            return 0.0
        scores.sort(reverse=True)
        top_scores = scores[: min(5, len(scores))]
        return sum(top_scores) / len(top_scores)

    def _group_team_color_similarity(
        self,
        left_local_ids: List[str],
        right_local_ids: List[str],
        player_identity_features: Dict[str, PlayerIdentityFeatureResponse],
    ) -> float:
        scores: List[float] = []
        for left_id in left_local_ids:
            left_feature = player_identity_features.get(left_id)
            if left_feature is None:
                continue
            for right_id in right_local_ids:
                right_feature = player_identity_features.get(right_id)
                if right_feature is None:
                    continue
                scores.append(self._signature_similarity(left_feature, right_feature))
        if not scores:
            return 0.0
        scores.sort(reverse=True)
        top_scores = scores[: min(5, len(scores))]
        return sum(top_scores) / len(top_scores)

    def _signature_similarity(
        self,
        left: PlayerIdentityFeatureResponse,
        right: PlayerIdentityFeatureResponse,
    ) -> float:
        keys = ["h_mean", "s_mean", "v_mean", "b_mean", "g_mean", "r_mean"]
        left_values = np.array([left.appearance_signature.get(key, 0.0) for key in keys], dtype=np.float32)
        right_values = np.array([right.appearance_signature.get(key, 0.0) for key in keys], dtype=np.float32)
        distance = float(np.linalg.norm(left_values - right_values))
        return max(0.0, min(1.0, 1.0 - distance / 1.75))

    def _segment_temporal_similarity(self, left_segments: set[int], right_segments: set[int]) -> float:
        if not left_segments or not right_segments:
            return 0.35
        min_gap = min(abs(left - right) for left in left_segments for right in right_segments)
        if min_gap <= 1:
            return 1.0
        if min_gap <= 3:
            return 0.85
        if min_gap <= 6:
            return 0.70
        if min_gap <= 12:
            return 0.55
        return 0.40

    def _identity_overlap_evidence(
        self,
        left_local_ids: List[str],
        right_local_ids: List[str],
        player_identity_features: Dict[str, PlayerIdentityFeatureResponse],
        left_segments: set[int],
        right_segments: set[int],
    ) -> tuple[List[str], float]:
        """Return hard conflict evidence and duplicate-overlap compatibility.

        If same-frame sampled boxes are far apart, the pair is a hard conflict.
        If same-frame boxes strongly overlap, they are likely duplicate detector
        boxes and should not block a merge-review candidate.
        """
        shared_segments = sorted(left_segments & right_segments)
        left_boxes = self._sampled_boxes_for_local_ids(left_local_ids, player_identity_features)
        right_boxes = self._sampled_boxes_for_local_ids(right_local_ids, player_identity_features)
        if not left_boxes or not right_boxes:
            if shared_segments:
                return (
                    [
                        "hard conflict: global IDs appear in same segment(s) without frame-level boxes "
                        + ", ".join(str(segment) for segment in shared_segments[:8])
                    ],
                    0.0,
                )
            return [], 0.35

        right_by_frame: Dict[int, List[Dict[str, float]]] = defaultdict(list)
        for box in right_boxes:
            right_by_frame[int(round(float(box.get("frame", -1))))].append(box)

        same_frame_ious: List[float] = []
        for left_box in left_boxes:
            frame = int(round(float(left_box.get("frame", -1))))
            for right_box in right_by_frame.get(frame, []):
                iou = self._box_iou(left_box, right_box)
                same_frame_ious.append(iou)
                if iou < 0.20:
                    return [f"hard conflict: same frame {frame} has separated boxes iou={iou:.2f}"], 0.0

        if same_frame_ious:
            return [], max(same_frame_ious)
        if shared_segments:
            return [], 0.25
        return [], 0.35

    def _sampled_boxes_for_local_ids(
        self,
        local_ids: List[str],
        player_identity_features: Dict[str, PlayerIdentityFeatureResponse],
    ) -> List[Dict[str, float]]:
        boxes: List[Dict[str, float]] = []
        for local_id in local_ids:
            feature = player_identity_features.get(local_id)
            if feature is None:
                continue
            boxes.extend(feature.sampled_boxes)
        return boxes

    def _box_iou(self, left: Dict[str, float], right: Dict[str, float]) -> float:
        left_x1 = float(left.get("x", 0.0))
        left_y1 = float(left.get("y", 0.0))
        left_x2 = left_x1 + max(0.0, float(left.get("w", 0.0)))
        left_y2 = left_y1 + max(0.0, float(left.get("h", 0.0)))
        right_x1 = float(right.get("x", 0.0))
        right_y1 = float(right.get("y", 0.0))
        right_x2 = right_x1 + max(0.0, float(right.get("w", 0.0)))
        right_y2 = right_y1 + max(0.0, float(right.get("h", 0.0)))
        inter_x1 = max(left_x1, right_x1)
        inter_y1 = max(left_y1, right_y1)
        inter_x2 = min(left_x2, right_x2)
        inter_y2 = min(left_y2, right_y2)
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
        right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
        union = left_area + right_area - inter_area
        if union <= 0.0:
            return 0.0
        return max(0.0, min(1.0, inter_area / union))

    def _detect_event_candidates(
        self,
        records: List[AnalysisRecordResponse],
    ) -> List[EventCandidateResponse]:
        """Detect low/medium confidence basketball event candidates from action records."""
        sorted_records = sorted(records, key=lambda record: (int(record.start_frame), int(record.player)))
        candidates: List[EventCandidateResponse] = []
        candidates.extend(self._detect_block_candidates(sorted_records))
        candidates.extend(self._detect_rebound_candidates(sorted_records))
        candidates.extend(self._detect_steal_candidates(sorted_records))
        candidates.sort(key=lambda event: (event.start_frame, event.event_type, event.player_id or ""))
        return candidates[:500]

    def _record_player_key(self, record: AnalysisRecordResponse) -> str:
        return record.global_player_id or record.local_player_id or f"player_{record.player}"

    def _detect_block_candidates(
        self,
        records: List[AnalysisRecordResponse],
    ) -> List[EventCandidateResponse]:
        block_records = [record for record in records if record.final.action == "block"]
        grouped: Dict[str, List[AnalysisRecordResponse]] = defaultdict(list)
        for record in block_records:
            grouped[self._record_player_key(record)].append(record)

        candidates: List[EventCandidateResponse] = []
        max_gap = int(self.settings.seq_length) + int(getattr(self.settings, "action_vid_stride", self.settings.vid_stride))
        for player_id, player_records in grouped.items():
            current: List[AnalysisRecordResponse] = []
            for record in sorted(player_records, key=lambda item: item.start_frame):
                if current and int(record.start_frame) - int(current[-1].end_frame) > max_gap:
                    candidates.append(self._make_block_candidate(player_id, current))
                    current = []
                current.append(record)
            if current:
                candidates.append(self._make_block_candidate(player_id, current))
        return candidates

    def _make_block_candidate(
        self,
        player_id: str,
        records: List[AnalysisRecordResponse],
    ) -> EventCandidateResponse:
        avg_confidence = sum(float(record.final.confidence) for record in records) / max(1, len(records))
        segment_ids = sorted({record.segment_id for record in records if record.segment_id is not None})
        return EventCandidateResponse(
            event_type="block_candidate",
            player_id=player_id,
            segment_id=segment_ids[0] if segment_ids else None,
            start_frame=min(int(record.start_frame) for record in records),
            end_frame=max(int(record.end_frame) for record in records),
            confidence=min(0.65, avg_confidence * 0.55),
            method="action_cluster_candidate_v1",
            status="candidate_requires_ball_rim_or_vlm_confirmation",
            evidence=[
                f"{len(records)} contiguous block-classified clips",
                "downgraded from official block because ball/rim/shot evidence is not available",
            ],
        )

    def _detect_rebound_candidates(
        self,
        records: List[AnalysisRecordResponse],
    ) -> List[EventCandidateResponse]:
        possession_actions = {"ball in hand", "dribble"}
        candidates: List[EventCandidateResponse] = []
        shots = [record for record in records if record.final.action == "shoot"]
        for shot in shots:
            next_possessions = [
                record
                for record in records
                if int(shot.end_frame) < int(record.start_frame) <= int(shot.end_frame) + 120
                and record.final.action in possession_actions
            ]
            if not next_possessions:
                continue
            receiver = min(next_possessions, key=lambda record: int(record.start_frame))
            candidates.append(
                EventCandidateResponse(
                    event_type="rebound_candidate",
                    player_id=self._record_player_key(receiver),
                    segment_id=receiver.segment_id,
                    start_frame=int(shot.start_frame),
                    end_frame=int(receiver.end_frame),
                    confidence=0.35,
                    method="shot_to_next_possession_candidate_v1",
                    status="candidate_requires_miss_and_ball_confirmation",
                    evidence=[
                        f"shoot action by {self._record_player_key(shot)}",
                        f"next possession-like action {receiver.final.action} by {self._record_player_key(receiver)}",
                    ],
                )
            )
        return candidates[:200]

    def _detect_steal_candidates(
        self,
        records: List[AnalysisRecordResponse],
    ) -> List[EventCandidateResponse]:
        possession_actions = {"ball in hand", "dribble"}
        pressure_actions = {"defense", "block"}
        possession_records = [
            record
            for record in records
            if record.final.action in possession_actions
        ]
        candidates: List[EventCandidateResponse] = []
        for previous, current in zip(possession_records, possession_records[1:]):
            previous_player = self._record_player_key(previous)
            current_player = self._record_player_key(current)
            if previous_player == current_player:
                continue
            gap = int(current.start_frame) - int(previous.end_frame)
            if gap < 0 or gap > 90:
                continue
            pressure = [
                record
                for record in records
                if self._record_player_key(record) == current_player
                and record.final.action in pressure_actions
                and int(previous.start_frame) - 60 <= int(record.start_frame) <= int(current.end_frame)
            ]
            if not pressure:
                continue
            candidates.append(
                EventCandidateResponse(
                    event_type="steal_candidate",
                    player_id=current_player,
                    segment_id=current.segment_id,
                    start_frame=int(previous.start_frame),
                    end_frame=int(current.end_frame),
                    confidence=0.30,
                    method="possession_switch_pressure_candidate_v1",
                    status="candidate_requires_ball_touch_confirmation",
                    evidence=[
                        f"possession-like action moved from {previous_player} to {current_player}",
                        "receiver had defense/block pressure action in the transition window",
                    ],
                )
            )
        return candidates[:200]

    def _estimate_player_statistics(self, action_counts: Counter[str]) -> PlayerBoxScoreEstimateResponse:
        """Estimate basic basketball box-score fields from action labels.

        The current model predicts actions rather than made-shot, possession, or ball-event
        outcomes, so these fields are intentionally marked as low-confidence proxies.
        """
        shots = int(action_counts.get("shoot", 0))
        passes = int(action_counts.get("pass", 0))
        block_candidates = int(action_counts.get("block", 0))
        rebounds = int(action_counts.get("rebound", 0))
        steals = int(action_counts.get("steal", 0))
        notes = [
            "points are estimated as 2 per shoot action; made/missed shots are not detected yet",
            "assists are estimated from pass actions; receiver score linkage is not detected yet",
            "block actions are emitted as event candidates and are not counted as official blocks without ball/rim/shot confirmation",
        ]
        if block_candidates:
            notes.append(f"{block_candidates} block-classified clips are available as block_candidate evidence")
        if rebounds == 0:
            notes.append("rebounds require missed-shot and next-possession confirmation")
        if steals == 0:
            notes.append("steals require possession-change and ball-touch confirmation")
        return PlayerBoxScoreEstimateResponse(
            points=shots * 2,
            assists=passes,
            rebounds=rebounds,
            blocks=0,
            steals=steals,
            confidence=0.35 if (shots or passes or block_candidates or rebounds or steals) else 0.15,
            notes=notes,
        )

    def _write_video_segment(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        fps: float,
        width: int,
        height: int,
    ) -> str:
        temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_path = temp.name
        temp.close()

        cap = cv2.VideoCapture(video_path)
        writer = cv2.VideoWriter(
            temp_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        try:
            if not cap.isOpened() or not writer.isOpened():
                raise RuntimeError("Failed to create temporary long-video segment.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_index = start_frame
            while frame_index <= end_frame:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                frame_index += 1
        finally:
            cap.release()
            writer.release()
        return temp_path

    def _sample_contact_sheet_frames(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        sample_count: int,
    ) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        frames: List[np.ndarray] = []
        try:
            if not cap.isOpened():
                return frames
            indices = np.linspace(start_frame, end_frame, max(1, sample_count), dtype=int)
            for frame_index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = cap.read()
                if ok:
                    frames.append(frame)
        finally:
            cap.release()
        return frames

    def _make_contact_sheet(self, frames: List[np.ndarray], cell_width: int = 320, columns: int = 3) -> np.ndarray:
        if not frames:
            return np.zeros((180, cell_width, 3), dtype=np.uint8)
        resized: List[np.ndarray] = []
        for frame in frames:
            height, width = frame.shape[:2]
            cell_height = max(1, int(cell_width * height / width))
            resized.append(cv2.resize(frame, (cell_width, cell_height), interpolation=cv2.INTER_AREA))
        cell_height = resized[0].shape[0]
        rows = int(np.ceil(len(resized) / columns))
        sheet = np.full((rows * cell_height, columns * cell_width, 3), 245, dtype=np.uint8)
        for index, frame in enumerate(resized):
            row, column = divmod(index, columns)
            sheet[row * cell_height : (row + 1) * cell_height, column * cell_width : (column + 1) * cell_width] = frame
        return sheet

    def _compare_segment_with_vlm(
        self,
        player_count: int,
        summary: AnalysisSummaryResponse,
        vlm_audit: Optional[VLMVideoAuditResponse],
    ) -> tuple[str, List[str]]:
        if vlm_audit is None:
            return "warn_vlm_not_configured", ["VLM audit was disabled."]
        if not vlm_audit.available:
            return "warn_vlm_unavailable", [vlm_audit.limitations or "VLM audit unavailable."]

        notes: List[str] = []
        if vlm_audit.player_count_min is not None and player_count < vlm_audit.player_count_min:
            notes.append(
                f"AGU counted {player_count} players, VLM saw at least {vlm_audit.player_count_min}."
            )
            return "fail_player_under_count", notes

        model_actions = set(summary.action_counts)
        audit_actions = {self._normalize_audit_action(action) for action in vlm_audit.actions}
        audit_actions.discard(None)
        if audit_actions and model_actions and not (audit_actions & model_actions):
            notes.append(
                f"AGU actions {sorted(model_actions)} did not overlap VLM actions {sorted(audit_actions)}."
            )
            return "fail_action_mismatch", notes

        if vlm_audit.confidence < 0.5:
            notes.append(f"VLM audit confidence is low: {vlm_audit.confidence:.2f}.")
            return "warn_low_confidence", notes

        return "pass", notes

    def _normalize_audit_action(self, action: str) -> Optional[str]:
        from app.analysis.vlm import normalize_action

        aliases = {
            "dribbling": "dribble",
            "运球": "dribble",
            "passing": "pass",
            "传球": "pass",
            "shooting": "shoot",
            "投篮": "shoot",
            "defending": "defense",
            "防守": "defense",
            "running": "run",
            "跑动": "run",
            "walking": "walk",
            "走动": "walk",
            "持球": "ball in hand",
            "rebound": "no_action",
            "抢篮板": "no_action",
        }
        cleaned = aliases.get(action.strip().lower(), action)
        return normalize_action(cleaned)

    def _summarize_response_records(self, records: List[AnalysisRecordResponse]) -> AnalysisSummaryResponse:
        action_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        needs_review = 0
        for record in records:
            action_counts[record.final.action] += 1
            source_counts[record.final.source] += 1
            needs_review += int(record.final.needs_review)
        return AnalysisSummaryResponse(
            clip_count=len(records),
            action_counts=dict(action_counts),
            needs_review_count=needs_review,
            source_counts=dict(source_counts),
        )
