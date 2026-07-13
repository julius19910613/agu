import unittest
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

import torch

from app.analysis.identity_embedding import SidecarHsvHistogramEmbedder, build_identity_embedder
from app.analysis.schemas import (
    AnalysisRecordResponse,
    AnalysisRequest,
    ConfirmedIdentityMergeResponse,
    FinalDecisionResponse,
    IdentityDuplicateCandidateResponse,
    JerseyNumberCandidateResponse,
    LongVideoPlayerSummaryResponse,
    ModelPrediction,
    MotionFeatures,
    PlayerIdentityFeatureResponse,
    ScoreboardCheckpointResponse,
    VLMDecisionResponse,
    VLMIdentityMergeDecisionResponse,
    VLMVideoAuditResponse,
)
from app.analysis.service import AnalysisService
from app.analysis.vlm import (
    OllamaVLMVerifier,
    _last_visible_scoreboard,
    _parse_scoreboard_checkpoints,
    extract_json_object,
    normalize_action,
)
from app.analysis.fusion import fuse_decision, should_call_vlm, apply_temporal_smoothing
from app.analysis.tracking import crop_windows, densify_track_boxes, resolve_yolo_tracker_config, select_active_track_ids
from scripts.build_identity_duplicate_report import build_duplicate_report


class HybridAnalysisTest(unittest.TestCase):
    def make_prediction(self, action="dribble", confidence=0.4):
        action_ids = {
            "block": 0,
            "pass": 1,
            "run": 2,
            "dribble": 3,
            "shoot": 4,
            "ball in hand": 5,
            "defense": 6,
            "pick": 7,
            "no_action": 8,
            "walk": 9,
        }
        return ModelPrediction(
            action_id=action_ids[action],
            action=action,
            confidence=confidence,
            probabilities={action: confidence},
        )

    def make_service(self):
        settings = MagicMock()
        settings.seq_length = 16
        settings.vid_stride = 8
        settings.action_vid_stride = 24
        settings.smoothing_confidence = 0.6
        settings.torch_num_threads = 0
        settings.progress_log = False
        settings.identity_embedding_backend = "sidecar_hsv_hist"
        settings.identity_embedding_weights = "default"
        settings.identity_embedding_device = "cpu"
        settings.identity_embedding_batch_size = 2
        settings.identity_embedding_allow_fallback = True
        settings.face_identity_backend = "off"
        settings.face_detection_model_path = ""
        settings.face_recognition_model_path = ""
        settings.face_detection_score_threshold = 0.60
        settings.face_identity_allow_fallback = True
        settings.jersey_number_vlm_enabled = False
        settings.jersey_number_vlm_frames = 2
        settings.vlm_identity_merge_enabled = False
        settings.vlm_identity_merge_max_candidates = 8
        settings.vlm_identity_merge_confidence = 0.78
        settings.vlm_identity_merge_crops_per_side = 3
        return AnalysisService(settings=settings, model=MagicMock(), device=torch.device("cpu"))

    def make_record(self, player, action, start_frame, end_frame, segment_id=0, confidence=0.7):
        action_ids = {
            "block": 0,
            "pass": 1,
            "run": 2,
            "dribble": 3,
            "shoot": 4,
            "ball in hand": 5,
            "defense": 6,
            "pick": 7,
            "no_action": 8,
            "walk": 9,
        }
        return AnalysisRecordResponse(
            player=player,
            clip_index=start_frame,
            start_frame=start_frame,
            end_frame=end_frame,
            segment_id=segment_id,
            local_player_id=f"segment_{segment_id}:player_{player}",
            global_player_id=f"player_{player:03d}",
            identity_confidence=0.6,
            r2plus1d=self.make_prediction(action, confidence),
            motion=MotionFeatures(avg_center_speed=1.0, max_center_speed=2.0, avg_box_area=100.0, area_change_ratio=1.0),
            vlm=None,
            final=FinalDecisionResponse(
                action_id=action_ids[action],
                action=action,
                confidence=confidence,
                source="r2plus1d",
                needs_review=False,
                reason="test",
            ),
        )

    def test_extract_json_object_accepts_wrapped_json(self):
        parsed = extract_json_object('Here is the result: {"action": "shoot", "confidence": 0.7}')
        self.assertEqual(parsed["action"], "shoot")
        self.assertEqual(parsed["confidence"], 0.7)

    def test_normalize_action_maps_aliases(self):
        self.assertEqual(normalize_action("ball_in_hand"), "ball in hand")
        self.assertEqual(normalize_action("no action"), "no_action")
        self.assertEqual(normalize_action("defence"), "defense")
        self.assertIsNone(normalize_action("guarding"))

    def test_fuse_uses_r2plus1d_when_vlm_unavailable(self):
        prediction = self.make_prediction("shoot", 0.8)
        final = fuse_decision(prediction, None, high_confidence=0.75, low_confidence=0.55)
        self.assertEqual(final.action, "shoot")
        self.assertEqual(final.source, "r2plus1d")
        self.assertFalse(final.needs_review)

    def test_fuse_vlm_override_for_low_confidence_prediction(self):
        prediction = self.make_prediction("dribble", 0.35)
        vlm = VLMDecisionResponse(
            action="defense",
            confidence=0.72,
            reason="No ball is visible and stance is defensive.",
            visible_ball=False,
            needs_review=False,
            raw_response="{}",
            available=True,
        )
        final = fuse_decision(prediction, vlm, high_confidence=0.75, low_confidence=0.55)
        self.assertEqual(final.action, "defense")
        self.assertEqual(final.source, "vlm_override")

    def test_should_call_vlm_respects_mode_and_limit(self):
        low_prediction = self.make_prediction("dribble", 0.4)
        high_prediction = self.make_prediction("shoot", 0.9)
        self.assertTrue(should_call_vlm("low-confidence", low_prediction, 0.55, 0, 2))
        self.assertFalse(should_call_vlm("low-confidence", high_prediction, 0.55, 0, 2))
        self.assertFalse(should_call_vlm("off", low_prediction, 0.55, 0, 2))
        self.assertFalse(should_call_vlm("always", low_prediction, 0.55, 2, 2))

    def test_crop_windows_pads_short_tail(self):
        frames = [np.full((32, 32, 3), fill_value=index, dtype=np.uint8) for index in range(10)]
        boxes = [[(0, 0, 16, 16)]] * 10
        windows = crop_windows(frames, boxes, seq_length=4, vid_stride=3)
        self.assertEqual(len(windows[0]), 3)
        self.assertEqual(windows[0][0].shape, (4, 176, 128, 3))

    def test_dense_track_boxes_do_not_carry_long_missing_tracks(self):
        raw = [
            {7: (0.0, 0.0, 10.0, 20.0)},
            {},
            {7: (20.0, 0.0, 10.0, 20.0)},
            {},
            {},
            {},
        ]

        dense = densify_track_boxes(raw, [7], max_gap_frames=1)

        self.assertEqual(dense[0][0], (0.0, 0.0, 10.0, 20.0))
        self.assertEqual(dense[1][0], (10.0, 0.0, 10.0, 20.0))
        self.assertEqual(dense[3][0], (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(dense[-1][0], (0.0, 0.0, 0.0, 0.0))

    def test_crop_windows_skips_mostly_missing_player_windows(self):
        frames = [np.full((32, 32, 3), 50, dtype=np.uint8) for _ in range(8)]
        boxes = [
            ((2.0, 2.0, 10.0, 20.0),) if index < 2 else ((0.0, 0.0, 0.0, 0.0),)
            for index in range(8)
        ]

        windows, indices = crop_windows(
            frames,
            boxes,
            seq_length=4,
            vid_stride=2,
            min_visible_ratio=0.75,
            return_clip_indices=True,
        )

        self.assertEqual(windows[0], [])
        self.assertEqual(indices[0], [])

    def test_acceleration_request_fields_are_supported(self):
        request = AnalysisRequest(
            video_path="examples/lebron_shoots.mp4",
            action_vid_stride=24,
            tracking_fps=8.0,
            yolo_imgsz=320,
            max_players_per_segment=12,
            yolo_device="cpu",
            tracker_backend="botsort",
            yolo_tracker_config="botsort.yaml",
            yolo_reid_enabled=True,
            yolo_reid_model="auto",
            identity_embedding_backend="torchvision_mobilenet_v3_small",
            identity_embedding_weights="none",
            identity_embedding_device="cpu",
            jersey_number_vlm_enabled=True,
            jersey_number_vlm_frames=2,
            confirmed_identity_merges=[
                ConfirmedIdentityMergeResponse(
                    canonical_global_player_id="player_004",
                    merged_global_player_ids=["player_006"],
                    source="manual_review",
                    confidence=0.95,
                    evidence=["same jersey number and appearance"],
                )
            ],
            vlm_identity_merge_enabled=True,
            vlm_identity_merge_max_candidates=4,
            vlm_identity_merge_confidence=0.82,
            r2plus1d_device="mps_if_available",
        )
        self.assertEqual(request.action_vid_stride, 24)
        self.assertEqual(request.tracking_fps, 8.0)
        self.assertEqual(request.yolo_imgsz, 320)
        self.assertEqual(request.max_players_per_segment, 12)
        self.assertEqual(request.yolo_device, "cpu")
        self.assertEqual(request.tracker_backend, "botsort")
        self.assertEqual(request.yolo_tracker_config, "botsort.yaml")
        self.assertTrue(request.yolo_reid_enabled)
        self.assertEqual(request.yolo_reid_model, "auto")
        self.assertEqual(request.identity_embedding_backend, "torchvision_mobilenet_v3_small")
        self.assertEqual(request.identity_embedding_weights, "none")
        self.assertEqual(request.identity_embedding_device, "cpu")
        self.assertTrue(request.jersey_number_vlm_enabled)
        self.assertEqual(request.jersey_number_vlm_frames, 2)
        self.assertEqual(request.confirmed_identity_merges[0].canonical_global_player_id, "player_004")
        self.assertEqual(request.confirmed_identity_merges[0].merged_global_player_ids, ["player_006"])
        self.assertTrue(request.vlm_identity_merge_enabled)
        self.assertEqual(request.vlm_identity_merge_max_candidates, 4)
        self.assertEqual(request.vlm_identity_merge_confidence, 0.82)
        self.assertEqual(request.r2plus1d_device, "mps_if_available")

    def test_yolo_tracker_adapter_config_resolution(self):
        self.assertEqual(resolve_yolo_tracker_config(), "bytetrack.yaml")
        self.assertEqual(resolve_yolo_tracker_config(tracker_backend="botsort"), "botsort.yaml")
        self.assertEqual(
            resolve_yolo_tracker_config(tracker_backend="custom", yolo_tracker_config="/tmp/custom.yaml"),
            "/tmp/custom.yaml",
        )

    def test_botsort_reid_adapter_generates_tracker_config(self):
        path = resolve_yolo_tracker_config(
            tracker_backend="botsort",
            reid_enabled=True,
            reid_model="auto",
        )
        self.assertTrue(path.endswith(".yaml"))
        with open(path, "r") as fp:
            text = fp.read()
        self.assertIn("tracker_type: botsort", text)
        self.assertIn("with_reid: True", text)
        self.assertIn("model: auto", text)

    def test_select_active_track_ids_caps_to_strongest_tracks(self):
        selected = select_active_track_ids(
            appearance_counts={3: 10, 1: 30, 2: 20},
            frame_count=40,
            min_appear_ratio=0.1,
            min_appear_abs=1,
            max_players=2,
        )
        self.assertEqual(selected, [1, 2])

    def test_block_actions_are_not_counted_as_official_blocks(self):
        service = self.make_service()
        stats = service._estimate_player_statistics(Counter({"block": 5, "shoot": 1}))
        self.assertEqual(stats.points, 0)
        self.assertEqual(stats.shot_attempts, 1)
        self.assertEqual(stats.point_candidate_count, 1)
        self.assertEqual(stats.blocks, 0)
        self.assertEqual(stats.status, "estimate_requires_event_confirmation")
        self.assertIn("points", stats.candidate_fields)
        self.assertIn("blocks", stats.candidate_fields)
        self.assertTrue(any("block_candidate" in note for note in stats.notes))

    def test_event_candidates_include_ranked_owner_candidates(self):
        service = self.make_service()
        records = [
            self.make_record(1, "shoot", 10, 25, confidence=0.8),
            self.make_record(2, "ball in hand", 35, 50, confidence=0.75),
            self.make_record(3, "defense", 30, 45, confidence=0.7),
        ]

        candidates = service._detect_event_candidates(records)
        rebound = next(candidate for candidate in candidates if candidate.event_type == "rebound_candidate")

        self.assertEqual(rebound.player_id, "player_002")
        self.assertTrue(rebound.owner_candidates)
        self.assertEqual(rebound.owner_candidates[0].rank, 1)
        self.assertGreaterEqual(rebound.owner_candidates[0].score, rebound.owner_candidates[-1].score)

    def test_single_low_confidence_block_does_not_become_event_candidate(self):
        service = self.make_service()
        records = [
            self.make_record(1, "block", 10, 25, confidence=0.62),
            self.make_record(2, "dribble", 20, 35, confidence=0.8),
        ]

        candidates = service._detect_event_candidates(records)

        self.assertFalse(any(candidate.event_type == "block_candidate" for candidate in candidates))

    def test_repeated_block_clips_remain_event_candidate(self):
        service = self.make_service()
        records = [
            self.make_record(1, "block", 10, 25, confidence=0.62),
            self.make_record(1, "block", 30, 45, confidence=0.66),
        ]

        candidates = service._detect_event_candidates(records)

        self.assertTrue(any(candidate.event_type == "block_candidate" for candidate in candidates))

    def test_vlm_audit_without_block_suppresses_block_candidate(self):
        service = self.make_service()
        records = [
            self.make_record(1, "block", 10, 25, confidence=0.82),
            self.make_record(1, "block", 30, 45, confidence=0.78),
        ]
        audit = VLMVideoAuditResponse(
            available=True,
            actions=["dribble", "defense", "ball in hand"],
            confidence=0.75,
        )

        candidates = service._detect_event_candidates(records, segment_audits={0: audit})

        self.assertFalse(any(candidate.event_type == "block_candidate" for candidate in candidates))

    def test_vlm_audit_with_block_keeps_block_candidate(self):
        service = self.make_service()
        records = [
            self.make_record(1, "block", 10, 25, confidence=0.82),
            self.make_record(1, "block", 30, 45, confidence=0.78),
        ]
        audit = VLMVideoAuditResponse(
            available=True,
            actions=["block", "defense"],
            confidence=0.75,
        )

        candidates = service._detect_event_candidates(records, segment_audits={0: audit})

        self.assertTrue(any(candidate.event_type == "block_candidate" for candidate in candidates))

    def test_confirmed_identity_merges_emit_merged_player_statistics(self):
        service = self.make_service()
        summaries = [
            LongVideoPlayerSummaryResponse(
                player_id="segment_0:player_4",
                global_player_id="player_004",
                identity_confidence=0.42,
                identity_method="appearance_continuity_stitch_v2",
                identity_evidence=["embedding similarity 0.91"],
                segments_seen=1,
                clip_count=4,
                action_counts={"shoot": 2, "pass": 1, "rebound": 1},
                needs_review_count=1,
                average_confidence=0.7,
            ),
            LongVideoPlayerSummaryResponse(
                player_id="segment_2:player_6",
                global_player_id="player_006",
                identity_confidence=0.38,
                identity_method="appearance_continuity_stitch_v2",
                identity_evidence=["jersey number 00"],
                segments_seen=1,
                clip_count=3,
                action_counts={"steal": 1, "block": 1, "shoot": 1},
                needs_review_count=0,
                average_confidence=0.8,
            ),
        ]
        merged = service._build_confirmed_merged_player_summaries(
            summaries,
            [
                ConfirmedIdentityMergeResponse(
                    canonical_global_player_id="player_004",
                    merged_global_player_ids=["player_006"],
                    source="manual_review",
                    confidence=0.96,
                    evidence=["review contact sheet confirmed same player"],
                )
            ],
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].global_player_id, "player_004")
        self.assertEqual(merged[0].merged_from_global_player_ids, ["player_004", "player_006"])
        self.assertEqual(merged[0].segments_seen, 2)
        self.assertEqual(merged[0].clip_count, 7)
        self.assertEqual(merged[0].action_counts["shoot"], 3)
        self.assertEqual(merged[0].statistics.points, 0)
        self.assertEqual(merged[0].statistics.shot_attempts, 3)
        self.assertEqual(merged[0].statistics.point_candidate_count, 3)
        self.assertEqual(merged[0].statistics.assists, 1)
        self.assertEqual(merged[0].statistics.rebounds, 1)
        self.assertEqual(merged[0].statistics.steals, 1)
        self.assertEqual(merged[0].statistics.blocks, 0)
        self.assertIn("review contact sheet confirmed same player", merged[0].merge_evidence)

    def test_vlm_identity_merge_decision_can_confirm_candidate(self):
        service = self.make_service()
        candidate = IdentityDuplicateCandidateResponse(
            left_global_player_id="player_004",
            right_global_player_id="player_006",
            confidence=0.86,
            left_local_player_ids=["segment_0:player_4"],
            right_local_player_ids=["segment_2:player_6"],
            evidence=["appearance embedding similarity 0.91"],
        )
        decision = VLMIdentityMergeDecisionResponse(
            left_global_player_id="player_004",
            right_global_player_id="player_006",
            is_same_player=True,
            confidence=0.88,
            canonical_global_player_id="player_004",
            merged_global_player_ids=["player_006"],
            reason="same jersey and body shape",
            evidence=["same jersey number"],
            available=True,
        )
        merge = service._confirmed_merge_from_vlm_decision(candidate, decision, confidence_threshold=0.78)
        self.assertIsNotNone(merge)
        self.assertEqual(merge.canonical_global_player_id, "player_004")
        self.assertEqual(merge.merged_global_player_ids, ["player_006"])
        self.assertEqual(merge.source, "vlm_identity_merge_v1")
        self.assertEqual(merge.confidence, 0.88)

    def test_vlm_identity_merge_decision_rejects_low_confidence_or_negative(self):
        service = self.make_service()
        candidate = IdentityDuplicateCandidateResponse(
            left_global_player_id="player_004",
            right_global_player_id="player_006",
            confidence=0.86,
        )
        low_confidence = VLMIdentityMergeDecisionResponse(
            left_global_player_id="player_004",
            right_global_player_id="player_006",
            is_same_player=True,
            confidence=0.55,
            available=True,
        )
        negative = VLMIdentityMergeDecisionResponse(
            left_global_player_id="player_004",
            right_global_player_id="player_006",
            is_same_player=False,
            confidence=0.95,
            available=True,
        )
        self.assertIsNone(service._confirmed_merge_from_vlm_decision(candidate, low_confidence, 0.78))
        self.assertIsNone(service._confirmed_merge_from_vlm_decision(candidate, negative, 0.78))

    def test_identity_review_crops_use_absolute_sampled_box_frames(self):
        service = self.make_service()
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "sample.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                2.0,
                (40, 40),
            )
            for index in range(5):
                frame = np.zeros((40, 40, 3), dtype=np.uint8)
                if index == 2:
                    frame[:, :] = (0, 255, 0)
                elif index == 4:
                    frame[:, :] = (0, 0, 255)
                writer.write(frame)
            writer.release()

            features = {
                "segment_1:player_2": PlayerIdentityFeatureResponse(
                    player=2,
                    segment_id=1,
                    local_player_id="segment_1:player_2",
                    start_frame=2,
                    end_frame=4,
                    sampled_boxes=[{"frame": 2, "x": 4, "y": 4, "w": 24, "h": 24}],
                )
            }
            self.assertEqual(
                service._sampled_box_absolute_frame(
                    features["segment_1:player_2"],
                    {"frame": 0, "x": 4, "y": 4, "w": 24, "h": 24},
                ),
                2,
            )
            self.assertEqual(
                service._sampled_box_absolute_frame(
                    features["segment_1:player_2"],
                    {"frame": 2, "x": 4, "y": 4, "w": 24, "h": 24},
                ),
                2,
            )

            crops = service._extract_identity_review_crops(
                str(video_path),
                ["segment_1:player_2"],
                features,
                label="LEFT player_002",
                max_crops=1,
            )

        self.assertEqual(len(crops), 1)
        body = crops[0][32:, :, :]
        self.assertGreater(float(body[:, :, 1].mean()), float(body[:, :, 2].mean()))

    def test_adjacent_segment_identity_stitch_assigns_global_ids(self):
        service = self.make_service()
        summaries = [
            LongVideoPlayerSummaryResponse(
                player_id="segment_0:player_2",
                segments_seen=1,
                clip_count=3,
                action_counts={"dribble": 2, "pass": 1},
                needs_review_count=0,
                average_confidence=0.7,
            ),
            LongVideoPlayerSummaryResponse(
                player_id="segment_1:player_2",
                segments_seen=1,
                clip_count=3,
                action_counts={"dribble": 2, "pass": 1},
                needs_review_count=0,
                average_confidence=0.7,
            ),
        ]
        identity_map, identity_confidences, identity_evidence = service._merge_segment_local_identities(summaries)
        self.assertEqual(identity_map["segment_0:player_2"], identity_map["segment_1:player_2"])
        self.assertGreater(identity_confidences["segment_1:player_2"], 0.4)
        self.assertTrue(identity_evidence["segment_1:player_2"])

    def test_identity_stitch_uses_appearance_and_track_continuity(self):
        service = self.make_service()
        summaries = [
            LongVideoPlayerSummaryResponse(
                player_id="segment_0:player_2",
                segments_seen=1,
                clip_count=3,
                action_counts={"dribble": 2, "pass": 1},
                needs_review_count=0,
                average_confidence=0.7,
            ),
            LongVideoPlayerSummaryResponse(
                player_id="segment_1:player_7",
                segments_seen=1,
                clip_count=3,
                action_counts={"dribble": 2, "pass": 1},
                needs_review_count=0,
                average_confidence=0.7,
            ),
        ]
        features = {
            "segment_0:player_2": PlayerIdentityFeatureResponse(
                player=2,
                segment_id=0,
                local_player_id="segment_0:player_2",
                start_frame=0,
                end_frame=100,
                first_center=[100.0, 200.0],
                last_center=[150.0, 220.0],
                appearance_signature={"h_mean": 0.2, "s_mean": 0.5, "v_mean": 0.7, "b_mean": 0.1, "g_mean": 0.2, "r_mean": 0.3},
                appearance_embedding=[0.8, 0.2, 0.1, 0.0],
                embedding_model="test_embedding",
                embedding_dim=4,
                track_coverage=1.0,
            ),
            "segment_1:player_7": PlayerIdentityFeatureResponse(
                player=7,
                segment_id=1,
                local_player_id="segment_1:player_7",
                start_frame=101,
                end_frame=200,
                first_center=[160.0, 225.0],
                last_center=[220.0, 250.0],
                appearance_signature={"h_mean": 0.21, "s_mean": 0.49, "v_mean": 0.69, "b_mean": 0.1, "g_mean": 0.21, "r_mean": 0.31},
                appearance_embedding=[0.79, 0.21, 0.1, 0.0],
                embedding_model="test_embedding",
                embedding_dim=4,
                track_coverage=1.0,
            ),
        }
        identity_map, identity_confidences, identity_evidence = service._merge_segment_local_identities(summaries, features)
        self.assertEqual(identity_map["segment_0:player_2"], identity_map["segment_1:player_7"])
        self.assertGreater(identity_confidences["segment_1:player_7"], 0.6)
        evidence = " ".join(identity_evidence["segment_1:player_7"])
        self.assertIn("embedding similarity", evidence)
        self.assertIn("SFace similarity", evidence)
        self.assertIn("jersey darkness gap", evidence)
        self.assertIn("track continuity", evidence)

    def test_identity_feature_extraction_generates_sidecar_embedding(self):
        service = self.make_service()
        frames = [np.full((32, 32, 3), fill_value=80 + index, dtype=np.uint8) for index in range(4)]
        boxes = [[(4.0, 4.0, 16.0, 20.0)]] * 4
        features = service._extract_player_identity_features(frames, boxes)
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].embedding_model, "sidecar_hsv_hist_embedding_v1")
        self.assertEqual(features[0].embedding_dim, 128)
        self.assertEqual(len(features[0].appearance_embedding), 128)
        self.assertIn("torso_luma_mean", features[0].appearance_signature)
        self.assertIn("jersey_dark_ratio", features[0].appearance_signature)

    def test_identity_feature_extraction_adds_face_sidecar_when_visible(self):
        service = self.make_service()
        frames = [np.full((64, 48, 3), fill_value=100 + index, dtype=np.uint8) for index in range(3)]
        boxes = [[(0.0, 0.0, 48.0, 64.0)]] * 3
        face = np.full((20, 20, 3), (80, 130, 180), dtype=np.uint8)
        with patch.object(service, "_detect_face_crop", return_value=face):
            features = service._extract_player_identity_features(frames, boxes)

        self.assertEqual(features[0].face_sample_count, 3)
        self.assertEqual(len(features[0].face_embedding), 128)
        self.assertIn("opencv_haar_face", features[0].face_embedding_model)

    def test_identity_similarity_rejects_conflicting_jersey_tones(self):
        service = self.make_service()
        common = dict(
            player=0,
            start_frame=0,
            end_frame=10,
            appearance_embedding=[1.0, 0.0],
            face_embedding=[1.0, 0.0],
            face_sample_count=1,
        )
        dark = PlayerIdentityFeatureResponse(
            **common,
            appearance_signature={"jersey_dark_ratio": 0.9},
        )
        light = PlayerIdentityFeatureResponse(
            **{**common, "player": 1},
            appearance_signature={"jersey_dark_ratio": 0.1},
        )

        self.assertLessEqual(service._appearance_similarity(dark, light), 0.35)

    def test_identity_similarity_rejects_conflicting_sface_embeddings(self):
        service = self.make_service()
        common = dict(
            start_frame=0,
            end_frame=10,
            appearance_signature={"jersey_dark_ratio": 0.4},
            appearance_embedding=[1.0, 0.0],
            face_sample_count=2,
            face_embedding_model="opencv_yunet_2023mar+sface_2021dec_embedding_v1",
            face_embedding_quality=0.8,
        )
        left = PlayerIdentityFeatureResponse(
            player=0,
            face_embedding=[1.0, 0.0],
            **common,
        )
        right = PlayerIdentityFeatureResponse(
            player=1,
            face_embedding=[0.0, 1.0],
            **common,
        )

        self.assertLessEqual(service._appearance_similarity(left, right), 0.40)

    def test_identity_feature_extraction_can_use_torchvision_backend(self):
        service = self.make_service()
        service.settings.identity_embedding_backend = "torchvision_mobilenet_v3_small"
        service.settings.identity_embedding_weights = "none"
        service.settings.identity_embedding_device = "cpu"
        frames = [np.full((48, 48, 3), fill_value=80 + index, dtype=np.uint8) for index in range(2)]
        boxes = [[(4.0, 4.0, 24.0, 28.0)]] * 2
        features = service._extract_player_identity_features(frames, boxes)
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].embedding_model, "torchvision_mobilenet_v3_small_none_embedding_v1")
        self.assertEqual(features[0].embedding_dim, 576)
        self.assertEqual(len(features[0].appearance_embedding), 576)

    def test_torchreid_osnet_backend_falls_back_when_optional_dependency_is_missing(self):
        with patch.dict("sys.modules", {"torchreid": None}):
            embedder = build_identity_embedder(
                backend="torchreid_osnet_x0_25",
                weights="none",
                device="cpu",
                allow_fallback=True,
            )
        self.assertIsInstance(embedder, SidecarHsvHistogramEmbedder)

    def test_torchreid_osnet_backend_raises_when_fallback_is_disabled(self):
        with patch.dict("sys.modules", {"torchreid": None}):
            with self.assertRaises(ImportError):
                build_identity_embedder(
                    backend="torchreid_osnet_x0_25",
                    weights="none",
                    device="cpu",
                    allow_fallback=False,
                )

    def test_identity_feature_extraction_can_attach_jersey_number_candidates(self):
        class FakeJerseyVerifier:
            def read_jersey_number(self, frames, scope=""):
                self.frames = frames
                self.scope = scope
                return [
                    JerseyNumberCandidateResponse(
                        number="00",
                        confidence=0.82,
                        visible=True,
                        reason="visible back jersey",
                    )
                ]

        service = self.make_service()
        frames = [np.full((48, 48, 3), fill_value=80 + index, dtype=np.uint8) for index in range(2)]
        boxes = [[(4.0, 4.0, 24.0, 28.0)]] * 2
        verifier = FakeJerseyVerifier()
        features = service._extract_player_identity_features(
            frames,
            boxes,
            jersey_number_verifier=verifier,
            jersey_number_frames=1,
        )
        self.assertEqual(features[0].jersey_number_candidates[0].number, "00")
        self.assertEqual(features[0].jersey_number_candidates[0].confidence, 0.82)
        self.assertEqual(len(verifier.frames), 1)

    def test_identity_stitch_does_not_merge_players_within_same_segment(self):
        service = self.make_service()
        summaries = [
            LongVideoPlayerSummaryResponse(
                player_id="segment_0:player_1",
                segments_seen=1,
                clip_count=3,
                action_counts={"dribble": 3},
                needs_review_count=0,
                average_confidence=0.7,
            ),
            LongVideoPlayerSummaryResponse(
                player_id="segment_0:player_2",
                segments_seen=1,
                clip_count=3,
                action_counts={"dribble": 3},
                needs_review_count=0,
                average_confidence=0.7,
            ),
        ]
        identity_map, _, _ = service._merge_segment_local_identities(summaries)
        self.assertNotEqual(identity_map["segment_0:player_1"], identity_map["segment_0:player_2"])

    def test_event_candidate_detection_emits_block_rebound_and_steal_candidates(self):
        service = self.make_service()
        records = [
            self.make_record(0, "shoot", 0, 15),
            self.make_record(1, "ball in hand", 40, 55),
            self.make_record(2, "defense", 56, 70),
            self.make_record(2, "dribble", 72, 87),
            self.make_record(2, "block", 100, 115),
            self.make_record(2, "block", 124, 139),
        ]
        candidates = service._detect_event_candidates(records)
        event_types = {candidate.event_type for candidate in candidates}
        self.assertIn("block_candidate", event_types)
        self.assertIn("rebound_candidate", event_types)
        self.assertIn("steal_candidate", event_types)

    def test_identity_duplicate_candidates_suggest_review_merge(self):
        service = self.make_service()
        summaries = [
            LongVideoPlayerSummaryResponse(
                player_id="segment_0:player_4",
                global_player_id="player_004",
                identity_confidence=0.25,
                segments_seen=1,
                clip_count=3,
                action_counts={"dribble": 2, "pass": 1},
                needs_review_count=0,
                average_confidence=0.7,
            ),
            LongVideoPlayerSummaryResponse(
                player_id="segment_2:player_6",
                global_player_id="player_006",
                identity_confidence=0.25,
                segments_seen=1,
                clip_count=3,
                action_counts={"dribble": 2, "pass": 1},
                needs_review_count=0,
                average_confidence=0.7,
            ),
        ]
        features = {
            "segment_0:player_4": PlayerIdentityFeatureResponse(
                player=4,
                segment_id=0,
                local_player_id="segment_0:player_4",
                start_frame=0,
                end_frame=100,
                appearance_signature={"h_mean": 0.05, "s_mean": 0.8, "v_mean": 0.7, "b_mean": 0.1, "g_mean": 0.25, "r_mean": 0.9},
                appearance_embedding=[0.8, 0.2, 0.1, 0.0],
                embedding_model="test_embedding",
                embedding_dim=4,
            ),
            "segment_2:player_6": PlayerIdentityFeatureResponse(
                player=6,
                segment_id=2,
                local_player_id="segment_2:player_6",
                start_frame=200,
                end_frame=300,
                appearance_signature={"h_mean": 0.06, "s_mean": 0.78, "v_mean": 0.72, "b_mean": 0.11, "g_mean": 0.24, "r_mean": 0.88},
                appearance_embedding=[0.79, 0.21, 0.1, 0.0],
                embedding_model="test_embedding",
                embedding_dim=4,
            ),
        }
        candidates = service._detect_identity_duplicate_candidates(summaries, features)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].left_global_player_id, "player_004")
        self.assertEqual(candidates[0].right_global_player_id, "player_006")
        self.assertGreater(candidates[0].confidence, 0.68)
        self.assertEqual(candidates[0].recommended_action, "review_merge")

    def test_identity_duplicate_candidates_respect_same_segment_conflict(self):
        service = self.make_service()
        summaries = [
            LongVideoPlayerSummaryResponse(
                player_id="segment_0:player_1",
                global_player_id="player_001",
                segments_seen=1,
                clip_count=3,
                action_counts={"dribble": 3},
                needs_review_count=0,
                average_confidence=0.7,
            ),
            LongVideoPlayerSummaryResponse(
                player_id="segment_0:player_2",
                global_player_id="player_002",
                segments_seen=1,
                clip_count=3,
                action_counts={"dribble": 3},
                needs_review_count=0,
                average_confidence=0.7,
            ),
        ]
        features = {
            "segment_0:player_1": PlayerIdentityFeatureResponse(
                player=1,
                segment_id=0,
                local_player_id="segment_0:player_1",
                start_frame=0,
                end_frame=100,
                appearance_signature={"h_mean": 0.2, "s_mean": 0.5, "v_mean": 0.7, "b_mean": 0.1, "g_mean": 0.2, "r_mean": 0.3},
                appearance_embedding=[0.8, 0.2, 0.1, 0.0],
                embedding_model="test_embedding",
                embedding_dim=4,
            ),
            "segment_0:player_2": PlayerIdentityFeatureResponse(
                player=2,
                segment_id=0,
                local_player_id="segment_0:player_2",
                start_frame=0,
                end_frame=100,
                appearance_signature={"h_mean": 0.2, "s_mean": 0.5, "v_mean": 0.7, "b_mean": 0.1, "g_mean": 0.2, "r_mean": 0.3},
                appearance_embedding=[0.8, 0.2, 0.1, 0.0],
                embedding_model="test_embedding",
                embedding_dim=4,
            ),
        }
        candidates = service._detect_identity_duplicate_candidates(summaries, features)
        self.assertEqual(candidates, [])

    def test_identity_duplicate_candidates_allow_same_frame_duplicate_boxes(self):
        service = self.make_service()
        summaries = [
            LongVideoPlayerSummaryResponse(
                player_id="segment_0:player_1",
                global_player_id="player_001",
                segments_seen=1,
                clip_count=3,
                action_counts={"dribble": 3},
                needs_review_count=0,
                average_confidence=0.7,
            ),
            LongVideoPlayerSummaryResponse(
                player_id="segment_0:player_2",
                global_player_id="player_002",
                segments_seen=1,
                clip_count=3,
                action_counts={"dribble": 3},
                needs_review_count=0,
                average_confidence=0.7,
            ),
        ]
        features = {
            "segment_0:player_1": PlayerIdentityFeatureResponse(
                player=1,
                segment_id=0,
                local_player_id="segment_0:player_1",
                start_frame=0,
                end_frame=100,
                appearance_signature={"h_mean": 0.2, "s_mean": 0.5, "v_mean": 0.7, "b_mean": 0.1, "g_mean": 0.2, "r_mean": 0.3},
                appearance_embedding=[0.8, 0.2, 0.1, 0.0],
                embedding_model="test_embedding",
                embedding_dim=4,
                sampled_boxes=[{"frame": 10, "x": 100, "y": 100, "w": 80, "h": 160}],
            ),
            "segment_0:player_2": PlayerIdentityFeatureResponse(
                player=2,
                segment_id=0,
                local_player_id="segment_0:player_2",
                start_frame=0,
                end_frame=100,
                appearance_signature={"h_mean": 0.2, "s_mean": 0.5, "v_mean": 0.7, "b_mean": 0.1, "g_mean": 0.2, "r_mean": 0.3},
                appearance_embedding=[0.8, 0.2, 0.1, 0.0],
                embedding_model="test_embedding",
                embedding_dim=4,
                sampled_boxes=[{"frame": 10, "x": 104, "y": 104, "w": 80, "h": 160}],
            ),
        }
        candidates = service._detect_identity_duplicate_candidates(summaries, features)
        self.assertEqual(len(candidates), 1)
        self.assertIn("bbox duplicate-overlap compatibility", " ".join(candidates[0].evidence))

    def test_identity_duplicate_candidates_reject_same_frame_separated_boxes(self):
        service = self.make_service()
        summaries = [
            LongVideoPlayerSummaryResponse(
                player_id="segment_0:player_1",
                global_player_id="player_001",
                segments_seen=1,
                clip_count=3,
                action_counts={"dribble": 3},
                needs_review_count=0,
                average_confidence=0.7,
            ),
            LongVideoPlayerSummaryResponse(
                player_id="segment_0:player_2",
                global_player_id="player_002",
                segments_seen=1,
                clip_count=3,
                action_counts={"dribble": 3},
                needs_review_count=0,
                average_confidence=0.7,
            ),
        ]
        features = {
            "segment_0:player_1": PlayerIdentityFeatureResponse(
                player=1,
                segment_id=0,
                local_player_id="segment_0:player_1",
                start_frame=0,
                end_frame=100,
                appearance_signature={"h_mean": 0.2, "s_mean": 0.5, "v_mean": 0.7, "b_mean": 0.1, "g_mean": 0.2, "r_mean": 0.3},
                appearance_embedding=[0.8, 0.2, 0.1, 0.0],
                embedding_model="test_embedding",
                embedding_dim=4,
                sampled_boxes=[{"frame": 10, "x": 100, "y": 100, "w": 80, "h": 160}],
            ),
            "segment_0:player_2": PlayerIdentityFeatureResponse(
                player=2,
                segment_id=0,
                local_player_id="segment_0:player_2",
                start_frame=0,
                end_frame=100,
                appearance_signature={"h_mean": 0.2, "s_mean": 0.5, "v_mean": 0.7, "b_mean": 0.1, "g_mean": 0.2, "r_mean": 0.3},
                appearance_embedding=[0.8, 0.2, 0.1, 0.0],
                embedding_model="test_embedding",
                embedding_dim=4,
                sampled_boxes=[{"frame": 10, "x": 400, "y": 100, "w": 80, "h": 160}],
            ),
        }
        candidates = service._detect_identity_duplicate_candidates(summaries, features)
        self.assertEqual(candidates, [])

    def test_offline_duplicate_report_recomputes_candidates(self):
        analysis = {
            "video": "example.mov",
            "identity_embedding_model": "test_embedding",
            "player_identity_features": [
                {
                    "player": 4,
                    "segment_id": 0,
                    "local_player_id": "segment_0:player_4",
                    "start_frame": 0,
                    "end_frame": 100,
                    "appearance_signature": {"h_mean": 0.05, "s_mean": 0.8, "v_mean": 0.7, "b_mean": 0.1, "g_mean": 0.25, "r_mean": 0.9},
                    "appearance_embedding": [0.8, 0.2, 0.1, 0.0],
                    "embedding_model": "test_embedding",
                    "embedding_dim": 4,
                },
                {
                    "player": 6,
                    "segment_id": 2,
                    "local_player_id": "segment_2:player_6",
                    "start_frame": 200,
                    "end_frame": 300,
                    "appearance_signature": {"h_mean": 0.06, "s_mean": 0.78, "v_mean": 0.72, "b_mean": 0.11, "g_mean": 0.24, "r_mean": 0.88},
                    "appearance_embedding": [0.79, 0.21, 0.1, 0.0],
                    "embedding_model": "test_embedding",
                    "embedding_dim": 4,
                },
            ],
            "long_video": {
                "players": [
                    {
                        "player_id": "segment_0:player_4",
                        "global_player_id": "player_004",
                        "identity_confidence": 0.25,
                        "segments_seen": 1,
                        "clip_count": 3,
                        "action_counts": {"dribble": 2, "pass": 1},
                        "needs_review_count": 0,
                        "average_confidence": 0.7,
                    },
                    {
                        "player_id": "segment_2:player_6",
                        "global_player_id": "player_006",
                        "identity_confidence": 0.25,
                        "segments_seen": 1,
                        "clip_count": 3,
                        "action_counts": {"dribble": 2, "pass": 1},
                        "needs_review_count": 0,
                        "average_confidence": 0.7,
                    },
                ],
                "identity_duplicate_candidates": [],
            },
        }
        report = build_duplicate_report(analysis, source_path="analysis.json")
        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(report["candidate_source"], "recomputed_from_players_and_identity_features")
        self.assertEqual(report["candidates"][0]["recommended_action"], "review_merge")

    def test_temporal_smoothing_replaces_isolated_low_confidence_label(self):
        records = [
            {
                "player": 0,
                "clip_index": 0,
                "final": FinalDecisionResponse(
                    action_id=6, action="defense", confidence=0.7, source="r2plus1d", needs_review=False, reason=""
                ),
            },
            {
                "player": 0,
                "clip_index": 1,
                "final": FinalDecisionResponse(
                    action_id=3, action="dribble", confidence=0.3, source="r2plus1d", needs_review=True, reason=""
                ),
            },
            {
                "player": 0,
                "clip_index": 2,
                "final": FinalDecisionResponse(
                    action_id=6, action="defense", confidence=0.8, source="r2plus1d", needs_review=False, reason=""
                ),
            },
        ]
        predictions = {0: {0: 6, 1: 3, 2: 6}}
        apply_temporal_smoothing(records, predictions, confidence_threshold=0.6)
        self.assertEqual(records[1]["final"].action, "defense")
        self.assertEqual(predictions[0], {0: 6, 1: 6, 2: 6})

    def test_tracker_failure_fallback(self):
        from unittest.mock import patch, MagicMock
        with patch('cv2.VideoCapture') as mock_vc, \
             patch('cv2.legacy.MultiTracker_create') as mock_mt:
            
            mock_cap = MagicMock()
            mock_cap.read.side_effect = [
                (True, np.zeros((100, 100, 3), dtype=np.uint8)),
                (True, np.zeros((100, 100, 3), dtype=np.uint8)),
                (True, np.zeros((100, 100, 3), dtype=np.uint8)),
                (False, None)
            ]
            mock_vc.return_value = mock_cap
            
            mock_tracker = MagicMock()
            mock_tracker.update.side_effect = [
                (True, [(10, 10, 20, 20)]),
                (False, []),
                (True, [(30, 30, 20, 20)]),
            ]
            mock_mt.return_value = mock_tracker
            
            from app.analysis.tracking import extract_tracked_frames
            frames, player_boxes, w, h, colors = extract_tracked_frames(
                video_path="dummy.mp4",
                tracker_type="CSRT",
                headless=True,
                boxes=[(5, 5, 20, 20)]
            )
            
            self.assertEqual(len(frames), 3)
            self.assertEqual(len(player_boxes), 3)
            self.assertEqual(player_boxes[0], ((5.0, 5.0, 20.0, 20.0),))
            self.assertEqual(player_boxes[1], ((10.0, 10.0, 20.0, 20.0),))
            self.assertEqual(player_boxes[2], ((10.0, 10.0, 20.0, 20.0),))

    def test_crop_windows_n_clips_boundary(self):
        frames_17 = [np.zeros((10, 10, 3)) for _ in range(17)]
        boxes_17 = [[(0, 0, 5, 5)]] * 17
        windows_17 = crop_windows(frames_17, boxes_17, seq_length=16, vid_stride=8)
        self.assertEqual(len(windows_17[0]), 2)

        frames_25 = [np.zeros((10, 10, 3)) for _ in range(25)]
        boxes_25 = [[(0, 0, 5, 5)]] * 25
        windows_25 = crop_windows(frames_25, boxes_25, seq_length=16, vid_stride=8)
        self.assertEqual(len(windows_25[0]), 3)

    def test_vlm_verifier_parses_response_or_thinking(self):
        import json
        from unittest.mock import patch, MagicMock
        from app.analysis.vlm import OllamaVLMVerifier
        from app.analysis.schemas import MotionFeatures
        
        verifier = OllamaVLMVerifier(model="test-model", host="http://localhost:11434")
        
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "response": '{"action": "shoot", "confidence": 0.95, "reason": "visible shot"}',
                "thinking": 'Let me think about this. The player is shooting...'
            }).encode('utf-8')
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            
            motion = MotionFeatures(
                avg_center_speed=1.0, max_center_speed=2.0, avg_box_area=100.0, area_change_ratio=1.0
            )
            prediction = self.make_prediction("shoot", 0.95)
            
            frames = [np.zeros((128, 176, 3), dtype=np.uint8)]
            vlm_decision = verifier.verify(frames, prediction, motion)
            
            self.assertTrue(vlm_decision.available)
            self.assertEqual(vlm_decision.action, "shoot")
            self.assertEqual(vlm_decision.confidence, 0.95)
            self.assertEqual(vlm_decision.reason, "visible shot")

            mock_resp.read.return_value = json.dumps({
                "thinking": '{"action": "shoot", "confidence": 0.95}'
            }).encode('utf-8')
            
            vlm_decision2 = verifier.verify(frames, prediction, motion)
            self.assertEqual(vlm_decision2.action, "shoot")
            self.assertEqual(vlm_decision2.confidence, 0.95)

    def test_vlm_audit_uses_configured_image_width(self):
        import json
        from unittest.mock import patch, MagicMock
        from app.analysis.vlm import OllamaVLMVerifier

        verifier = OllamaVLMVerifier(model="test-model", host="http://localhost:11434", image_width=160)
        frames = [np.zeros((128, 176, 3), dtype=np.uint8)]

        with patch("app.analysis.vlm.encode_frames_jpeg", return_value=["encoded"]) as mock_encode, \
             patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {
                    "thinking": (
                        '{"player_count_min": 1, "player_count_max": 2, '
                        '"actions": ["dribble"], "confidence": 0.8}'
                    )
                }
            ).encode("utf-8")
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            audit = verifier.audit_video_frames(frames, scope="0.0s-1.0s")

        mock_encode.assert_called_once_with(frames, max_width=160)
        self.assertTrue(audit.available)
        self.assertEqual(audit.player_count_min, 1)

    def test_lifespan_mounts_static_directories_with_absolute_paths(self):
        from unittest.mock import patch, MagicMock
        from app.main import lifespan
        from fastapi import FastAPI
        
        app = FastAPI()
        
        with patch('app.main.build_r2plus1d_model'), \
             patch('app.main.init_globals'), \
             patch('os.makedirs') as mock_makedirs, \
             patch('os.path.isdir', return_value=True), \
             patch('os.path.abspath') as mock_abspath, \
             patch.object(app, 'mount') as mock_mount, \
             patch('app.main.get_settings') as mock_get_settings:
            
            mock_settings = MagicMock()
            mock_settings.output_dir = "rel_output"
            mock_settings.video_output_dir = "rel_video_output"
            mock_get_settings.return_value = mock_settings
            
            mock_abspath.side_effect = lambda x: f"/abs/{x}"
            
            import anyio
            async def run_lifespan():
                async with lifespan(app):
                    pass
            
            anyio.run(run_lifespan)
            
            mock_abspath.assert_any_call("rel_output")
            mock_abspath.assert_any_call("rel_video_output")
            mock_makedirs.assert_any_call("/abs/rel_output", exist_ok=True)
            mock_makedirs.assert_any_call("/abs/rel_video_output", exist_ok=True)
            
            self.assertEqual(mock_mount.call_count, 2)
            first_call_args = mock_mount.call_args_list[0]
            second_call_args = mock_mount.call_args_list[1]
            
            self.assertEqual(first_call_args[0][0], "/static/outputs")
            self.assertEqual(first_call_args[0][1].directory, "/abs/rel_output")
            self.assertEqual(second_call_args[0][0], "/static/videos")
            self.assertEqual(second_call_args[0][1].directory, "/abs/rel_video_output")

    def test_crop_video_resize_failure_idx_zero(self):
        import cv2
        from unittest.mock import patch
        from app.analysis.tracking import crop_video

        clip = [np.zeros((10, 10, 3), dtype=np.uint8)]
        crop_window = [[(0, 0, 5, 5)]]
        
        with patch('cv2.resize', side_effect=cv2.error("Mocked resize error")):
            result = crop_video(clip, crop_window, player=0, output_size=(128, 176))
            
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (176, 128, 3))
        self.assertTrue(np.all(result[0] == 0))

    def test_motion_features_invalid_player_index(self):
        from app.analysis.motion import compute_motion_features
        player_boxes = [[[(0.0, 0.0, 10.0, 10.0)]]]
        with self.assertRaises(IndexError):
            compute_motion_features(
                player_boxes=player_boxes,
                player=2,
                clip_index=0,
                seq_length=1,
                vid_stride=1,
            )

    def test_write_annotated_video_player_count_mismatch(self):
        from app.video.writer import write_annotated_video
        import tempfile
        import shutil
        import os

        temp_dir = tempfile.mkdtemp()
        try:
            video_path = os.path.join(temp_dir, "test_out.mp4")
            video_frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
            player_boxes = [[
                (10, 10, 20, 20),
                (30, 30, 20, 20),
                (50, 50, 20, 20),
            ]]
            predictions = {
                0: {0: 1},
                2: {0: 3},
                5: {0: 4},
            }
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            
            write_annotated_video(
                video_path=video_path,
                video_frames=video_frames,
                player_boxes=player_boxes,
                predictions=predictions,
                colors=colors,
                frame_width=100,
                frame_height=100,
                vid_stride=8,
                fps=30.0,
            )
            self.assertTrue(os.path.exists(video_path))
        finally:
            shutil.rmtree(temp_dir)

    def test_writer_clip_index_overflow(self):
        from app.video.writer import write_annotated_video
        import tempfile
        import shutil
        import os

        temp_dir = tempfile.mkdtemp()
        try:
            video_path = os.path.join(temp_dir, "test_overflow.mp4")
            video_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(20)]
            player_boxes = [[(10, 10, 20, 20)]] * 20
            
            predictions_list = {
                0: [1, 2]
            }
            
            predictions_dict = {
                0: {0: 1, 1: 3}
            }
            
            colors = [(255, 0, 0)]
            
            write_annotated_video(
                video_path=video_path,
                video_frames=video_frames,
                player_boxes=player_boxes,
                predictions=predictions_list,
                colors=colors,
                frame_width=100,
                frame_height=100,
                vid_stride=8,
                fps=30.0,
            )
            self.assertTrue(os.path.exists(video_path))
            
            write_annotated_video(
                video_path=video_path,
                video_frames=video_frames,
                player_boxes=player_boxes,
                predictions=predictions_dict,
                colors=colors,
                frame_width=100,
                frame_height=100,
                vid_stride=8,
                fps=30.0,
            )
            self.assertTrue(os.path.exists(video_path))
        finally:
            shutil.rmtree(temp_dir)

    def test_path_traversal_video_path(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from unittest.mock import patch, MagicMock
        
        with patch('app.main.build_r2plus1d_model', return_value=MagicMock()):
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/analysis/run",
                    json={
                        "video_path": "../suspicious_file.mp4",
                        "vlm_mode": "off",
                    }
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("Access denied", response.json()["detail"])

    def test_video_path_allows_configured_external_root(self):
        from app.analysis.router import _video_path_is_allowed
        from app.config import Settings

        settings = Settings(allowed_video_roots="/Users/example/Movies,/mnt/videos")

        self.assertTrue(_video_path_is_allowed("/Users/example/Movies/game.mov", settings))
        self.assertTrue(_video_path_is_allowed("/mnt/videos/game.mov", settings))
        self.assertFalse(_video_path_is_allowed("/Users/example/Documents/game.mov", settings))

    def test_temporal_smoothing_non_contiguous_indices(self):
        records = [
            {
                "player": 0,
                "clip_index": 0,
                "final": FinalDecisionResponse(
                    action_id=6, action="defense", confidence=0.7, source="r2plus1d", needs_review=False, reason=""
                ),
            },
            {
                "player": 0,
                "clip_index": 4,
                "final": FinalDecisionResponse(
                    action_id=3, action="dribble", confidence=0.3, source="r2plus1d", needs_review=True, reason=""
                ),
            },
            {
                "player": 0,
                "clip_index": 8,
                "final": FinalDecisionResponse(
                    action_id=6, action="defense", confidence=0.8, source="r2plus1d", needs_review=False, reason=""
                ),
            },
        ]
        predictions = {0: {0: 6, 4: 3, 8: 6}}
        apply_temporal_smoothing(records, predictions, confidence_threshold=0.6)
        self.assertEqual(records[1]["final"].action, "defense")
        self.assertEqual(predictions[0][4], 6)

    def test_video_capture_release_on_error(self):
        from unittest.mock import patch, MagicMock
        import cv2
        from app.analysis.service import AnalysisService
        from app.analysis.schemas import AnalysisRequest
        from app.config import Settings

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = Exception("Mock CAP error")

        with patch('cv2.VideoCapture', return_value=mock_cap), \
             patch('app.analysis.service.extract_tracked_frames') as mock_etf, \
             patch('app.analysis.service.crop_windows') as mock_cw, \
             patch('app.analysis.service.predict_player_clips') as mock_ppc, \
             patch('app.analysis.service.write_annotated_video') as mock_wav:

             mock_etf.return_value = ([], {}, 100, 100, [])
             mock_cw.return_value = {}
             mock_ppc.return_value = {}

             settings = Settings(video_output_dir="dummy_out")
             service = AnalysisService(settings=settings, model=MagicMock(), device="cpu")

             request = AnalysisRequest(
                 video_path="dummy.mp4",
                 vlm_mode="off",
                 generate_video=True,
                 segmented_analysis=False,
             )

             with self.assertRaises(Exception) as context:
                 service.run_analysis(request)

             self.assertIn("Mock CAP error", str(context.exception))
             mock_cap.release.assert_called_once()

    def test_fuse_vlm_action_unknown_label(self):
        from app.analysis.fusion import fuse_decision
        from app.analysis.schemas import ModelPrediction, VLMDecisionResponse

        prediction = ModelPrediction(
            action_id=6,
            action="defense",
            confidence=0.5,
            probabilities={"defense": 0.5}
        )
        vlm = VLMDecisionResponse(
            available=True,
            action="unknown_vlm_action_name",
            confidence=0.9,
            needs_review=False,
            reason="VLM proposed an action not in LABEL_TO_ID",
            visible_ball=False,
            raw_response="{}",
        )

        decision = fuse_decision(
            prediction=prediction,
            vlm=vlm,
            high_confidence=0.8,
            low_confidence=0.4
        )

        self.assertEqual(decision.action_id, prediction.action_id)
        self.assertEqual(decision.action, prediction.action)
        self.assertEqual(decision.confidence, prediction.confidence)
        self.assertEqual(decision.source, "r2plus1d")
        self.assertIn("VLM returned unknown action label", decision.reason)

    def test_path_traversal_symlink(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from unittest.mock import patch, MagicMock

        with patch('app.main.build_r2plus1d_model', return_value=MagicMock()):
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/analysis/run",
                    json={
                        "video_path": "/tmp/outside_file.mp4",
                        "vlm_mode": "off",
                    }
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("Access denied", response.json()["detail"])

    def test_end_frame_clamped(self):
        from unittest.mock import patch, MagicMock
        import numpy as np
        from app.analysis.service import AnalysisService
        from app.analysis.schemas import AnalysisRequest
        from app.config import Settings
        from app.analysis.schemas import ModelPrediction

        dummy_frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(5)]

        with patch('app.analysis.service.extract_tracked_frames') as mock_etf, \
             patch('app.analysis.service.crop_windows') as mock_cw, \
             patch('app.analysis.service.predict_player_clips') as mock_ppc:

             mock_etf.return_value = (dummy_frames, [((0.0, 0.0, 10.0, 10.0),)] * 5, 100, 100, [])
             mock_cw.return_value = {0: [np.zeros((16, 10, 10, 3))]}
             mock_ppc.return_value = {0: [ModelPrediction(action_id=0, action="run", confidence=0.9, probabilities={"run": 0.9})]}

             settings = Settings(seq_length=16, vid_stride=8)
             service = AnalysisService(settings=settings, model=MagicMock(), device="cpu")

             request = AnalysisRequest(
                 video_path="dummy.mp4",
                 vlm_mode="off",
                 generate_video=False,
                 segmented_analysis=False,
             )

             response = service.run_analysis(request)
             self.assertEqual(len(response.records), 1)
             self.assertEqual(response.records[0].end_frame, 4)

    def test_writer_unknown_action_id(self):
        from unittest.mock import patch, MagicMock
        import numpy as np
        from app.video.writer import write_annotated_video

        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        boxes = [[(10.0, 10.0, 20.0, 20.0)]]
        predictions = {0: {0: 999}}
        colors = [(255, 0, 0)]

        with patch('cv2.VideoWriter') as mock_writer, \
             patch('cv2.putText') as mock_put_text:

             mock_out = MagicMock()
             mock_writer.return_value = mock_out

             write_annotated_video(
                 video_path="dummy_out.mp4",
                 video_frames=frames,
                 player_boxes=boxes,
                 predictions=predictions,
                 colors=colors,
                 frame_width=100,
                 frame_height=100,
                 vid_stride=8
             )

             mock_put_text.assert_called()
             called_args = mock_put_text.call_args[0]
             self.assertEqual(called_args[1], "unknown")

    def test_temporal_smoothing_dict_not_list(self):
        from app.analysis.fusion import apply_temporal_smoothing
        from app.analysis.schemas import FinalDecisionResponse

        records = [
            {
                "player": 0,
                "clip_index": 0,
                "final": FinalDecisionResponse(
                    action_id=6, action="defense", confidence=0.7, source="r2plus1d", needs_review=False, reason=""
                ),
            },
            {
                "player": 0,
                "clip_index": 1,
                "final": FinalDecisionResponse(
                    action_id=3, action="dribble", confidence=0.3, source="r2plus1d", needs_review=True, reason=""
                ),
            },
            {
                "player": 0,
                "clip_index": 2,
                "final": FinalDecisionResponse(
                    action_id=6, action="defense", confidence=0.8, source="r2plus1d", needs_review=False, reason=""
                ),
            },
        ]
        predictions = {0: {0: 6, 1: 3, 2: 6}}
        apply_temporal_smoothing(records, predictions, confidence_threshold=0.6)
        self.assertEqual(records[1]["final"].action, "defense")
        self.assertEqual(predictions[0][1], 6)


    def test_async_analysis_flow(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from unittest.mock import patch, MagicMock
        from app.analysis.task_manager import get_task_manager
        from app.analysis.schemas import AnalysisResponse, Size2D, AnalysisSummaryResponse
        
        task_manager = get_task_manager()
        
        with patch('app.main.build_r2plus1d_model', return_value=MagicMock()), \
             patch('os.path.exists', return_value=True):
            
            with TestClient(app) as client:
                with patch('app.analysis.service.AnalysisService.run_analysis') as mock_run:
                    dummy_response = AnalysisResponse(
                        video="examples/lebron_shoots.mp4",
                        created_at_unix=123456789.0,
                        runtime_seconds=1.5,
                        frame_size=Size2D(width=640, height=480),
                        seq_length=16,
                        vid_stride=8,
                        vlm_mode="off",
                        ollama_model=None,
                        records=[],
                        summary=AnalysisSummaryResponse(
                            clip_count=0,
                            action_counts={},
                            needs_review_count=0,
                            source_counts={}
                        )
                    )
                    mock_run.return_value = dummy_response
                    
                    response = client.post(
                        "/api/v1/analysis/run",
                        json={
                            "video_path": "examples/lebron_shoots.mp4",
                            "vlm_mode": "off",
                            "generate_video": False
                        }
                    )
                    self.assertEqual(response.status_code, 200)
                    data = response.json()
                    self.assertIn("task_id", data)
                    self.assertEqual(data["status"], "pending")
                    
                    task_id = data["task_id"]
                    
                    import time
                    # Small wait to ensure background task completes or is polled safely
                    time.sleep(0.5)
                    
                    status_response = client.get(f"/api/v1/analysis/status/{task_id}")
                    self.assertEqual(status_response.status_code, 200)
                    status_data = status_response.json()
                    self.assertEqual(status_data["task_id"], task_id)
                    self.assertIn(status_data["status"], ["pending", "processing", "completed"])

    def test_long_video_segment_ranges(self):
        from unittest.mock import MagicMock
        from app.analysis.service import AnalysisService
        from app.config import Settings

        service = AnalysisService(settings=Settings(), model=MagicMock(), device="cpu")
        ranges = service._build_segment_ranges(
            duration_sec=31.0,
            fps=10.0,
            frame_count=310,
            segment_duration_sec=10.0,
            segment_overlap_sec=2.0,
            max_segments=None,
        )

        self.assertEqual(len(ranges), 4)
        self.assertEqual(ranges[0]["start_frame"], 0)
        self.assertEqual(ranges[0]["end_frame"], 99)
        self.assertEqual(ranges[1]["start_frame"], 80)
        self.assertEqual(ranges[-1]["end_frame"], 309)

    def test_long_video_vlm_audit_detects_player_under_count(self):
        from unittest.mock import MagicMock
        from app.analysis.service import AnalysisService
        from app.config import Settings
        from app.analysis.schemas import AnalysisSummaryResponse, VLMVideoAuditResponse

        service = AnalysisService(settings=Settings(), model=MagicMock(), device="cpu")
        status, notes = service._compare_segment_with_vlm(
            player_count=1,
            summary=AnalysisSummaryResponse(
                clip_count=3,
                action_counts={"dribble": 2, "walk": 1},
                needs_review_count=0,
                source_counts={"r2plus1d": 3},
            ),
            vlm_audit=VLMVideoAuditResponse(
                available=True,
                player_count_min=5,
                player_count_max=10,
                actions=["dribble", "pass"],
                confidence=0.9,
            ),
        )

        self.assertEqual(status, "fail_player_under_count")
        self.assertIn("VLM saw at least 5", notes[0])

    def test_segmented_analysis_preserves_vlm_mode_inside_segments(self):
        from unittest.mock import MagicMock, patch
        from app.analysis.service import AnalysisService
        from app.config import Settings
        from app.analysis.schemas import AnalysisResponse, AnalysisRequest, AnalysisSummaryResponse, Size2D

        service = AnalysisService(settings=Settings(), model=MagicMock(), device="cpu")
        request = AnalysisRequest(
            video_path="examples/lebron_shoots.mp4",
            generate_video=False,
            segmented_analysis=True,
            vlm_mode="low-confidence",
            vlm_audit=False,
        )
        segment_response = AnalysisResponse(
            video="segment.mp4",
            created_at_unix=1.0,
            runtime_seconds=0.1,
            frame_size=Size2D(width=64, height=64),
            seq_length=16,
            vid_stride=8,
            vlm_mode="low-confidence",
            ollama_model="test-model",
            records=[],
            summary=AnalysisSummaryResponse(
                clip_count=0,
                action_counts={},
                needs_review_count=0,
                source_counts={},
            ),
            player_identity_features=[],
        )

        with patch.object(
            service,
            "_read_video_metadata",
            return_value={"fps": 10.0, "frame_count": 20, "duration_sec": 2.0, "width": 64, "height": 64},
        ), patch.object(
            service,
            "_build_segment_ranges",
            return_value=[{"segment_id": 0, "start_sec": 0.0, "end_sec": 2.0, "start_frame": 0, "end_frame": 19}],
        ), patch.object(service, "_write_video_segment", return_value="missing-temp-segment.mp4"), patch.object(
            service,
            "_run_single_analysis",
            return_value=segment_response,
        ) as mock_single:
            service.run_long_video_analysis(request)

        segment_request = mock_single.call_args.args[0]
        self.assertEqual(segment_request.vlm_mode, "low-confidence")
        self.assertFalse(segment_request.segmented_analysis)

    def test_segmented_analysis_emits_intermediate_progress(self):
        from unittest.mock import MagicMock, patch
        from app.analysis.service import AnalysisService
        from app.config import Settings
        from app.analysis.schemas import AnalysisResponse, AnalysisRequest, AnalysisSummaryResponse, Size2D

        service = AnalysisService(settings=Settings(), model=MagicMock(), device="cpu")
        request = AnalysisRequest(
            video_path="examples/lebron_shoots.mp4",
            generate_video=False,
            segmented_analysis=True,
            vlm_mode="off",
            vlm_audit=False,
        )
        segment_response = AnalysisResponse(
            video="segment.mp4",
            created_at_unix=1.0,
            runtime_seconds=0.1,
            frame_size=Size2D(width=64, height=64),
            seq_length=16,
            vid_stride=8,
            vlm_mode="off",
            ollama_model=None,
            records=[],
            summary=AnalysisSummaryResponse(
                clip_count=0,
                action_counts={},
                needs_review_count=0,
                source_counts={},
            ),
            player_identity_features=[],
        )
        progress_updates = []

        with patch.object(
            service,
            "_read_video_metadata",
            return_value={"fps": 10.0, "frame_count": 40, "duration_sec": 4.0, "width": 64, "height": 64},
        ), patch.object(
            service,
            "_build_segment_ranges",
            return_value=[
                {"segment_id": 0, "start_sec": 0.0, "end_sec": 2.0, "start_frame": 0, "end_frame": 19},
                {"segment_id": 1, "start_sec": 2.0, "end_sec": 4.0, "start_frame": 20, "end_frame": 39},
            ],
        ), patch.object(service, "_write_video_segment", return_value="missing-temp-segment.mp4"), patch.object(
            service,
            "_run_single_analysis",
            return_value=segment_response,
        ):
            service.run_long_video_analysis(request, progress_callback=progress_updates.append)

        self.assertIn(10, progress_updates)
        self.assertTrue(any(10 < progress < 100 for progress in progress_updates))
        self.assertGreaterEqual(max(progress_updates), 96)

    def test_run_analysis_defaults_to_segmented_mode(self):
        from unittest.mock import MagicMock, patch
        from app.analysis.service import AnalysisService
        from app.config import Settings
        from app.analysis.schemas import AnalysisRequest

        service = AnalysisService(settings=Settings(), model=MagicMock(), device="cpu")
        request = AnalysisRequest(video_path="dummy.mp4", generate_video=False)

        with patch.object(service, "run_long_video_analysis") as mock_long:
            service.run_analysis(request)

        mock_long.assert_called_once_with(request, progress_callback=None)

    def test_background_analysis_forwards_progress_callback(self):
        from app.analysis.router import bg_run_analysis
        from app.analysis.task_manager import TaskManager
        from app.analysis.schemas import AnalysisResponse, AnalysisRequest, AnalysisSummaryResponse, Size2D

        class FakeService:
            def run_analysis(self, request, progress_callback=None):
                progress_callback(42)
                return AnalysisResponse(
                    video=request.video_path,
                    created_at_unix=1.0,
                    runtime_seconds=0.1,
                    frame_size=Size2D(width=64, height=64),
                    seq_length=16,
                    vid_stride=8,
                    vlm_mode="off",
                    ollama_model=None,
                    records=[],
                    summary=AnalysisSummaryResponse(
                        clip_count=0,
                        action_counts={},
                        needs_review_count=0,
                        source_counts={},
                    ),
                    player_identity_features=[],
                )

        task_manager = TaskManager()
        task_id = task_manager.create_task()
        bg_run_analysis(
            task_id,
            AnalysisRequest(video_path="examples/lebron_shoots.mp4", vlm_mode="off", segmented_analysis=False),
            FakeService(),
            task_manager,
        )

        state = task_manager.get_task(task_id)
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.progress, 100)
        self.assertIsNotNone(state.result)

    def test_player_statistics_estimate_from_action_counts(self):
        from unittest.mock import MagicMock
        from collections import Counter
        from app.analysis.service import AnalysisService
        from app.config import Settings

        service = AnalysisService(settings=Settings(), model=MagicMock(), device="cpu")
        stats = service._estimate_player_statistics(Counter({"shoot": 2, "pass": 3, "block": 1}))

        self.assertEqual(stats.points, 0)
        self.assertEqual(stats.shot_attempts, 2)
        self.assertEqual(stats.point_candidate_count, 2)
        self.assertEqual(stats.assists, 3)
        self.assertEqual(stats.blocks, 0)
        self.assertEqual(stats.status, "estimate_requires_event_confirmation")
        self.assertIn("points", stats.candidate_fields)
        self.assertTrue(any("block_candidate" in note for note in stats.notes))
        self.assertEqual(stats.rebounds, 0)
        self.assertEqual(stats.steals, 0)

    def test_scoreboard_parser_uses_latest_readable_checkpoint(self):
        checkpoints = _parse_scoreboard_checkpoints(
            [
                {"index": 0, "visible": True, "left_score": 0, "right_score": 0, "confidence": 0.9},
                {"index": 1, "visible": False, "confidence": 0.1},
                {"index": 2, "visible": True, "left_score": 10, "right_score": 6, "confidence": 0.8},
            ],
            frame_times=[90.0, 600.0, 1020.0],
            frame_numbers=[2700, 18000, 30600],
            raw_response="{}",
        )

        final = _last_visible_scoreboard(checkpoints)

        self.assertIsNotNone(final)
        self.assertEqual(final.left_score, 10)
        self.assertEqual(final.right_score, 6)
        self.assertEqual(final.time_sec, 1020.0)

    def test_scoreboard_candidate_scoring_prefers_led_panel(self):
        service = self.make_service()
        frame = np.full((360, 640, 3), 210, dtype=np.uint8)
        cv2.rectangle(frame, (360, 130), (570, 230), (20, 20, 20), -1)
        cv2.putText(frame, "120", (375, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 40), 5)
        cv2.putText(frame, "96", (490, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 40), 5)
        cv2.putText(frame, "Q4", (440, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 3)

        score, box = service._score_scoreboard_candidate(frame)

        self.assertGreater(score, 150.0)
        self.assertIsNotNone(box)
        self.assertGreater(box[0] + box[2], 350)

    def test_scoreboard_candidate_scoring_rejects_large_solid_color_regions(self):
        service = self.make_service()
        frame = np.full((360, 640, 3), 220, dtype=np.uint8)
        cv2.rectangle(frame, (10, 60), (250, 190), (0, 0, 230), -1)
        cv2.circle(frame, (80, 100), 16, (0, 230, 255), -1)

        score, box = service._score_scoreboard_candidate(frame)

        self.assertEqual(score, 0.0)
        self.assertIsNone(box)

    def test_scoreboard_candidate_scoring_retains_multiple_led_panels(self):
        service = self.make_service()
        frame = np.full((360, 640, 3), 210, dtype=np.uint8)
        for left in (30, 390):
            cv2.rectangle(frame, (left, 130), (left + 200, 230), (20, 20, 20), -1)
            cv2.putText(frame, "12", (left + 15, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 40), 5)
            cv2.putText(frame, "9", (left + 135, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (60, 255, 60), 5)
            cv2.putText(frame, "Q4", (left + 70, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 240), 2)

        candidates = service._score_scoreboard_candidates(frame, max_candidates=4)

        centers = sorted(box[0] + box[2] / 2.0 for _, box in candidates)
        self.assertGreaterEqual(len(centers), 2)
        self.assertLess(centers[0], 250)
        self.assertGreater(centers[-1], 390)

    def test_scoreboard_candidate_scoring_recovers_occluded_led_panel(self):
        service = self.make_service()
        frame = np.full((360, 640, 3), 220, dtype=np.uint8)
        cv2.rectangle(frame, (370, 125), (565, 225), (25, 25, 25), -1)
        cv2.rectangle(frame, (462, 110), (474, 245), (210, 210, 210), -1)
        cv2.putText(frame, "108", (378, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 5)
        cv2.putText(frame, "90", (500, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 5)
        cv2.putText(frame, "A", (390, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 3)
        cv2.putText(frame, "B", (530, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 3)
        cv2.line(frame, (440, 145), (495, 145), (0, 0, 255), 3)
        cv2.line(frame, (440, 155), (495, 155), (0, 255, 255), 3)

        candidates = service._score_scoreboard_candidates(frame, max_candidates=4)

        self.assertTrue(candidates)
        self.assertTrue(any(box[0] >= 350 and box[0] + box[2] <= 590 for _, box in candidates))

    def test_scoreboard_vlm_gate_rejects_rolling_shutter_bar(self):
        service = self.make_service()
        rolling = np.zeros((420, 1200, 3), dtype=np.uint8)
        cv2.rectangle(rolling, (220, 165), (980, 255), (255, 255, 0), -1)

        clear = np.zeros((420, 1200, 3), dtype=np.uint8)
        cv2.rectangle(clear, (220, 165), (420, 255), (255, 255, 0), -1)
        cv2.rectangle(clear, (790, 165), (970, 255), (255, 255, 0), -1)

        self.assertFalse(service._scoreboard_crop_has_separable_score_digits(rolling))
        self.assertTrue(service._scoreboard_crop_has_separable_score_digits(clear))

    def test_identity_continuity_uses_time_aligned_overlap_boxes(self):
        from app.analysis.schemas import PlayerIdentityFeatureResponse

        service = self.make_service()
        left = PlayerIdentityFeatureResponse(
            player=0,
            start_frame=0,
            end_frame=100,
            first_center=[100.0, 100.0],
            last_center=[900.0, 500.0],
            sampled_boxes=[
                {"time_sec": 27.0, "x": 300.0, "y": 200.0, "w": 100.0, "h": 220.0, "cx": 350.0, "cy": 310.0},
                {"time_sec": 29.5, "x": 700.0, "y": 220.0, "w": 100.0, "h": 220.0, "cx": 750.0, "cy": 330.0},
            ],
        )
        matching = PlayerIdentityFeatureResponse(
            player=1,
            start_frame=90,
            end_frame=190,
            first_center=[0.0, 0.0],
            last_center=[100.0, 100.0],
            sampled_boxes=[
                {"time_sec": 27.05, "x": 304.0, "y": 202.0, "w": 100.0, "h": 220.0, "cx": 354.0, "cy": 312.0},
                {"time_sec": 29.55, "x": 704.0, "y": 222.0, "w": 100.0, "h": 220.0, "cx": 754.0, "cy": 332.0},
            ],
        )
        different = matching.model_copy(
            update={
                "sampled_boxes": [
                    {"time_sec": 27.05, "x": 1200.0, "y": 200.0, "w": 100.0, "h": 220.0, "cx": 1250.0, "cy": 310.0},
                ]
            }
        )

        self.assertGreater(service._track_continuity_similarity(left, matching), 0.90)
        self.assertLess(service._track_continuity_similarity(left, different), 0.10)

    def test_identity_stitch_does_not_merge_without_timed_overlap_evidence(self):
        service = self.make_service()
        summaries = [
            LongVideoPlayerSummaryResponse(
                player_id=f"segment_{segment}:player_2",
                segments_seen=1,
                clip_count=5,
                action_counts={"run": 5},
                needs_review_count=0,
                average_confidence=0.8,
            )
            for segment in (0, 1)
        ]
        features = {
            "segment_0:player_2": PlayerIdentityFeatureResponse(
                player=2,
                segment_id=0,
                local_player_id="segment_0:player_2",
                start_frame=0,
                end_frame=100,
                first_center=[100.0, 100.0],
                last_center=[100.0, 100.0],
                appearance_embedding=[1.0, 0.0],
                sampled_boxes=[{"time_sec": 20.0, "x": 50.0, "y": 50.0, "w": 50.0, "h": 100.0, "cx": 75.0, "cy": 100.0}],
            ),
            "segment_1:player_2": PlayerIdentityFeatureResponse(
                player=2,
                segment_id=1,
                local_player_id="segment_1:player_2",
                start_frame=90,
                end_frame=190,
                first_center=[100.0, 100.0],
                last_center=[100.0, 100.0],
                appearance_embedding=[1.0, 0.0],
                sampled_boxes=[{"time_sec": 32.0, "x": 50.0, "y": 50.0, "w": 50.0, "h": 100.0, "cx": 75.0, "cy": 100.0}],
            ),
        }

        identity_map, _, _ = service._merge_segment_local_identities(summaries, features)

        self.assertNotEqual(identity_map["segment_0:player_2"], identity_map["segment_1:player_2"])

    def test_identity_stitch_allows_quality_sface_without_overlap_track(self):
        service = self.make_service()
        summaries = [
            LongVideoPlayerSummaryResponse(
                player_id=f"segment_{segment}:player_2",
                segments_seen=1,
                clip_count=5,
                action_counts={"run": 5},
                needs_review_count=0,
                average_confidence=0.8,
            )
            for segment in (0, 1)
        ]
        shared_face = [1.0, 0.0, 0.0]
        features = {
            "segment_0:player_2": PlayerIdentityFeatureResponse(
                player=2,
                segment_id=0,
                local_player_id="segment_0:player_2",
                start_frame=0,
                end_frame=100,
                appearance_embedding=[1.0, 0.0],
                appearance_signature={"jersey_dark_ratio": 0.82},
                face_embedding=shared_face,
                face_embedding_model="opencv_sface_test",
                face_sample_count=3,
                face_embedding_quality=0.80,
                sampled_boxes=[{"time_sec": 20.0, "cx": 75.0, "cy": 100.0}],
            ),
            "segment_1:player_2": PlayerIdentityFeatureResponse(
                player=2,
                segment_id=1,
                local_player_id="segment_1:player_2",
                start_frame=90,
                end_frame=190,
                appearance_embedding=[1.0, 0.0],
                appearance_signature={"jersey_dark_ratio": 0.78},
                face_embedding=shared_face,
                face_embedding_model="opencv_sface_test",
                face_sample_count=2,
                face_embedding_quality=0.75,
                sampled_boxes=[{"time_sec": 32.0, "cx": 75.0, "cy": 100.0}],
            ),
        }

        identity_map, _, evidence = service._merge_segment_local_identities(summaries, features)

        self.assertEqual(identity_map["segment_0:player_2"], identity_map["segment_1:player_2"])
        self.assertIn("SFace similarity", " ".join(evidence["segment_1:player_2"]))

    def test_identity_stitch_rejects_sface_match_with_different_jersey_darkness(self):
        service = self.make_service()
        summaries = [
            LongVideoPlayerSummaryResponse(
                player_id=f"segment_{segment}:player_2",
                segments_seen=1,
                clip_count=5,
                action_counts={"run": 5},
                needs_review_count=0,
                average_confidence=0.8,
            )
            for segment in (0, 1)
        ]
        features = {
            f"segment_{segment}:player_2": PlayerIdentityFeatureResponse(
                player=2,
                segment_id=segment,
                local_player_id=f"segment_{segment}:player_2",
                start_frame=segment * 100,
                end_frame=segment * 100 + 99,
                appearance_embedding=[1.0, 0.0],
                appearance_signature={"jersey_dark_ratio": darkness},
                face_embedding=[1.0, 0.0, 0.0],
                face_embedding_model="opencv_sface_test",
                face_sample_count=3,
                face_embedding_quality=0.80,
                sampled_boxes=[{"time_sec": time_sec, "cx": 75.0, "cy": 100.0}],
            )
            for segment, darkness, time_sec in ((0, 0.90, 20.0), (1, 0.10, 32.0))
        }

        identity_map, _, _ = service._merge_segment_local_identities(summaries, features)

        self.assertNotEqual(identity_map["segment_0:player_2"], identity_map["segment_1:player_2"])

    def test_scoreboard_candidate_selection_reserves_slots_for_latest_panels(self):
        service = self.make_service()
        candidates = [
            {"time_sec": 10.0, "score": 900.0, "box": (0, 0, 10, 10)},
            {"time_sec": 100.0, "score": 800.0, "box": (0, 0, 10, 10)},
            {"time_sec": 101.0, "score": 790.0, "box": (0, 0, 10, 10)},
            {"time_sec": 130.0, "score": 700.0, "box": (0, 0, 10, 10)},
            {"time_sec": 145.0, "score": 151.0, "box": (200, 100, 10, 10)},
        ]

        selected = service._select_scoreboard_candidates(candidates, max_frames=3)

        self.assertEqual([item["time_sec"] for item in selected], [100.0, 130.0, 145.0])

    def test_scoreboard_candidate_selection_uses_best_frame_in_late_time_cluster(self):
        service = self.make_service()
        candidates = [
            {"time_sec": 656.5, "score": 1400.0, "box": (100, 100, 200, 90)},
            {"time_sec": 819.5, "score": 999.0, "box": (110, 100, 200, 90)},
            {"time_sec": 823.5, "score": 224.0, "box": (120, 100, 200, 90)},
            {"time_sec": 836.0, "score": 1000.0, "box": (130, 100, 200, 90)},
            {"time_sec": 851.5, "score": 824.0, "box": (140, 100, 200, 90)},
        ]

        selected = service._select_scoreboard_candidates(candidates, max_frames=4)

        selected_times = [item["time_sec"] for item in selected]
        self.assertIn(819.5, selected_times)
        self.assertNotIn(823.5, selected_times)

    def test_scoreboard_candidate_selection_keeps_strong_terminal_before_weak_tail(self):
        service = self.make_service()
        candidates = [
            {"time_sec": 695.0, "score": 968.0, "box": (1400, 420, 240, 93)},
            *[
                {
                    "time_sec": float(time_sec),
                    "score": 2000.0 + index,
                    "box": (100 + index * 10, 300, 260, 100),
                }
                for index, time_sec in enumerate(range(100, 650, 25))
            ],
            *[
                {
                    "time_sec": float(time_sec),
                    "score": 180.0 + index,
                    "box": (300 + index * 10, 400, 220, 90),
                }
                for index, time_sec in enumerate((710, 720, 730, 740, 750, 760, 770, 780))
            ],
        ]

        selected = service._select_scoreboard_candidates(candidates, max_frames=8)
        selected_times = {float(candidate["time_sec"]) for candidate in selected}

        self.assertIn(695.0, selected_times)
        self.assertIn(780.0, selected_times)

    def test_scoreboard_prompt_explains_vertical_multi_digit_layout(self):
        verifier = OllamaVLMVerifier()

        prompt = verifier._build_scoreboard_prompt([695.1], "same-anchor phase comparison")

        self.assertIn("laid out horizontally OR vertically", prompt)
        self.assertIn("concatenate them from top to bottom", prompt)
        self.assertIn("left_score 69", prompt)
        self.assertIn("right_score 52", prompt)

    def test_scoreboard_prior_context_uses_reconciled_earlier_score(self):
        service = self.make_service()
        checkpoints = [
            ScoreboardCheckpointResponse(
                time_sec=589.3,
                frame=17679,
                visible=True,
                left_score=52,
                right_score=50,
                confidence=0.95,
                source="rapidocr_scoreboard_v1",
            ),
            ScoreboardCheckpointResponse(
                time_sec=589.3,
                frame=17680,
                visible=True,
                left_score=52,
                right_score=50,
                confidence=0.92,
                source="rapidocr_scoreboard_v1",
            ),
        ]

        context = service._scoreboard_prior_context(checkpoints, 695.1)

        self.assertIn("left 52, right 50", context)
        self.assertIn("cannot decrease", context)
        self.assertIn("physically implausible", context)

    def test_scoreboard_burst_spans_rolling_shutter_phase(self):
        service = self.make_service()

        self.assertEqual(service._scoreboard_burst_radius_frames(30.0), 18)
        self.assertEqual(service._scoreboard_burst_radius_frames(10.0), 6)

    def test_scoreboard_burst_prefers_fresh_panel_detection_during_camera_pan(self):
        service = self.make_service()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        fresh_box = (120, 20, 50, 30)
        stale_box = (10, 20, 50, 30)

        with patch.object(
            service,
            "_score_scoreboard_candidates",
            return_value=[(900.0, fresh_box)],
        ), patch.object(service, "_track_scoreboard_template", return_value=stale_box):
            resolved = service._resolve_scoreboard_burst_box(frame, frame, stale_box)

        self.assertEqual(resolved, fresh_box)

    def test_scoreboard_burst_keeps_stable_template_when_detection_is_nearby(self):
        service = self.make_service()
        frame = np.zeros((100, 300, 3), dtype=np.uint8)
        tracked_box = (100, 20, 50, 30)
        fresh_box = (115, 22, 50, 30)

        with patch.object(
            service,
            "_score_scoreboard_candidates",
            return_value=[(900.0, fresh_box)],
        ), patch.object(service, "_track_scoreboard_template", return_value=tracked_box):
            resolved = service._resolve_scoreboard_burst_box(frame, frame, tracked_box)

        self.assertEqual(resolved, tracked_box)

    def test_scoreboard_reconciliation_accepts_burst_consensus(self):
        service = self.make_service()
        checkpoints = [
            ScoreboardCheckpointResponse(time_sec=134.5, frame=4035, visible=True, left_score=120, right_score=96, confidence=0.95),
            ScoreboardCheckpointResponse(time_sec=134.52, frame=4036, visible=True, left_score=120, right_score=96, confidence=0.90),
            ScoreboardCheckpointResponse(time_sec=134.5, frame=4035, visible=True, left_score=20, right_score=19, confidence=0.80),
        ]

        summary = service._reconcile_scoreboard_checkpoints(checkpoints)

        self.assertEqual(summary.status, "ok")
        self.assertEqual(summary.final_left_score, 120)
        self.assertEqual(summary.final_right_score, 96)
        self.assertEqual(summary.final_total_points, 216)

    def test_scoreboard_reconciliation_rejects_implausible_late_ocr_jump(self):
        service = self.make_service()
        checkpoints = [
            ScoreboardCheckpointResponse(time_sec=588.6, frame=17658, visible=True, left_score=52, right_score=50, confidence=0.99, source="rapidocr_scoreboard_v1"),
            ScoreboardCheckpointResponse(time_sec=588.6, frame=17659, visible=True, left_score=52, right_score=50, confidence=0.98, source="rapidocr_scoreboard_v1"),
            ScoreboardCheckpointResponse(time_sec=694.7, frame=20841, visible=True, left_score=95, right_score=92, confidence=0.99, source="rapidocr_scoreboard_v1"),
            ScoreboardCheckpointResponse(time_sec=694.7, frame=20842, visible=True, left_score=95, right_score=92, confidence=0.98, source="rapidocr_scoreboard_v1"),
        ]

        summary = service._reconcile_scoreboard_checkpoints(checkpoints)

        self.assertEqual(summary.status, "inconsistent_scoreboard")
        self.assertIsNone(summary.final_left_score)

    def test_scoreboard_reconciliation_rejects_late_score_decrease(self):
        service = self.make_service()
        checkpoints = [
            ScoreboardCheckpointResponse(time_sec=409.4, frame=12282, visible=True, left_score=67, right_score=63, confidence=0.991, source="rapidocr_scoreboard_v1"),
            ScoreboardCheckpointResponse(time_sec=463.8, frame=13914, visible=True, left_score=11, right_score=65, confidence=0.99, source="rapidocr_scoreboard_v1"),
            ScoreboardCheckpointResponse(time_sec=463.8, frame=13915, visible=True, left_score=11, right_score=65, confidence=0.98, source="rapidocr_scoreboard_v1"),
        ]

        summary = service._reconcile_scoreboard_checkpoints(checkpoints)

        self.assertEqual(summary.status, "inconsistent_scoreboard")
        self.assertIsNone(summary.final_left_score)

    def test_scoreboard_reconciliation_rejects_single_read_and_score_decrease(self):
        service = self.make_service()
        single = [
            ScoreboardCheckpointResponse(time_sec=10.0, frame=300, visible=True, left_score=120, right_score=96, confidence=0.95),
        ]
        self.assertEqual(service._reconcile_scoreboard_checkpoints(single).status, "inconsistent_scoreboard")

        decreasing = [
            ScoreboardCheckpointResponse(time_sec=10.0, frame=300, visible=True, left_score=120, right_score=96, confidence=0.95, period="4", game_clock="3"),
            ScoreboardCheckpointResponse(time_sec=10.02, frame=301, visible=True, left_score=120, right_score=96, confidence=0.90, period="4", game_clock="3"),
            ScoreboardCheckpointResponse(time_sec=20.0, frame=600, visible=True, left_score=20, right_score=61, confidence=0.95, period="4", game_clock="3"),
            ScoreboardCheckpointResponse(time_sec=20.02, frame=601, visible=True, left_score=20, right_score=61, confidence=0.90, period="4", game_clock="3"),
        ]
        self.assertEqual(service._reconcile_scoreboard_checkpoints(decreasing).status, "inconsistent_scoreboard")

    def test_scoreboard_reconciliation_ignores_clock_increase_and_lower_quality_score(self):
        service = self.make_service()
        checkpoints = [
            ScoreboardCheckpointResponse(time_sec=149.9, frame=4497, visible=True, left_score=120, right_score=96, confidence=0.95, period="4", game_clock="3"),
            ScoreboardCheckpointResponse(time_sec=149.9, frame=4497, visible=True, left_score=120, right_score=96, confidence=0.90, period="4", game_clock="3"),
            ScoreboardCheckpointResponse(time_sec=152.1, frame=4563, visible=True, left_score=97, right_score=96, confidence=0.85, period="4", game_clock="35"),
            ScoreboardCheckpointResponse(time_sec=152.1, frame=4563, visible=True, left_score=97, right_score=96, confidence=0.80, period="4", game_clock="35"),
        ]

        summary = service._reconcile_scoreboard_checkpoints(checkpoints)

        self.assertEqual(summary.status, "ok")
        self.assertEqual((summary.final_left_score, summary.final_right_score), (120, 96))

    def test_scoreboard_reconciliation_rejects_weak_implausible_score_jump(self):
        service = self.make_service()
        checkpoints = [
            ScoreboardCheckpointResponse(time_sec=693.6, frame=20808, visible=True, left_score=99, right_score=80, confidence=0.95, period="4", game_clock="3"),
            ScoreboardCheckpointResponse(time_sec=693.6, frame=20808, visible=True, left_score=99, right_score=80, confidence=0.92, period="4", game_clock="3"),
            ScoreboardCheckpointResponse(time_sec=693.6, frame=20808, visible=True, left_score=99, right_score=80, confidence=0.90, period="4", game_clock="3"),
            ScoreboardCheckpointResponse(time_sec=706.0, frame=21180, visible=True, left_score=99, right_score=128, confidence=0.90, period="4", game_clock="24"),
            ScoreboardCheckpointResponse(time_sec=706.0, frame=21180, visible=True, left_score=99, right_score=128, confidence=0.87, period="4", game_clock="24"),
        ]

        summary = service._reconcile_scoreboard_checkpoints(checkpoints)

        self.assertEqual(summary.status, "ok")
        self.assertEqual((summary.final_left_score, summary.final_right_score), (99, 80))
        self.assertTrue(any("implausible" in note for note in summary.notes))

    def test_scoreboard_reconciliation_prefers_independent_cross_anchor_reads(self):
        service = self.make_service()
        checkpoints = [
            *[
                ScoreboardCheckpointResponse(time_sec=71.6, frame=2148, visible=True, left_score=24, right_score=16, confidence=0.85)
                for _ in range(4)
            ],
            ScoreboardCheckpointResponse(time_sec=638.0, frame=19140, visible=True, left_score=5, right_score=8, confidence=0.95),
            ScoreboardCheckpointResponse(time_sec=659.5, frame=19785, visible=True, left_score=5, right_score=8, confidence=0.90),
        ]

        summary = service._reconcile_scoreboard_checkpoints(checkpoints)

        self.assertEqual(summary.status, "ok")
        self.assertEqual((summary.final_left_score, summary.final_right_score), (5, 8))
        self.assertTrue(any("Cross-anchor consensus" in note for note in summary.notes))

    def test_scoreboard_reconciliation_accepts_plausible_later_score_with_slightly_lower_quality(self):
        service = self.make_service()
        checkpoints = [
            ScoreboardCheckpointResponse(time_sec=486.9, frame=14607, visible=True, left_score=6, right_score=6, confidence=0.95),
            ScoreboardCheckpointResponse(time_sec=486.9, frame=14607, visible=True, left_score=6, right_score=6, confidence=0.90),
            ScoreboardCheckpointResponse(time_sec=844.3, frame=25329, visible=True, left_score=21, right_score=19, confidence=0.85),
            ScoreboardCheckpointResponse(time_sec=844.3, frame=25329, visible=True, left_score=21, right_score=19, confidence=0.82),
        ]

        summary = service._reconcile_scoreboard_checkpoints(checkpoints)

        self.assertEqual((summary.final_left_score, summary.final_right_score), (21, 19))

    def test_scoreboard_reconciliation_prioritizes_consensus_ocr_over_vlm_conflict(self):
        service = self.make_service()
        checkpoints = [
            ScoreboardCheckpointResponse(time_sec=819.3, frame=24590, visible=True, left_score=38, right_score=36, confidence=0.985, source="rapidocr_scoreboard_v1"),
            ScoreboardCheckpointResponse(time_sec=819.3, frame=24590, visible=True, left_score=38, right_score=36, confidence=0.989, source="rapidocr_scoreboard_v1"),
            ScoreboardCheckpointResponse(time_sec=836.2, frame=25086, visible=True, left_score=12, right_score=35, confidence=0.95, source="vlm_scoreboard_burst_audit_v3"),
            ScoreboardCheckpointResponse(time_sec=836.2, frame=25086, visible=True, left_score=12, right_score=35, confidence=0.94, source="vlm_scoreboard_burst_audit_v3"),
        ]

        summary = service._reconcile_scoreboard_checkpoints(checkpoints)

        self.assertEqual(summary.status, "ok")
        self.assertEqual((summary.final_left_score, summary.final_right_score), (38, 36))
        self.assertTrue(any("deterministic OCR" in note for note in summary.notes))

    def test_scoreboard_reconciliation_uses_higher_confidence_sharpened_ocr_candidate(self):
        service = self.make_service()
        checkpoints = [
            ScoreboardCheckpointResponse(time_sec=692.4, frame=20798, visible=True, left_score=66, right_score=80, confidence=0.994, source="rapidocr_scoreboard_v1"),
            ScoreboardCheckpointResponse(time_sec=692.4, frame=20798, visible=True, left_score=99, right_score=80, confidence=0.996, source="rapidocr_scoreboard_v1"),
            ScoreboardCheckpointResponse(time_sec=692.4, frame=20798, visible=True, left_score=99, right_score=0, confidence=0.95, source="vlm_scoreboard_burst_audit_v3"),
            ScoreboardCheckpointResponse(time_sec=692.4, frame=20798, visible=True, left_score=99, right_score=0, confidence=0.94, source="vlm_scoreboard_burst_audit_v3"),
        ]

        summary = service._reconcile_scoreboard_checkpoints(checkpoints)

        self.assertEqual(summary.status, "ok")
        self.assertEqual((summary.final_left_score, summary.final_right_score), (99, 80))

    def test_scoreboard_reconciliation_fuses_reliable_sides_from_ocr_and_vlm(self):
        from unittest.mock import MagicMock
        from app.analysis.schemas import ScoreboardCheckpointResponse
        from app.analysis.service import AnalysisService
        from app.config import Settings

        service = AnalysisService(settings=Settings(), model=MagicMock(), device="cpu")
        checkpoints = [
            ScoreboardCheckpointResponse(time_sec=549.9, frame=16497, visible=True, left_score=92, right_score=77, confidence=0.85, source="vlm_scoreboard_burst_audit_v3"),
            ScoreboardCheckpointResponse(time_sec=549.9, frame=16494, visible=True, left_score=92, right_score=0, confidence=0.85, source="vlm_scoreboard_burst_audit_v3"),
            ScoreboardCheckpointResponse(time_sec=549.9, frame=16500, visible=True, left_score=92, right_score=0, confidence=0.84, source="vlm_scoreboard_burst_audit_v3"),
            ScoreboardCheckpointResponse(time_sec=692.4, frame=20772, visible=True, left_score=66, right_score=80, confidence=0.97, source="rapidocr_scoreboard_v1"),
            ScoreboardCheckpointResponse(time_sec=692.4, frame=20780, visible=True, left_score=99, right_score=0, confidence=0.85, source="vlm_scoreboard_burst_audit_v3"),
            ScoreboardCheckpointResponse(time_sec=692.4, frame=20781, visible=True, left_score=99, right_score=0, confidence=0.84, source="vlm_scoreboard_burst_audit_v3"),
            ScoreboardCheckpointResponse(time_sec=692.4, frame=20782, visible=True, left_score=99, right_score=None, confidence=0.85, source="vlm_scoreboard_burst_audit_v3"),
        ]

        summary = service._reconcile_scoreboard_checkpoints(checkpoints)

        self.assertEqual(summary.status, "ok")
        self.assertEqual((summary.final_left_score, summary.final_right_score), (99, 80))
        self.assertTrue(
            any(checkpoint.source == "scoreboard_component_fusion_v1" for checkpoint in summary.checkpoints)
        )

    def test_scoreboard_reconciliation_rejects_repeated_field_misalignment(self):
        from app.analysis.schemas import ScoreboardCheckpointResponse

        service = self.make_service()
        checkpoints = [
            *[
                ScoreboardCheckpointResponse(time_sec=265.5, frame=7965, visible=True, left_score=106, right_score=90, confidence=0.98, source="rapidocr_scoreboard_v1")
                for _ in range(5)
            ],
            *[
                ScoreboardCheckpointResponse(time_sec=323.4, frame=9702, visible=True, left_score=120, right_score=130, confidence=0.85, source="vlm_scoreboard_burst_audit_v3")
                for _ in range(2)
            ],
            *[
                ScoreboardCheckpointResponse(time_sec=347.7, frame=10431, visible=True, left_score=108, right_score=90, confidence=0.75, source="vlm_scoreboard_burst_audit_v3")
                for _ in range(3)
            ],
            ScoreboardCheckpointResponse(time_sec=359.6, frame=10788, visible=True, left_score=120, right_score=130, confidence=0.85, source="vlm_scoreboard_burst_audit_v3"),
        ]

        summary = service._reconcile_scoreboard_checkpoints(checkpoints)

        self.assertEqual(summary.status, "ok")
        self.assertEqual((summary.final_left_score, summary.final_right_score), (108, 90))
        self.assertTrue(any("implausible" in note for note in summary.notes))

    def test_scoreboard_reconciliation_rejects_late_truncated_three_digit_score(self):
        from app.analysis.schemas import ScoreboardCheckpointResponse

        service = self.make_service()
        checkpoints = [
            *[
                ScoreboardCheckpointResponse(time_sec=348.1, frame=10443, visible=True, left_score=108, right_score=90, confidence=0.85)
                for _ in range(3)
            ],
            *[
                ScoreboardCheckpointResponse(time_sec=382.4, frame=11472, visible=True, left_score=10, right_score=90, confidence=0.95)
                for _ in range(5)
            ],
            *[
                ScoreboardCheckpointResponse(time_sec=384.5, frame=11535, visible=True, left_score=10, right_score=90, confidence=0.96)
                for _ in range(6)
            ],
        ]

        summary = service._reconcile_scoreboard_checkpoints(checkpoints)

        self.assertEqual(summary.status, "ok")
        self.assertEqual((summary.final_left_score, summary.final_right_score), (108, 90))
        self.assertTrue(any("truncated score" in note for note in summary.notes))


    def test_scoreboard_conflict_selects_latest_distinct_candidates(self):
        service = self.make_service()
        checkpoints = [
            ScoreboardCheckpointResponse(time_sec=120.0, frame=3600, visible=True, left_score=118, right_score=94, confidence=0.9),
            ScoreboardCheckpointResponse(time_sec=150.1, frame=4503, visible=True, left_score=120, right_score=98, confidence=0.95),
            ScoreboardCheckpointResponse(time_sec=150.1, frame=4503, visible=True, left_score=120, right_score=96, confidence=0.95),
        ]

        conflict = service._latest_scoreboard_conflict(checkpoints)

        self.assertEqual(conflict, (150.1, [(120, 98), (120, 96)]))

    def test_identity_graph_summary_counts_review_nodes(self):
        service = self.make_service()
        summaries = [
            LongVideoPlayerSummaryResponse(
                player_id="segment_0:player_1",
                global_player_id="player_001",
                segments_seen=1,
                clip_count=2,
                action_counts={"shoot": 2},
                needs_review_count=0,
                average_confidence=0.8,
            ),
            LongVideoPlayerSummaryResponse(
                player_id="segment_1:player_2",
                global_player_id="player_002",
                segments_seen=1,
                clip_count=2,
                action_counts={"shoot": 2},
                needs_review_count=0,
                average_confidence=0.8,
            ),
        ]
        duplicate = IdentityDuplicateCandidateResponse(
            left_global_player_id="player_001",
            right_global_player_id="player_002",
            confidence=0.7,
        )

        summary = service._build_identity_graph_summary(
            player_summaries=summaries,
            duplicate_candidates=[duplicate],
            confirmed_merges=[],
            merge_decisions=[],
        )

        self.assertEqual(summary.node_count, 2)
        self.assertEqual(summary.duplicate_candidate_count, 1)
        self.assertIn("review-oriented", summary.notes[0])


if __name__ == "__main__":
    unittest.main()
