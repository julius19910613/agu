from pathlib import Path

import cv2
import numpy as np

from scripts.build_player_markdown_reports import build_player_markdown_reports


def _write_sample_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (64, 64),
    )
    for index in range(4):
        frame = np.full((64, 64, 3), 30 + index * 20, dtype=np.uint8)
        cv2.rectangle(frame, (10, 10), (30, 40), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()


def _sample_analysis() -> dict:
    return {
        "frame_size": {"width": 64, "height": 64},
        "records": [
            {
                "start_frame": 2,
                "end_frame": 3,
                "global_player_id": "player_001",
                "final": {
                    "action": "shoot",
                    "confidence": 0.92,
                    "needs_review": False,
                },
            }
        ],
        "player_identity_features": [
            {
                "local_player_id": "segment_0:player_1",
                "start_frame": 2,
                "sampled_boxes": [
                    {"frame": 2, "x": 10, "y": 10, "w": 20, "h": 30},
                    {"frame": 3, "x": 11, "y": 10, "w": 20, "h": 30},
                ],
            }
        ],
        "long_video": {
            "duration_sec": 2.0,
            "segments": [{"segment_index": 0}],
            "players": [
                {
                    "player_id": "segment_0:player_1",
                    "global_player_id": "player_001",
                    "clip_count": 1,
                    "segments_seen": 1,
                    "needs_review_count": 0,
                    "action_counts": {"shoot": 1},
                    "statistics": {
                        "points": 2,
                        "assists": 0,
                        "rebounds": 0,
                        "blocks": 0,
                        "steals": 0,
                    },
                    "identity_confidence": 0.86,
                    "identity_evidence": ["synthetic identity evidence"],
                }
            ],
        },
    }


def _sample_analysis_with_duplicate_players() -> dict:
    analysis = _sample_analysis()
    analysis["records"].append(
        {
            "start_frame": 2,
            "end_frame": 3,
            "global_player_id": "player_002",
            "final": {
                "action": "shoot",
                "confidence": 0.88,
                "needs_review": False,
            },
        }
    )
    analysis["player_identity_features"].extend(
        [
            {
                "local_player_id": "segment_1:player_2",
                "start_frame": 2,
                "appearance_embedding": [0.99, 0.01, 0.0],
                "sampled_boxes": [
                    {"frame": 2, "x": 12, "y": 10, "w": 20, "h": 30},
                ],
            },
            {
                "local_player_id": "segment_2:player_3",
                "start_frame": 2,
                "appearance_embedding": [0.0, 1.0, 0.0],
                "sampled_boxes": [
                    {"frame": 2, "x": 32, "y": 10, "w": 20, "h": 30},
                ],
            },
        ]
    )
    analysis["player_identity_features"][0]["appearance_embedding"] = [1.0, 0.0, 0.0]
    analysis["long_video"]["players"].extend(
        [
            {
                "player_id": "segment_1:player_2",
                "global_player_id": "player_002",
                "clip_count": 1,
                "segments_seen": 1,
                "needs_review_count": 0,
                "action_counts": {"shoot": 1},
                "statistics": {
                    "points": 2,
                    "assists": 0,
                    "rebounds": 0,
                    "blocks": 0,
                    "steals": 0,
                },
                "identity_confidence": 0.84,
                "identity_evidence": ["duplicate synthetic identity evidence"],
            },
            {
                "player_id": "segment_2:player_3",
                "global_player_id": "player_003",
                "clip_count": 1,
                "segments_seen": 1,
                "needs_review_count": 0,
                "action_counts": {"pass": 1},
                "statistics": {
                    "points": 0,
                    "assists": 1,
                    "rebounds": 0,
                    "blocks": 0,
                    "steals": 0,
                },
                "identity_confidence": 0.72,
                "identity_evidence": ["different synthetic identity evidence"],
            },
        ]
    )
    return analysis


def test_build_player_markdown_reports_writes_per_player_assets(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_sample_video(video_path)

    summary = build_player_markdown_reports(
        analysis=_sample_analysis(),
        video_path=str(video_path),
        output_dir=str(tmp_path / "reports"),
        crops_per_player=2,
    )

    assert summary["player_count"] == 1
    assert summary["roster_player_count"] == 1
    assert Path(summary["index_markdown"]).exists()
    assert Path(summary["roster_json"]).exists()
    assert Path(summary["roster_markdown"]).exists()
    report = summary["reports"][0]
    player_markdown = Path(report["markdown"])
    markdown_text = player_markdown.read_text(encoding="utf-8")
    assert "# player_001" in markdown_text
    assert "Technical Statistics" in markdown_text
    assert "green box marks the player" in markdown_text
    assert "VLM Player Verification" in markdown_text
    assert "<video" in markdown_text
    assert Path(report["screenshot"]).exists()
    assert Path(report["contact_sheet"]).exists()
    assert Path(report["video"]).exists()
    roster = Path(summary["roster_json"]).read_text(encoding="utf-8")
    assert "team_or_side" in roster
    assert "jersey_number_candidate" in roster


def test_build_player_markdown_reports_filters_vlm_non_players(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_sample_video(video_path)

    def verifier(crops, global_id):
        assert crops
        assert global_id == "player_001"
        return {
            "status": "available",
            "is_player": False,
            "confidence": 0.05,
            "reason": "boxed target is not a basketball player",
        }

    summary = build_player_markdown_reports(
        analysis=_sample_analysis(),
        video_path=str(video_path),
        output_dir=str(tmp_path / "reports"),
        crops_per_player=2,
        vlm_player_filter=True,
        vlm_player_verifier=verifier,
    )

    assert summary["player_count"] == 0
    assert summary["filtered_player_count"] == 1
    assert summary["roster_player_count"] == 0
    roster_payload = Path(summary["roster_json"]).read_text(encoding="utf-8")
    assert '"player_count": 0' in roster_payload
    assert summary["filtered_players"][0]["global_player_id"] == "player_001"
    assert not (tmp_path / "reports" / "player_001.md").exists()


def test_build_player_markdown_reports_can_require_available_vlm_player(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_sample_video(video_path)

    summary = build_player_markdown_reports(
        analysis=_sample_analysis(),
        video_path=str(video_path),
        output_dir=str(tmp_path / "reports"),
        crops_per_player=2,
        vlm_player_filter=True,
        require_vlm_player=True,
        vlm_player_verifier=lambda crops, global_id: {
            "status": "unavailable",
            "is_player": True,
            "confidence": 0.0,
            "reason": "synthetic timeout",
        },
    )

    assert summary["player_count"] == 0
    assert summary["filtered_player_count"] == 1


def test_build_player_markdown_reports_dedupes_similar_global_players(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_sample_video(video_path)

    summary = build_player_markdown_reports(
        analysis=_sample_analysis_with_duplicate_players(),
        video_path=str(video_path),
        output_dir=str(tmp_path / "reports"),
        crops_per_player=1,
        dedupe_players=True,
        dedupe_similarity_threshold=0.95,
    )

    kept_ids = {report["global_player_id"] for report in summary["reports"]}
    assert kept_ids == {"player_001", "player_003"}
    assert summary["dedupe"]["dropped_players"][0]["global_player_id"] == "player_002"
    assert summary["dedupe"]["dropped_players"][0]["duplicate_of"] == "player_001"
    roster_entries = summary["roster_player_count"]
    assert roster_entries == 2
    roster_payload = Path(summary["roster_json"]).read_text(encoding="utf-8")
    assert "\"points\": 4" in roster_payload
    assert "\"merged_from_global_player_ids\": [" in roster_payload


def test_build_player_markdown_reports_reuses_vlm_cache(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    cache_path = tmp_path / "vlm-cache.json"
    _write_sample_video(video_path)
    calls = {"count": 0}

    def verifier(crops, global_id):
        calls["count"] += 1
        return {
            "status": "available",
            "is_player": True,
            "confidence": 0.88,
            "reason": "cached player verification",
        }

    first = build_player_markdown_reports(
        analysis=_sample_analysis(),
        video_path=str(video_path),
        output_dir=str(tmp_path / "reports-first"),
        crops_per_player=2,
        vlm_player_filter=True,
        vlm_cache_path=str(cache_path),
        vlm_player_verifier=verifier,
    )
    second = build_player_markdown_reports(
        analysis=_sample_analysis(),
        video_path=str(video_path),
        output_dir=str(tmp_path / "reports-second"),
        crops_per_player=2,
        vlm_player_filter=True,
        vlm_cache_path=str(cache_path),
        vlm_player_verifier=lambda crops, global_id: (_ for _ in ()).throw(AssertionError("cache miss")),
    )

    assert calls["count"] == 1
    assert first["player_count"] == 1
    assert second["player_count"] == 1
    assert second["reports"][0]["vlm_player_verification"]["cache_hit"] is True


def test_build_player_markdown_reports_can_filter_low_support_roster_players(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_sample_video(video_path)
    analysis = _sample_analysis_with_duplicate_players()
    analysis["long_video"]["players"].append(
        {
            "player_id": "segment_3:player_4",
            "global_player_id": "player_004",
            "clip_count": 1,
            "segments_seen": 1,
            "needs_review_count": 0,
            "action_counts": {"no_action": 1},
            "statistics": {
                "points": 0,
                "assists": 0,
                "rebounds": 0,
                "blocks": 0,
                "steals": 0,
            },
            "identity_confidence": 0.25,
            "identity_evidence": ["low support synthetic identity evidence"],
        }
    )
    analysis["player_identity_features"].append(
        {
            "local_player_id": "segment_3:player_4",
            "start_frame": 2,
            "track_coverage": 0.05,
            "appearance_signature": {
                "h_mean": 0.1,
                "s_mean": 0.1,
                "v_mean": 0.1,
                "b_mean": 0.1,
                "g_mean": 0.1,
                "r_mean": 0.1,
            },
            "sampled_boxes": [
                {"frame": 2, "x": 1, "y": 1, "w": 3, "h": 4},
            ],
        }
    )

    summary = build_player_markdown_reports(
        analysis=analysis,
        video_path=str(video_path),
        output_dir=str(tmp_path / "reports"),
        crops_per_player=1,
        min_roster_score=10.0,
    )

    assert summary["roster_player_count"] == 3
    assert summary["roster_score_filtered_players"][0]["global_player_id"] == "player_004"
