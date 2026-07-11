from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib import request
from urllib.error import URLError

import cv2
import numpy as np


STAT_KEYS = ["points", "assists", "rebounds", "blocks", "steals"]


def build_player_markdown_reports(
    analysis: Dict[str, Any],
    video_path: str,
    output_dir: str,
    max_players: Optional[int] = None,
    min_roster_score: float = 0.0,
    crops_per_player: int = 8,
    video_fps: float = 2.0,
    vlm_player_filter: bool = False,
    vlm_model: str = "qwen3-vl:4b",
    vlm_endpoint: Optional[str] = None,
    vlm_confidence_threshold: float = 0.55,
    vlm_timeout_sec: float = 45.0,
    vlm_concurrency: int = 1,
    vlm_cache_path: Optional[str] = None,
    vlm_progress: bool = False,
    require_vlm_player: bool = False,
    dedupe_players: bool = False,
    dedupe_similarity_threshold: float = 0.92,
    vlm_player_verifier: Optional[Callable[[List[np.ndarray], str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build one Markdown report per global player from an AGU analysis JSON."""
    output_root = Path(output_dir)
    assets_dir = output_root / "assets"
    output_root.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    long_video = analysis.get("long_video") or {}
    player_summaries = long_video.get("merged_players") or long_video.get("players") or []
    features = analysis.get("player_identity_features") or []
    records = analysis.get("records") or []

    local_to_global = {
        summary.get("player_id"): summary.get("global_player_id") or summary.get("player_id")
        for summary in player_summaries
        if summary.get("player_id")
    }
    grouped_features: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for feature in features:
        local_id = feature.get("local_player_id")
        global_id = local_to_global.get(local_id)
        if global_id:
            grouped_features[global_id].append(feature)

    grouped_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        global_id = record.get("global_player_id")
        if global_id:
            grouped_records[global_id].append(record)

    players = _aggregate_global_players(player_summaries)
    players.sort(key=lambda item: item["clip_count"], reverse=True)
    dedupe_summary: Dict[str, Any] = {
        "enabled": bool(dedupe_players),
        "similarity_threshold": float(dedupe_similarity_threshold),
        "dropped_players": [],
    }
    if dedupe_players:
        players, dedupe_summary["dropped_players"], merge_map = _dedupe_report_players(
            players=players,
            grouped_features=grouped_features,
            similarity_threshold=float(dedupe_similarity_threshold),
        )
        _apply_group_merge_map(grouped_features, merge_map)
        _apply_group_merge_map(grouped_records, merge_map)
    players = _sort_players_for_roster(
        players=players,
        grouped_features=grouped_features,
        frame_size=analysis.get("frame_size") or {},
    )
    dropped_for_score: List[Dict[str, Any]] = []
    if float(min_roster_score) > 0.0:
        kept_players: List[Dict[str, Any]] = []
        for player in players:
            score = _player_support_score(
                player=player,
                features=grouped_features.get(player["global_player_id"], []),
                frame_size=analysis.get("frame_size") or {},
            )
            if score >= float(min_roster_score):
                kept_players.append(player)
                continue
            dropped_for_score.append(
                {
                    "global_player_id": player["global_player_id"],
                    "score": round(score, 4),
                    "reason": "support score below the configured roster threshold",
                }
            )
        players = kept_players
    if max_players is not None:
        players = players[: max(0, int(max_players))]

    prepared_reports: List[Dict[str, Any]] = []
    for player in players:
        global_id = player["global_player_id"]
        safe_id = _safe_slug(global_id)
        player_assets_dir = assets_dir / safe_id
        player_assets_dir.mkdir(parents=True, exist_ok=True)

        crop_evidence = _extract_player_crops(
            video_path=video_path,
            features=grouped_features.get(global_id, []),
            max_crops=max(1, int(crops_per_player)),
        )
        crops = [item["image"] for item in crop_evidence]
        prepared_reports.append(
            {
                "player": player,
                "global_id": global_id,
                "safe_id": safe_id,
                "player_assets_dir": player_assets_dir,
                "crops": crops,
                "screenshot_path": player_assets_dir / "screenshot.jpg",
                "contact_sheet_path": player_assets_dir / "contact-sheet.jpg",
                "video_output_path": player_assets_dir / "evidence.mp4",
                "markdown_path": output_root / f"{safe_id}.md",
                "vlm_verification": None,
            }
        )

    filtered_players: List[Dict[str, Any]] = []
    if vlm_player_filter:
        _attach_vlm_player_verifications(
            prepared_reports=prepared_reports,
            model=vlm_model,
            endpoint=vlm_endpoint,
            confidence_threshold=vlm_confidence_threshold,
            timeout_sec=vlm_timeout_sec,
            concurrency=max(1, int(vlm_concurrency)),
            cache_path=Path(vlm_cache_path) if vlm_cache_path else output_root / "vlm-player-verification-cache.json",
            progress=vlm_progress,
            verifier=vlm_player_verifier,
        )

    reports: List[Dict[str, Any]] = []
    for prepared in prepared_reports:
        player = prepared["player"]
        global_id = prepared["global_id"]
        vlm_verification = prepared.get("vlm_verification")
        if _should_filter_vlm_player_report(
            enabled=vlm_player_filter,
            verification=vlm_verification,
            require_available_player=require_vlm_player,
        ):
            filtered_players.append(
                {
                    "global_player_id": global_id,
                    "reason": vlm_verification.get("reason") or "VLM reported that the boxed target is not a basketball player.",
                    "confidence": vlm_verification.get("confidence"),
                    "raw_response": vlm_verification.get("raw_response"),
                }
            )
            continue
        crops = prepared["crops"]
        screenshot_path = prepared["screenshot_path"]
        contact_sheet_path = prepared["contact_sheet_path"]
        video_output_path = prepared["video_output_path"]

        screenshot_written = _write_screenshot(crops, screenshot_path, global_id)
        contact_sheet_written = _write_contact_sheet(crops, contact_sheet_path, global_id)
        video_written = _write_crop_video(crops, video_output_path, global_id, fps=max(0.5, float(video_fps)))

        markdown_path = prepared["markdown_path"]
        markdown = _render_player_markdown(
            player=player,
            records=grouped_records.get(global_id, []),
            screenshot_path=_relative_path(screenshot_path, markdown_path.parent) if screenshot_written else None,
            contact_sheet_path=_relative_path(contact_sheet_path, markdown_path.parent) if contact_sheet_written else None,
            video_path=_relative_path(video_output_path, markdown_path.parent) if video_written else None,
            source_video=video_path,
            vlm_verification=vlm_verification,
        )
        markdown_path.write_text(markdown, encoding="utf-8")
        reports.append(
            {
                "global_player_id": global_id,
                "markdown": str(markdown_path),
                "screenshot": str(screenshot_path) if screenshot_written else None,
                "contact_sheet": str(contact_sheet_path) if contact_sheet_written else None,
                "video": str(video_output_path) if video_written else None,
                "clip_count": player["clip_count"],
                "stats": player["statistics"],
                "vlm_player_verification": vlm_verification,
            }
        )

    index_path = output_root / "index.md"
    index_path.write_text(_render_index_markdown(reports, analysis, video_path), encoding="utf-8")
    reported_player_ids = {report["global_player_id"] for report in reports}
    roster_players = [player for player in players if player["global_player_id"] in reported_player_ids]
    roster_entries = _build_roster_entries(
        players=roster_players,
        reports=reports,
        grouped_features=grouped_features,
        analysis=analysis,
    )
    roster_json_path = output_root / "roster-summary.json"
    roster_markdown_path = output_root / "roster-summary.md"
    roster_json_path.write_text(
        json.dumps(
            {
                "source_video": video_path,
                "player_count": len(roster_entries),
                "players": roster_entries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    roster_markdown_path.write_text(
        _render_roster_markdown(
            roster_entries=roster_entries,
            analysis=analysis,
            video_path=video_path,
        ),
        encoding="utf-8",
    )
    summary = {
        "output_dir": str(output_root),
        "index_markdown": str(index_path),
        "roster_json": str(roster_json_path),
        "roster_markdown": str(roster_markdown_path),
        "roster_player_count": len(roster_entries),
        "player_count": len(reports),
        "filtered_player_count": len(filtered_players),
        "filtered_players": filtered_players,
        "roster_score_filtered_players": dropped_for_score,
        "dedupe": dedupe_summary,
        "vlm_player_filter": bool(vlm_player_filter),
        "require_vlm_player": bool(require_vlm_player),
        "vlm_model": vlm_model if vlm_player_filter else None,
        "vlm_concurrency": max(1, int(vlm_concurrency)) if vlm_player_filter else None,
        "vlm_cache_path": str(Path(vlm_cache_path) if vlm_cache_path else output_root / "vlm-player-verification-cache.json") if vlm_player_filter else None,
        "source_video": video_path,
        "reports": reports,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _aggregate_global_players(player_summaries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for summary in player_summaries:
        global_id = summary.get("global_player_id") or summary.get("player_id")
        if not global_id:
            continue
        item = grouped.setdefault(
            global_id,
            {
                "global_player_id": global_id,
                "local_player_ids": [],
                "clip_count": 0,
                "segments_seen": set(),
                "needs_review_count": 0,
                "action_counts": Counter(),
                "statistics": Counter(),
                "identity_confidences": [],
                "identity_evidence": [],
            },
        )
        local_id = summary.get("player_id")
        if local_id:
            item["local_player_ids"].append(local_id)
            segment_id = _segment_id_from_local_id(local_id)
            if segment_id is not None:
                item["segments_seen"].add(segment_id)
        item["clip_count"] += int(summary.get("clip_count") or 0)
        item["needs_review_count"] += int(summary.get("needs_review_count") or 0)
        item["action_counts"].update(summary.get("action_counts") or {})
        stats = summary.get("statistics") or {}
        for key in STAT_KEYS:
            item["statistics"][key] += int(stats.get(key) or 0)
        confidence = summary.get("identity_confidence")
        if isinstance(confidence, (int, float)):
            item["identity_confidences"].append(float(confidence))
        item["identity_evidence"].extend(summary.get("identity_evidence") or [])

    aggregated: List[Dict[str, Any]] = []
    for item in grouped.values():
        confidences = item["identity_confidences"]
        aggregated.append(
            {
                "global_player_id": item["global_player_id"],
                "merged_from_global_player_ids": [item["global_player_id"]],
                "local_player_ids": item["local_player_ids"],
                "clip_count": item["clip_count"],
                "segments_seen": len(item["segments_seen"]),
                "needs_review_count": item["needs_review_count"],
                "action_counts": dict(item["action_counts"]),
                "statistics": {key: int(item["statistics"][key]) for key in STAT_KEYS},
                "identity_confidence_avg": sum(confidences) / len(confidences) if confidences else 0.0,
                "identity_evidence": item["identity_evidence"][:12],
            }
        )
    return aggregated


def _dedupe_report_players(
    players: List[Dict[str, Any]],
    grouped_features: Dict[str, List[Dict[str, Any]]],
    similarity_threshold: float,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
    accepted: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    merge_map: Dict[str, str] = {}
    for player in players:
        duplicate_of: Optional[Dict[str, Any]] = None
        duplicate_score = 0.0
        for existing in accepted:
            if _report_player_has_hard_conflict(
                grouped_features.get(player["global_player_id"], []),
                grouped_features.get(existing["global_player_id"], []),
            ):
                continue
            score = _report_player_similarity(
                grouped_features.get(player["global_player_id"], []),
                grouped_features.get(existing["global_player_id"], []),
            )
            if score >= similarity_threshold and score > duplicate_score:
                duplicate_of = existing
                duplicate_score = score
        if duplicate_of is None:
            accepted.append(_clone_player_summary(player))
            continue
        _merge_report_player_summary(duplicate_of, player)
        merge_map[player["global_player_id"]] = duplicate_of["global_player_id"]
        dropped.append(
            {
                "global_player_id": player["global_player_id"],
                "duplicate_of": duplicate_of["global_player_id"],
                "similarity": round(duplicate_score, 4),
                "reason": "appearance embeddings are highly similar in the human-facing report view",
            }
        )
    return accepted, dropped, merge_map


def _clone_player_summary(player: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **player,
        "merged_from_global_player_ids": list(player.get("merged_from_global_player_ids") or [player["global_player_id"]]),
        "local_player_ids": list(player.get("local_player_ids") or []),
        "action_counts": dict(player.get("action_counts") or {}),
        "statistics": _normalize_statistics(player.get("statistics") or {}),
        "identity_evidence": list(player.get("identity_evidence") or []),
    }


def _merge_report_player_summary(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    target["merged_from_global_player_ids"] = list(
        dict.fromkeys(
            [
                *(target.get("merged_from_global_player_ids") or [target["global_player_id"]]),
                *(source.get("merged_from_global_player_ids") or [source["global_player_id"]]),
            ]
        )
    )
    target["local_player_ids"] = list(
        dict.fromkeys([*(target.get("local_player_ids") or []), *(source.get("local_player_ids") or [])])
    )
    target["clip_count"] = int(target.get("clip_count") or 0) + int(source.get("clip_count") or 0)
    target["needs_review_count"] = int(target.get("needs_review_count") or 0) + int(source.get("needs_review_count") or 0)
    merged_segments = {
        sid for sid in (_segment_id_from_local_id(local_id) for local_id in target.get("local_player_ids") or []) if sid is not None
    }
    target["segments_seen"] = len(merged_segments)
    merged_actions = Counter(target.get("action_counts") or {})
    merged_actions.update(source.get("action_counts") or {})
    target["action_counts"] = dict(merged_actions)
    target["statistics"] = _merge_statistics(target.get("statistics") or {}, source.get("statistics") or {})
    target_clips = max(1, int(target.get("clip_count") or 0))
    source_clips = max(1, int(source.get("clip_count") or 0))
    combined_weight = target_clips + source_clips
    target["identity_confidence_avg"] = (
        float(target.get("identity_confidence_avg") or 0.0) * target_clips
        + float(source.get("identity_confidence_avg") or 0.0) * source_clips
    ) / max(1, combined_weight)
    target["identity_evidence"] = list(
        dict.fromkeys([*(target.get("identity_evidence") or []), *(source.get("identity_evidence") or [])])
    )[:16]


def _normalize_statistics(stats: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {key: int(stats.get(key) or 0) for key in STAT_KEYS}
    normalized["confidence"] = float(stats.get("confidence") or 0.0)
    normalized["method"] = str(stats.get("method") or "action_proxy_v1")
    normalized["notes"] = [str(note) for note in (stats.get("notes") or [])]
    return normalized


def _merge_statistics(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    left_stats = _normalize_statistics(left)
    right_stats = _normalize_statistics(right)
    merged = {key: left_stats[key] + right_stats[key] for key in STAT_KEYS}
    merged["confidence"] = max(float(left_stats.get("confidence") or 0.0), float(right_stats.get("confidence") or 0.0))
    merged["method"] = str(left_stats.get("method") or right_stats.get("method") or "action_proxy_v1")
    merged["notes"] = list(dict.fromkeys([*left_stats.get("notes", []), *right_stats.get("notes", [])]))
    return merged


def _apply_group_merge_map(grouped: Dict[str, List[Dict[str, Any]]], merge_map: Dict[str, str]) -> None:
    for source_id, target_id in merge_map.items():
        source_items = grouped.pop(source_id, [])
        if not source_items:
            continue
        grouped.setdefault(target_id, []).extend(source_items)


def _report_player_has_hard_conflict(
    left_features: List[Dict[str, Any]],
    right_features: List[Dict[str, Any]],
) -> bool:
    right_by_frame: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for feature in right_features:
        for box in feature.get("sampled_boxes") or []:
            right_by_frame[int(round(float(box.get("frame", -1))))].append(box)
    for feature in left_features:
        for box in feature.get("sampled_boxes") or []:
            frame = int(round(float(box.get("frame", -1))))
            for right_box in right_by_frame.get(frame, []):
                if _sample_box_iou(box, right_box) < 0.20:
                    return True
    return False


def _sample_box_iou(left: Dict[str, Any], right: Dict[str, Any]) -> float:
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


def _sort_players_for_roster(
    players: List[Dict[str, Any]],
    grouped_features: Dict[str, List[Dict[str, Any]]],
    frame_size: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return sorted(
        players,
        key=lambda player: (
            -_player_support_score(
                player=player,
                features=grouped_features.get(player["global_player_id"], []),
                frame_size=frame_size,
            ),
            -int(player.get("clip_count") or 0),
            player["global_player_id"],
        ),
    )


def _player_support_score(
    player: Dict[str, Any],
    features: List[Dict[str, Any]],
    frame_size: Dict[str, Any],
) -> float:
    metrics = _player_support_metrics(player=player, features=features, frame_size=frame_size)
    total_actions = max(1, sum(int(count) for count in (player.get("action_counts") or {}).values()))
    no_action_ratio = float((player.get("action_counts") or {}).get("no_action", 0)) / total_actions
    return (
        float(player.get("clip_count") or 0)
        + float(player.get("segments_seen") or 0) * 4.0
        + metrics["avg_track_coverage"] * 8.0
        + metrics["avg_box_area_ratio"] * 1200.0
        + metrics["avg_box_height_ratio"] * 6.0
        + min(1.0, float(player.get("identity_confidence_avg") or 0.0)) * 3.0
        + min(5.0, float(metrics["feature_count"])) * 0.35
        - no_action_ratio * 2.0
    )


def _player_support_metrics(
    player: Dict[str, Any],
    features: List[Dict[str, Any]],
    frame_size: Dict[str, Any],
) -> Dict[str, float]:
    track_coverages: List[float] = []
    box_area_ratios: List[float] = []
    box_height_ratios: List[float] = []
    frame_width = max(1, int(frame_size.get("width") or 1))
    frame_height = max(1, int(frame_size.get("height") or 1))
    frame_area = float(frame_width * frame_height)
    for feature in features:
        coverage = feature.get("track_coverage")
        if isinstance(coverage, (int, float)):
            track_coverages.append(max(0.0, min(1.0, float(coverage))))
        for box in feature.get("sampled_boxes") or []:
            area = max(0.0, float(box.get("w", 0.0))) * max(0.0, float(box.get("h", 0.0)))
            if frame_area > 0.0 and area > 0.0:
                box_area_ratios.append(area / frame_area)
            height_ratio = max(0.0, float(box.get("h", 0.0))) / max(1.0, float(frame_height))
            if height_ratio > 0.0:
                box_height_ratios.append(height_ratio)
    return {
        "avg_track_coverage": sum(track_coverages) / len(track_coverages) if track_coverages else 0.0,
        "avg_box_area_ratio": sum(box_area_ratios) / len(box_area_ratios) if box_area_ratios else 0.0,
        "avg_box_height_ratio": sum(box_height_ratios) / len(box_height_ratios) if box_height_ratios else 0.0,
        "feature_count": float(len(features)),
    }


def _report_player_similarity(
    left_features: List[Dict[str, Any]],
    right_features: List[Dict[str, Any]],
) -> float:
    embedding_scores: List[float] = []
    signature_scores: List[float] = []
    for left in left_features:
        for right in right_features:
            embedding_score = _cosine_similarity(left.get("appearance_embedding"), right.get("appearance_embedding"))
            if embedding_score > 0.0:
                embedding_scores.append(embedding_score)
            signature_score = _signature_similarity(left.get("appearance_signature"), right.get("appearance_signature"))
            if signature_score > 0.0:
                signature_scores.append(signature_score)
    if embedding_scores:
        embedding_scores.sort(reverse=True)
        top = embedding_scores[: min(5, len(embedding_scores))]
        return sum(top) / len(top)
    if signature_scores:
        signature_scores.sort(reverse=True)
        top = signature_scores[: min(5, len(signature_scores))]
        return sum(top) / len(top)
    return 0.0


def _cosine_similarity(left: Any, right: Any) -> float:
    if not isinstance(left, list) or not isinstance(right, list) or not left or not right:
        return 0.0
    length = min(len(left), len(right))
    left_values = np.array(left[:length], dtype=np.float32)
    right_values = np.array(right[:length], dtype=np.float32)
    left_norm = float(np.linalg.norm(left_values))
    right_norm = float(np.linalg.norm(right_values))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(left_values, right_values) / (left_norm * right_norm))))


def _signature_similarity(left: Any, right: Any) -> float:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return 0.0
    keys = ["h_mean", "s_mean", "v_mean", "b_mean", "g_mean", "r_mean"]
    left_values = np.array([float(left.get(key, 0.0) or 0.0) for key in keys], dtype=np.float32)
    right_values = np.array([float(right.get(key, 0.0) or 0.0) for key in keys], dtype=np.float32)
    distance = float(np.linalg.norm(left_values - right_values))
    return max(0.0, min(1.0, 1.0 - distance / 1.75))


def _extract_player_crops(
    video_path: str,
    features: List[Dict[str, Any]],
    max_crops: int,
) -> List[Dict[str, Any]]:
    crop_specs: List[Dict[str, Any]] = []
    for feature in features:
        for box in feature.get("sampled_boxes") or []:
            area = float(box.get("w", 0.0)) * float(box.get("h", 0.0))
            crop_specs.append({"feature": feature, "box": box, "area": area})
    crop_specs.sort(key=lambda item: item["area"], reverse=True)
    if not crop_specs:
        return []

    selected = _spread_crop_specs(crop_specs, max_crops=max_crops)
    cap = cv2.VideoCapture(video_path)
    crops: List[Dict[str, Any]] = []
    try:
        if not cap.isOpened():
            return []
        for spec in selected:
            feature = spec["feature"]
            box = spec["box"]
            frame_index = _sampled_box_absolute_frame(feature, box)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            crop_result = _crop_box_with_bbox(frame, box)
            if crop_result is None:
                continue
            crop, bbox = crop_result
            image = _label_crop(crop, bbox, f"{feature.get('local_player_id', '')} f{frame_index}")
            crops.append(
                {
                    "image": image,
                    "frame_index": frame_index,
                    "local_player_id": feature.get("local_player_id", ""),
                    "bbox": bbox,
                }
            )
    finally:
        cap.release()
    return crops


def _sampled_box_absolute_frame(feature: Dict[str, Any], box: Dict[str, Any]) -> int:
    frame_value = float(box.get("frame", 0.0))
    start_frame = float(feature.get("start_frame", 0.0) or 0.0)
    if start_frame > 0 and frame_value < start_frame:
        frame_value += start_frame
    return int(round(frame_value))


def _spread_crop_specs(crop_specs: List[Dict[str, Any]], max_crops: int) -> List[Dict[str, Any]]:
    if len(crop_specs) <= max_crops:
        return crop_specs
    by_local: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for spec in crop_specs:
        by_local[str(spec["feature"].get("local_player_id", ""))].append(spec)
    selected: List[Dict[str, Any]] = []
    for specs in by_local.values():
        if specs and len(selected) < max_crops:
            selected.append(specs[0])
    if len(selected) < max_crops:
        selected_ids = {id(spec) for spec in selected}
        for spec in crop_specs:
            if id(spec) in selected_ids:
                continue
            selected.append(spec)
            if len(selected) >= max_crops:
                break
    return selected[:max_crops]


def _crop_box_with_bbox(frame: np.ndarray, box: Dict[str, Any]) -> Optional[tuple[np.ndarray, tuple[int, int, int, int]]]:
    height, width = frame.shape[:2]
    x = float(box.get("x", 0.0))
    y = float(box.get("y", 0.0))
    w = float(box.get("w", 0.0))
    h = float(box.get("h", 0.0))
    if w <= 1.0 or h <= 1.0:
        return None
    padding = 0.10
    x1 = max(0, min(width - 1, int(round(x - w * padding))))
    y1 = max(0, min(height - 1, int(round(y - h * padding))))
    x2 = max(x1 + 1, min(width, int(round(x + w * (1.0 + padding)))))
    y2 = max(y1 + 1, min(height, int(round(y + h * (1.0 + padding)))))
    crop = frame[y1:y2, x1:x2]
    if not crop.size:
        return None
    relative_box = (
        max(0, int(round(x - x1))),
        max(0, int(round(y - y1))),
        min(crop.shape[1] - 1, int(round(x + w - x1))),
        min(crop.shape[0] - 1, int(round(y + h - y1))),
    )
    return crop, relative_box


def _label_crop(crop: np.ndarray, bbox: tuple[int, int, int, int], label: str) -> np.ndarray:
    source_h, source_w = crop.shape[:2]
    tile = cv2.resize(crop, (240, 320), interpolation=cv2.INTER_AREA)
    scale_x = 240.0 / max(1, source_w)
    scale_y = 320.0 / max(1, source_h)
    x1, y1, x2, y2 = bbox
    box = (
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    )
    cv2.rectangle(tile, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=3)
    cv2.rectangle(tile, (0, 0), (239, 28), (0, 0, 0), thickness=-1)
    cv2.putText(tile, label[:32], (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 292), (239, 319), (0, 0, 0), thickness=-1)
    cv2.putText(tile, "green box = detected player", (6, 311), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    return tile


def _write_screenshot(crops: List[np.ndarray], path: Path, global_id: str) -> bool:
    if not crops:
        return False
    image = crops[0].copy()
    cv2.putText(image, global_id, (8, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return bool(cv2.imwrite(str(path), image))


def _write_contact_sheet(crops: List[np.ndarray], path: Path, global_id: str) -> bool:
    if not crops:
        return False
    columns = min(4, max(1, len(crops)))
    rows = math.ceil(len(crops) / columns)
    tile_h, tile_w = crops[0].shape[:2]
    header_h = 42
    sheet = np.full((header_h + rows * tile_h, columns * tile_w, 3), 245, dtype=np.uint8)
    cv2.putText(sheet, f"{global_id} evidence crops", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    for index, crop in enumerate(crops):
        row = index // columns
        col = index % columns
        y1 = header_h + row * tile_h
        x1 = col * tile_w
        sheet[y1 : y1 + tile_h, x1 : x1 + tile_w] = crop
    return bool(cv2.imwrite(str(path), sheet))


def _write_crop_video(crops: List[np.ndarray], path: Path, global_id: str, fps: float) -> bool:
    if not crops:
        return False
    height, width = crops[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        return False
    try:
        for crop in crops:
            frame = crop.copy()
            cv2.putText(frame, global_id, (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            writer.write(frame)
    finally:
        writer.release()
    return path.exists() and path.stat().st_size > 0


def _attach_vlm_player_verifications(
    prepared_reports: List[Dict[str, Any]],
    model: str,
    endpoint: Optional[str],
    confidence_threshold: float,
    timeout_sec: float,
    concurrency: int,
    cache_path: Path,
    progress: bool,
    verifier: Optional[Callable[[List[np.ndarray], str], Dict[str, Any]]],
) -> None:
    cache = _load_vlm_cache(cache_path)
    pending: List[Dict[str, Any]] = []
    for prepared in prepared_reports:
        key = _vlm_cache_key(model=model, crops=prepared["crops"])
        prepared["vlm_cache_key"] = key
        cached = cache.get(key)
        if isinstance(cached, dict):
            prepared["vlm_verification"] = {**cached, "cache_hit": True}
        else:
            pending.append(prepared)

    total = len(prepared_reports)
    _progress_log(
        progress,
        f"VLM player filter: total={total}, cache_hits={total - len(pending)}, pending={len(pending)}, concurrency={concurrency}",
    )
    if not pending:
        return

    completed = total - len(pending)
    if verifier is not None:
        for prepared in pending:
            result = verifier(prepared["crops"], prepared["global_id"])
            result = {**result, "cache_hit": False}
            prepared["vlm_verification"] = result
            cache[prepared["vlm_cache_key"]] = result
            completed += 1
            _save_vlm_cache(cache_path, cache)
            _progress_log(progress, _format_vlm_progress(completed, total, prepared["global_id"], result))
        return

    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = {
            executor.submit(
                _verify_player_box_with_vlm,
                crops=prepared["crops"],
                global_id=prepared["global_id"],
                model=model,
                endpoint=endpoint,
                confidence_threshold=confidence_threshold,
                timeout_sec=timeout_sec,
            ): prepared
            for prepared in pending
        }
        for future in as_completed(futures):
            prepared = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive guard for long-running CLI use.
                result = {
                    "status": "unavailable",
                    "is_player": True,
                    "confidence": 0.0,
                    "reason": f"VLM worker failed; report was kept. {exc}",
                }
            result = {**result, "cache_hit": False}
            prepared["vlm_verification"] = result
            cache[prepared["vlm_cache_key"]] = result
            completed += 1
            _save_vlm_cache(cache_path, cache)
            _progress_log(progress, _format_vlm_progress(completed, total, prepared["global_id"], result))


def _format_vlm_progress(completed: int, total: int, global_id: str, result: Dict[str, Any]) -> str:
    status = result.get("status")
    is_player = result.get("is_player")
    confidence = result.get("confidence")
    cache_hit = result.get("cache_hit")
    return f"VLM player filter: {completed}/{total} {global_id} status={status} is_player={is_player} confidence={confidence} cache_hit={cache_hit}"


def _progress_log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def _should_filter_vlm_player_report(
    enabled: bool,
    verification: Optional[Dict[str, Any]],
    require_available_player: bool,
) -> bool:
    if not enabled:
        return False
    if not verification:
        return bool(require_available_player)
    status = verification.get("status")
    if status == "available":
        return not bool(verification.get("is_player", True))
    return bool(require_available_player)


def _load_vlm_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_vlm_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _vlm_cache_key(model: str, crops: List[np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(b"agu-player-box-vlm-filter-v2")
    digest.update(model.encode("utf-8"))
    if crops:
        ok, encoded = cv2.imencode(".jpg", crops[0])
        if ok:
            digest.update(encoded.tobytes())
    return digest.hexdigest()


def _render_player_markdown(
    player: Dict[str, Any],
    records: List[Dict[str, Any]],
    screenshot_path: Optional[str],
    contact_sheet_path: Optional[str],
    video_path: Optional[str],
    source_video: str,
    vlm_verification: Optional[Dict[str, Any]],
) -> str:
    stats = player["statistics"]
    action_counts = Counter(player["action_counts"])
    top_records = sorted(records, key=lambda record: float((record.get("final") or {}).get("confidence") or 0.0), reverse=True)[:12]
    lines = [
        f"# {player['global_player_id']}",
        "",
        f"- Source video: `{source_video}`",
        f"- Canonical global player IDs: `{', '.join(player.get('merged_from_global_player_ids') or [player['global_player_id']])}`",
        f"- Segment-local tracks: `{len(player['local_player_ids'])}`",
        f"- Segments seen: `{player['segments_seen']}`",
        f"- Clip count: `{player['clip_count']}`",
        f"- Needs review clips: `{player['needs_review_count']}`",
        f"- Average identity confidence: `{player['identity_confidence_avg']:.3f}`",
        "",
        "## Player Screenshot",
        "",
    ]
    lines.append(f"![{player['global_player_id']} screenshot]({screenshot_path})" if screenshot_path else "_No screenshot available._")
    lines.extend(["", "The green box marks the player detected by the traditional tracking/embedding pipeline.", ""])
    lines.extend(["", "## Evidence Contact Sheet", ""])
    lines.append(f"![{player['global_player_id']} contact sheet]({contact_sheet_path})" if contact_sheet_path else "_No contact sheet available._")
    lines.extend(["", "## Evidence Video", ""])
    if video_path:
        lines.extend([
            f'<video src="{video_path}" controls width="360"></video>',
            "",
            f"[Open evidence video]({video_path})",
        ])
    else:
        lines.append("_No evidence video available._")
    lines.extend(["", "## VLM Player Verification", ""])
    if vlm_verification:
        status = vlm_verification.get("status", "unknown")
        is_player = vlm_verification.get("is_player")
        confidence = vlm_verification.get("confidence")
        reason = vlm_verification.get("reason") or ""
        lines.extend(
            [
                f"- Status: `{status}`",
                f"- Box contains basketball player: `{is_player}`",
                f"- Confidence: `{confidence}`",
                f"- Reason: {reason or '_No reason provided._'}",
            ]
        )
    else:
        lines.append("_VLM player verification was not enabled for this report._")
    lines.extend(
        [
            "",
            "## Technical Statistics",
            "",
            "| Stat | Value |",
            "| --- | ---: |",
            f"| Points | {stats['points']} |",
            f"| Assists | {stats['assists']} |",
            f"| Rebounds | {stats['rebounds']} |",
            f"| Blocks | {stats['blocks']} |",
            f"| Steals | {stats['steals']} |",
            "",
            "## Action Counts",
            "",
            "| Action | Count |",
            "| --- | ---: |",
        ]
    )
    for action, count in action_counts.most_common():
        lines.append(f"| {action} | {count} |")
    lines.extend(["", "## Representative Clips", "", "| Start frame | End frame | Action | Confidence | Review |", "| ---: | ---: | --- | ---: | --- |"])
    for record in top_records:
        final = record.get("final") or {}
        lines.append(
            f"| {record.get('start_frame')} | {record.get('end_frame')} | {final.get('action')} | "
            f"{float(final.get('confidence') or 0.0):.3f} | {bool(final.get('needs_review'))} |"
        )
    lines.extend(["", "## Identity Evidence", ""])
    for evidence in player["identity_evidence"][:12]:
        lines.append(f"- {evidence}")
    return "\n".join(lines) + "\n"


def _render_index_markdown(reports: List[Dict[str, Any]], analysis: Dict[str, Any], video_path: str) -> str:
    long_video = analysis.get("long_video") or {}
    lines = [
        "# AGU Player Markdown Reports",
        "",
        f"- Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Source video: `{video_path}`",
        f"- Duration seconds: `{long_video.get('duration_sec', '')}`",
        f"- Segment count: `{len(long_video.get('segments') or [])}`",
        f"- Player reports: `{len(reports)}`",
        "- Canonical roster summary: [roster-summary.md](roster-summary.md)",
        "- Green boxes mark the player crop selected from traditional tracking/embedding evidence.",
        "",
        "| Player | Clips | Points | Assists | Rebounds | Blocks | Steals | VLM player check | Report |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for report in reports:
        stats = report["stats"]
        report_path = Path(report["markdown"]).name
        vlm_check = report.get("vlm_player_verification") or {}
        vlm_label = str(vlm_check.get("is_player")) if vlm_check else "not enabled"
        lines.append(
            f"| {report['global_player_id']} | {report['clip_count']} | {stats['points']} | {stats['assists']} | "
            f"{stats['rebounds']} | {stats['blocks']} | {stats['steals']} | {vlm_label} | [{report['global_player_id']}]({report_path}) |"
        )
    return "\n".join(lines) + "\n"


def _build_roster_entries(
    players: List[Dict[str, Any]],
    reports: List[Dict[str, Any]],
    grouped_features: Dict[str, List[Dict[str, Any]]],
    analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    frame_size = analysis.get("frame_size") or {}
    report_by_id = {report["global_player_id"]: report for report in reports}
    team_assignments = _infer_team_assignments(players, grouped_features)
    entries: List[Dict[str, Any]] = []
    for player in players:
        global_id = player["global_player_id"]
        stats = _normalize_statistics(player.get("statistics") or {})
        features = grouped_features.get(global_id, [])
        support_metrics = _player_support_metrics(player=player, features=features, frame_size=frame_size)
        support_score = _player_support_score(player=player, features=features, frame_size=frame_size)
        jersey = _best_jersey_number_candidate(features)
        team = team_assignments.get(global_id) or {
            "team_or_side": "未识别",
            "confidence": None,
            "method": "unavailable",
        }
        report = report_by_id.get(global_id, {})
        notes = list(dict.fromkeys([*(stats.get("notes") or []), *player.get("identity_evidence", [])]))[:8]
        if team["team_or_side"] == "未识别":
            notes.append("队伍或阵营未识别")
        if jersey["number"] == "未识别":
            notes.append("号码候选未识别")
        entries.append(
            {
                "global_player_id": global_id,
                "merged_from_global_player_ids": list(player.get("merged_from_global_player_ids") or [global_id]),
                "team_or_side": team["team_or_side"],
                "team_confidence": team["confidence"],
                "team_method": team["method"],
                "jersey_number_candidate": jersey["number"],
                "jersey_number_confidence": jersey["confidence"],
                "clip_count": int(player.get("clip_count") or 0),
                "segments_seen": int(player.get("segments_seen") or 0),
                "needs_review_count": int(player.get("needs_review_count") or 0),
                "points": int(stats["points"]),
                "rebounds": int(stats["rebounds"]),
                "assists": int(stats["assists"]),
                "steals": int(stats["steals"]),
                "blocks": int(stats["blocks"]),
                "confidence": round(min(0.99, max(0.0, float(player.get("identity_confidence_avg") or 0.0))), 4),
                "support_score": round(support_score, 4),
                "avg_track_coverage": round(support_metrics["avg_track_coverage"], 4),
                "avg_box_area_ratio": round(support_metrics["avg_box_area_ratio"], 5),
                "avg_box_height_ratio": round(support_metrics["avg_box_height_ratio"], 4),
                "notes": notes,
                "report_markdown": report.get("markdown"),
                "screenshot": report.get("screenshot"),
                "contact_sheet": report.get("contact_sheet"),
                "video": report.get("video"),
            }
        )
    return entries


def _infer_team_assignments(
    players: List[Dict[str, Any]],
    grouped_features: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    vectors: Dict[str, np.ndarray] = {}
    for player in players:
        global_id = player["global_player_id"]
        signatures = [
            feature.get("appearance_signature") or {}
            for feature in grouped_features.get(global_id, [])
            if isinstance(feature.get("appearance_signature"), dict)
        ]
        if not signatures:
            continue
        vector = np.array(
            [
                sum(float(signature.get("h_mean", 0.0) or 0.0) for signature in signatures) / len(signatures),
                sum(float(signature.get("s_mean", 0.0) or 0.0) for signature in signatures) / len(signatures),
                sum(float(signature.get("v_mean", 0.0) or 0.0) for signature in signatures) / len(signatures),
                sum(float(signature.get("b_mean", 0.0) or 0.0) for signature in signatures) / len(signatures),
                sum(float(signature.get("g_mean", 0.0) or 0.0) for signature in signatures) / len(signatures),
                sum(float(signature.get("r_mean", 0.0) or 0.0) for signature in signatures) / len(signatures),
            ],
            dtype=np.float32,
        )
        vectors[global_id] = vector
    if len(vectors) < 4:
        return {}

    player_ids = list(vectors)
    farthest_pair: Optional[tuple[str, str, float]] = None
    for index, left_id in enumerate(player_ids):
        for right_id in player_ids[index + 1 :]:
            distance = float(np.linalg.norm(vectors[left_id] - vectors[right_id]))
            if farthest_pair is None or distance > farthest_pair[2]:
                farthest_pair = (left_id, right_id, distance)
    if farthest_pair is None or farthest_pair[2] < 0.20:
        return {}

    centroids = [vectors[farthest_pair[0]].copy(), vectors[farthest_pair[1]].copy()]
    assignments: Dict[str, int] = {}
    for _ in range(6):
        buckets = {0: [], 1: []}
        for global_id, vector in vectors.items():
            distances = [float(np.linalg.norm(vector - centroid)) for centroid in centroids]
            bucket = 0 if distances[0] <= distances[1] else 1
            assignments[global_id] = bucket
            buckets[bucket].append(vector)
        for bucket, bucket_vectors in buckets.items():
            if bucket_vectors:
                centroids[bucket] = np.mean(np.stack(bucket_vectors), axis=0)

    result: Dict[str, Dict[str, Any]] = {}
    for global_id, vector in vectors.items():
        distances = [float(np.linalg.norm(vector - centroid)) for centroid in centroids]
        bucket = assignments[global_id]
        confidence = 1.0 - (distances[bucket] / max(0.01, distances[0] + distances[1]))
        result[global_id] = {
            "team_or_side": "阵营A" if bucket == 0 else "阵营B",
            "confidence": round(max(0.0, min(0.95, confidence)), 4),
            "method": "appearance_signature_cluster_v1",
        }
    return result


def _best_jersey_number_candidate(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores: Dict[str, Dict[str, float]] = {}
    for feature in features:
        for candidate in feature.get("jersey_number_candidates") or []:
            number = str(candidate.get("number") or "").strip()
            if not re.fullmatch(r"\d{1,2}", number):
                continue
            confidence = _safe_float(candidate.get("confidence"), 0.0)
            item = scores.setdefault(number, {"score": 0.0, "max_confidence": 0.0, "count": 0.0})
            item["score"] += confidence
            item["max_confidence"] = max(item["max_confidence"], confidence)
            item["count"] += 1.0
    if not scores:
        return {"number": "未识别", "confidence": None}
    best_number, best_stats = max(scores.items(), key=lambda item: (item[1]["score"], item[1]["max_confidence"], item[0]))
    confidence = max(best_stats["max_confidence"], best_stats["score"] / max(1.0, best_stats["count"]))
    return {"number": best_number, "confidence": round(min(0.99, confidence), 4)}


def _render_roster_markdown(
    roster_entries: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    video_path: str,
) -> str:
    long_video = analysis.get("long_video") or {}
    lines = [
        "# AGU Curated Roster Summary",
        "",
        f"- Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Source video: `{video_path}`",
        f"- Segment count: `{len(long_video.get('segments') or [])}`",
        f"- Final roster players: `{len(roster_entries)}`",
        "",
        "| Player ID | 阵营 | 号码候选 | 得分 | 篮板 | 助攻 | 抢断 | 盖帽 | 置信度 | Clips | 报告 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in roster_entries:
        report_link = Path(entry["report_markdown"]).name if entry.get("report_markdown") else ""
        report_cell = f"[详情]({report_link})" if report_link else "无"
        lines.append(
            f"| {entry['global_player_id']} | {entry['team_or_side']} | {entry['jersey_number_candidate']} | "
            f"{entry['points']} | {entry['rebounds']} | {entry['assists']} | {entry['steals']} | {entry['blocks']} | "
            f"{entry['confidence']:.2f} | {entry['clip_count']} | {report_cell} |"
        )
    lines.extend(["", "## Notes", ""])
    for entry in roster_entries:
        lines.append(f"### {entry['global_player_id']}")
        lines.append("")
        lines.append(f"- 阵营: `{entry['team_or_side']}`")
        lines.append(f"- 号码候选: `{entry['jersey_number_candidate']}`")
        lines.append(f"- 合并来源: `{', '.join(entry['merged_from_global_player_ids'])}`")
        lines.append(f"- Support score: `{entry['support_score']}`")
        lines.append(f"- 备注: {'; '.join(entry['notes']) if entry['notes'] else '无'}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _verify_player_box_with_vlm(
    crops: List[np.ndarray],
    global_id: str,
    model: str,
    endpoint: Optional[str],
    confidence_threshold: float,
    timeout_sec: float,
) -> Dict[str, Any]:
    if not crops:
        return {
            "status": "skipped",
            "is_player": True,
            "confidence": 0.0,
            "reason": "No boxed crop was available, so the report was kept.",
        }
    base_url = os.environ.get("BASKETBALL_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    api_endpoint = endpoint or f"{base_url}/api/generate"
    prompt = (
        "You are reviewing a basketball analysis report. The image contains a crop/contact tile from a video. "
        "A green box marks the target detected by a traditional tracking or embedding pipeline. "
        "Decide whether the green boxed target is a real basketball player. "
        "Return only compact JSON with keys: is_player boolean, confidence number from 0 to 1, reason string. "
        "If the box is empty, background, ball, court line, spectator, referee, or unreadable non-player, set is_player=false."
    )
    image_b64 = _encode_image_b64(crops[0])
    if not image_b64:
        return {
            "status": "unavailable",
            "is_player": True,
            "confidence": 0.0,
            "reason": "Failed to encode player crop; report was kept.",
        }
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }
    try:
        req = request.Request(
            api_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=max(1.0, float(timeout_sec))) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "status": "unavailable",
            "is_player": True,
            "confidence": 0.0,
            "reason": f"VLM request failed; report was kept. {exc}",
        }

    raw_response = str(body.get("response") or "")
    parsed = _parse_json_object(raw_response)
    if not parsed:
        return {
            "status": "unavailable",
            "is_player": True,
            "confidence": 0.0,
            "reason": "VLM did not return parseable JSON; report was kept.",
            "raw_response": raw_response[:500],
        }
    is_player = bool(parsed.get("is_player", True))
    confidence = _safe_float(parsed.get("confidence"), 0.0)
    return {
        "status": "available",
        "is_player": is_player,
        "confidence": confidence,
        "reason": str(parsed.get("reason") or f"VLM confidence threshold reference: {confidence_threshold:.2f}."),
        "raw_response": raw_response[:500],
    }


def _encode_image_b64(image: np.ndarray) -> Optional[str]:
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _segment_id_from_local_id(local_id: str) -> Optional[int]:
    match = re.match(r"segment_(\d+):", str(local_id))
    return int(match.group(1)) if match else None


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-") or "player"


def _relative_path(path: Path, start: Path) -> str:
    return path.relative_to(start).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-player Markdown reports from an AGU analysis JSON.")
    parser.add_argument("--analysis-json", required=True, help="Path to AGU analysis JSON.")
    parser.add_argument("--video-path", required=True, help="Path to the source video.")
    parser.add_argument("--output-dir", required=True, help="Directory where Markdown reports and assets are written.")
    parser.add_argument("--max-players", type=int, default=None, help="Optional cap for generated player reports.")
    parser.add_argument("--min-roster-score", type=float, default=0.0, help="Optional minimum support score for a player to remain in the curated roster output.")
    parser.add_argument("--crops-per-player", type=int, default=8, help="Maximum player crops per report.")
    parser.add_argument("--video-fps", type=float, default=2.0, help="FPS for generated evidence videos.")
    parser.add_argument("--vlm-player-filter", action="store_true", help="Ask a VLM whether the green boxed target is a basketball player and filter clear non-player results.")
    parser.add_argument("--vlm-model", default=os.environ.get("BASKETBALL_VLM_MODEL", "qwen3-vl:4b"), help="Ollama vision model used by --vlm-player-filter.")
    parser.add_argument("--vlm-endpoint", default=None, help="Optional full Ollama generate endpoint. Defaults to BASKETBALL_OLLAMA_BASE_URL/api/generate or local Ollama.")
    parser.add_argument("--vlm-confidence-threshold", type=float, default=0.55, help="Reference confidence value included in VLM non-player verification metadata.")
    parser.add_argument("--vlm-timeout-sec", type=float, default=45.0, help="Per-player VLM request timeout in seconds.")
    parser.add_argument("--vlm-concurrency", type=int, default=int(os.environ.get("BASKETBALL_REPORT_VLM_CONCURRENCY", "1")), help="Number of concurrent per-player VLM requests.")
    parser.add_argument("--vlm-cache-path", default=None, help="Reusable JSON cache for VLM player verification results. Defaults to the output directory.")
    parser.add_argument("--vlm-progress", action="store_true", help="Print per-player VLM progress to stderr during long full-video runs.")
    parser.add_argument("--require-vlm-player", action="store_true", help="When --vlm-player-filter is enabled, keep only reports with an available VLM is_player=true result.")
    parser.add_argument("--dedupe-players", action="store_true", help="Drop visually duplicate global player IDs from the human-facing report output.")
    parser.add_argument("--dedupe-similarity-threshold", type=float, default=0.92, help="Appearance similarity threshold used by --dedupe-players.")
    args = parser.parse_args()

    analysis = json.loads(Path(args.analysis_json).read_text(encoding="utf-8"))
    summary = build_player_markdown_reports(
        analysis=analysis,
        video_path=args.video_path,
        output_dir=args.output_dir,
        max_players=args.max_players,
        min_roster_score=args.min_roster_score,
        crops_per_player=args.crops_per_player,
        video_fps=args.video_fps,
        vlm_player_filter=args.vlm_player_filter,
        vlm_model=args.vlm_model,
        vlm_endpoint=args.vlm_endpoint,
        vlm_confidence_threshold=args.vlm_confidence_threshold,
        vlm_timeout_sec=args.vlm_timeout_sec,
        vlm_concurrency=args.vlm_concurrency,
        vlm_cache_path=args.vlm_cache_path,
        vlm_progress=args.vlm_progress,
        require_vlm_player=args.require_vlm_player,
        dedupe_players=args.dedupe_players,
        dedupe_similarity_threshold=args.dedupe_similarity_threshold,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
