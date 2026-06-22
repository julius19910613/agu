from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.analysis.schemas import (
    IdentityDuplicateCandidateResponse,
    LongVideoPlayerSummaryResponse,
    PlayerIdentityFeatureResponse,
)
from app.analysis.service import AnalysisService


def build_duplicate_report(
    analysis: Dict[str, Any],
    source_path: str,
    screenshot_dir: Optional[str] = None,
) -> Dict[str, Any]:
    long_video = analysis.get("long_video") or {}
    players = [
        LongVideoPlayerSummaryResponse.model_validate(player)
        for player in long_video.get("players", [])
    ]
    features = {}
    for feature in analysis.get("player_identity_features", []):
        parsed = PlayerIdentityFeatureResponse.model_validate(feature)
        if parsed.local_player_id:
            features[parsed.local_player_id] = parsed

    candidates = _compute_candidates(players, features)
    existing = [
        IdentityDuplicateCandidateResponse.model_validate(candidate)
        for candidate in long_video.get("identity_duplicate_candidates", [])
    ]
    source = "recomputed_from_players_and_identity_features"
    if not candidates and existing:
        candidates = existing
        source = "existing_analysis_result"

    candidate_dicts = [candidate.model_dump() for candidate in candidates]
    review_pairs = _attach_review_images(candidate_dicts, screenshot_dir)
    return {
        "source_analysis": source_path,
        "video": analysis.get("video", ""),
        "generated_at_unix": time.time(),
        "candidate_source": source,
        "candidate_count": len(candidate_dicts),
        "player_count": len(players),
        "identity_feature_count": len(features),
        "identity_embedding_model": analysis.get("identity_embedding_model"),
        "candidates": review_pairs,
    }


def write_review_contact_sheet(report: Dict[str, Any], output_path: Path) -> Optional[Path]:
    rows: List[np.ndarray] = []
    for candidate in report.get("candidates", []):
        left_path = candidate.get("left_review_image")
        right_path = candidate.get("right_review_image")
        if not left_path or not right_path:
            continue
        left = cv2.imread(left_path)
        right = cv2.imread(right_path)
        if left is None or right is None:
            continue
        rows.append(_make_pair_row(candidate, left, right))

    if not rows:
        return None

    width = max(row.shape[1] for row in rows)
    padded_rows = []
    for row in rows:
        if row.shape[1] == width:
            padded_rows.append(row)
            continue
        canvas = np.full((row.shape[0], width, 3), 245, dtype=np.uint8)
        canvas[:, : row.shape[1]] = row
        padded_rows.append(canvas)

    sheet = np.vstack(padded_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return output_path


def _compute_candidates(
    players: List[LongVideoPlayerSummaryResponse],
    features: Dict[str, PlayerIdentityFeatureResponse],
) -> List[IdentityDuplicateCandidateResponse]:
    if not players or not features:
        return []
    settings = SimpleNamespace(torch_num_threads=0, progress_log=False)
    service = AnalysisService(settings=settings, model=torch.nn.Identity(), device=torch.device("cpu"))
    return service._detect_identity_duplicate_candidates(players, features)


def _attach_review_images(
    candidates: List[Dict[str, Any]],
    screenshot_dir: Optional[str],
) -> List[Dict[str, Any]]:
    if not screenshot_dir:
        return candidates
    root = Path(screenshot_dir)
    for candidate in candidates:
        left = root / f"{candidate['left_global_player_id']}_stats_box.jpg"
        right = root / f"{candidate['right_global_player_id']}_stats_box.jpg"
        if left.exists():
            candidate["left_review_image"] = str(left)
        if right.exists():
            candidate["right_review_image"] = str(right)
    return candidates


def _make_pair_row(candidate: Dict[str, Any], left: np.ndarray, right: np.ndarray) -> np.ndarray:
    thumb_width = 420
    left_thumb = _resize_to_width(left, thumb_width)
    right_thumb = _resize_to_width(right, thumb_width)
    image_height = max(left_thumb.shape[0], right_thumb.shape[0])
    header_height = 92
    row = np.full((image_height + header_height, thumb_width * 2, 3), 245, dtype=np.uint8)
    row[header_height : header_height + left_thumb.shape[0], :thumb_width] = left_thumb
    row[header_height : header_height + right_thumb.shape[0], thumb_width : thumb_width * 2] = right_thumb
    title = (
        f"{candidate['left_global_player_id']} <-> {candidate['right_global_player_id']} "
        f"conf={candidate['confidence']:.2f}"
    )
    cv2.putText(row, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
    evidence = "; ".join(candidate.get("evidence", [])[:3])
    cv2.putText(row, evidence[:105], (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    return row


def _resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    height = max(1, int(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an offline AGU identity duplicate review report.")
    parser.add_argument("--analysis-json", required=True, help="Path to an AGU analysis JSON result.")
    parser.add_argument("--output-json", required=True, help="Path to write the duplicate report JSON.")
    parser.add_argument("--screenshot-dir", default=None, help="Optional directory containing per-player screenshot JPGs.")
    parser.add_argument("--contact-sheet", default=None, help="Optional output JPG for duplicate review pairs.")
    args = parser.parse_args()

    analysis_path = Path(args.analysis_json)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    report = build_duplicate_report(
        analysis=analysis,
        source_path=str(analysis_path),
        screenshot_dir=args.screenshot_dir,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    contact_sheet = None
    if args.contact_sheet:
        contact_sheet = write_review_contact_sheet(report, Path(args.contact_sheet))

    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "candidate_count": report["candidate_count"],
                "contact_sheet": str(contact_sheet) if contact_sheet else None,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
