from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


EVENT_ACTION_HINTS = {
    "block_candidate": {"block", "defense"},
    "rebound_candidate": {"ball in hand", "dribble", "shoot"},
    "steal_candidate": {"ball in hand", "dribble", "defense", "block"},
    "shoot": {"shoot"},
    "pass": {"pass"},
}


def build_event_owner_candidates(
    records: Iterable[Any],
    *,
    event_type: str,
    start_frame: int,
    end_frame: int,
    primary_player_id: Optional[str] = None,
    max_candidates: int = 5,
    search_margin_frames: int = 90,
) -> List[Dict[str, Any]]:
    """Rank likely event-owner players from nearby action records.

    This is a deterministic, dependency-light owner scoring layer. It is not a
    final identity recognizer; it exposes the candidate set needed for VLM or
    human review and for later supervised actor selection.
    """
    event_center = (int(start_frame) + int(end_frame)) // 2
    lower = int(start_frame) - int(search_margin_frames)
    upper = int(end_frame) + int(search_margin_frames)
    grouped: Dict[str, List[Any]] = defaultdict(list)
    for record in records:
        record_center = (_record_int(record, "start_frame") + _record_int(record, "end_frame")) // 2
        if record_center < lower or record_center > upper:
            continue
        player_id = _record_player_id(record)
        if player_id:
            grouped[player_id].append(record)

    action_hints = EVENT_ACTION_HINTS.get(event_type, set())
    scored: List[Dict[str, Any]] = []
    for player_id, player_records in grouped.items():
        action_matches = [
            record for record in player_records if _record_final_action(record) in action_hints
        ]
        relevant_records = action_matches or player_records
        avg_confidence = sum(_record_final_confidence(record) for record in relevant_records) / max(1, len(relevant_records))
        nearest_gap = min(
            abs(((_record_int(record, "start_frame") + _record_int(record, "end_frame")) // 2) - event_center)
            for record in relevant_records
        )
        temporal_score = max(0.0, 1.0 - nearest_gap / max(1.0, float(search_margin_frames + max(1, end_frame - start_frame))))
        action_score = len(action_matches) / max(1, len(player_records))
        primary_bonus = 1.0 if primary_player_id and player_id == primary_player_id else 0.0
        identity_confidence = max(_record_identity_confidence(record) for record in player_records)
        clip_support = min(1.0, len(player_records) / 4.0)
        score = (
            avg_confidence * 0.30
            + temporal_score * 0.25
            + action_score * 0.20
            + identity_confidence * 0.15
            + clip_support * 0.05
            + primary_bonus * 0.05
        )
        scored.append(
            {
                "global_player_id": player_id,
                "local_player_ids": sorted(
                    {
                        str(_record_attr(record, "local_player_id") or "")
                        for record in player_records
                        if _record_attr(record, "local_player_id")
                    }
                ),
                "score": round(max(0.0, min(0.99, score)), 4),
                "clip_count": len(player_records),
                "action_match_count": len(action_matches),
                "avg_confidence": round(avg_confidence, 4),
                "nearest_frame_gap": int(nearest_gap),
                "evidence": [
                    f"{len(player_records)} nearby clips",
                    f"{len(action_matches)} clips match event action hints",
                    f"nearest clip gap {int(nearest_gap)} frames",
                ],
            }
        )

    scored.sort(key=lambda item: (-float(item["score"]), item["nearest_frame_gap"], item["global_player_id"]))
    for rank, item in enumerate(scored[: max(0, int(max_candidates))], start=1):
        item["rank"] = rank
    return scored[: max(0, int(max_candidates))]


def _record_attr(record: Any, name: str) -> Any:
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def _record_int(record: Any, name: str) -> int:
    return int(_record_attr(record, name) or 0)


def _record_player_id(record: Any) -> str:
    return str(
        _record_attr(record, "global_player_id")
        or _record_attr(record, "local_player_id")
        or f"player_{_record_attr(record, 'player')}"
    )


def _record_final(record: Any) -> Any:
    final = _record_attr(record, "final")
    return final or {}


def _record_final_action(record: Any) -> str:
    final = _record_final(record)
    if isinstance(final, dict):
        return str(final.get("action") or "")
    return str(getattr(final, "action", "") or "")


def _record_final_confidence(record: Any) -> float:
    final = _record_final(record)
    if isinstance(final, dict):
        return float(final.get("confidence") or 0.0)
    return float(getattr(final, "confidence", 0.0) or 0.0)


def _record_identity_confidence(record: Any) -> float:
    value = _record_attr(record, "identity_confidence")
    return float(value or 0.0)
