from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)
_DIGITS = re.compile(r"^\d{1,3}$")


@dataclass(frozen=True)
class ScoreboardOCRRead:
    left_score: int
    right_score: int
    confidence: float
    evidence: str


class RapidOCRScoreboardReader:
    method = "rapidocr_scoreboard_v1"

    def __init__(self, confidence_threshold: float = 0.75) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._ocr = RapidOCR()
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))

    def read(self, image_bgr: np.ndarray) -> Optional[ScoreboardOCRRead]:
        results, _ = self._ocr(image_bgr)
        if not results:
            return None
        candidates: list[dict[str, Any]] = []
        for box, text, confidence in results:
            normalized = str(text or "").strip()
            confidence = float(confidence or 0.0)
            if not _DIGITS.fullmatch(normalized) or confidence < self.confidence_threshold:
                continue
            points = np.asarray(box, dtype=np.float32)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            left = max(0, int(np.floor(x_min)))
            top = max(0, int(np.floor(y_min)))
            right = min(image_bgr.shape[1], int(np.ceil(x_max)) + 1)
            bottom = min(image_bgr.shape[0], int(np.ceil(y_max)) + 1)
            crop = image_bgr[top:bottom, left:right]
            if crop.size == 0:
                continue
            upright_text, upright_confidence = self._recognize_upright_crop(
                crop,
                fallback_text=normalized,
                fallback_confidence=confidence,
            )
            normalized = upright_text
            confidence = upright_confidence
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hue, saturation, value = cv2.split(hsv)
            cyan_ratio = float(
                (((hue >= 42) & (hue < 110)) & (saturation > 50) & (value > 120)).mean()
            )
            if cyan_ratio < 0.12:
                continue
            candidates.append(
                {
                    "value": int(normalized),
                    "confidence": confidence,
                    "center_x": float((x_min + x_max) / 2.0),
                    "center_y": float((y_min + y_max) / 2.0),
                    "height": float(y_max - y_min),
                    "cyan_ratio": cyan_ratio,
                }
            )
        if len(candidates) < 2:
            return None
        max_height = max(item["height"] for item in candidates)
        large = [item for item in candidates if item["height"] >= max_height * 0.68]
        if len(large) < 2:
            return None
        best_pair: Optional[tuple[float, dict[str, Any], dict[str, Any]]] = None
        image_width = max(1.0, float(image_bgr.shape[1]))
        image_height = max(1.0, float(image_bgr.shape[0]))
        for left in large:
            for right in large:
                if left["center_x"] >= right["center_x"]:
                    continue
                separation = (right["center_x"] - left["center_x"]) / image_width
                vertical_gap = abs(right["center_y"] - left["center_y"]) / image_height
                if (
                    separation < 0.25
                    or vertical_gap > 0.16
                    or left["center_x"] >= image_width * 0.45
                    or right["center_x"] <= image_width * 0.55
                ):
                    continue
                pair_score = separation - vertical_gap + min(left["confidence"], right["confidence"])
                if best_pair is None or pair_score > best_pair[0]:
                    best_pair = (pair_score, left, right)
        if best_pair is None:
            return None
        _, left, right = best_pair
        if max(int(left["value"]), int(right["value"])) >= 30 and min(
            int(left["value"]), int(right["value"])
        ) < 10:
            return None
        confidence = min(float(left["confidence"]), float(right["confidence"]))
        return ScoreboardOCRRead(
            left_score=int(left["value"]),
            right_score=int(right["value"]),
            confidence=confidence,
            evidence=f"RapidOCR large side digits confidence={confidence:.3f}",
        )

    def _recognize_upright_crop(
        self,
        crop_bgr: np.ndarray,
        fallback_text: str,
        fallback_confidence: float,
    ) -> tuple[str, float]:
        """Re-read a detected score without orientation classification.

        Seven-segment strings such as ``19`` are valid upside down and RapidOCR's
        direction classifier can rotate them into ``61``. Scoreboards in video
        frames are already upright, so recognition-only inference is authoritative.
        """
        try:
            results, _ = self._ocr(
                crop_bgr,
                use_det=False,
                use_cls=False,
                use_rec=True,
            )
        except (TypeError, ValueError):
            return fallback_text, fallback_confidence
        if not results or not isinstance(results[0], (list, tuple)) or len(results[0]) < 2:
            return fallback_text, fallback_confidence
        text = str(results[0][0] or "").strip()
        try:
            confidence = float(results[0][1] or 0.0)
        except (TypeError, ValueError):
            return fallback_text, fallback_confidence
        if not _DIGITS.fullmatch(text) or confidence < self.confidence_threshold:
            return fallback_text, fallback_confidence
        return text, confidence


def build_scoreboard_ocr_reader(
    backend: str,
    confidence_threshold: float = 0.75,
) -> Optional[RapidOCRScoreboardReader]:
    normalized = (backend or "off").strip().lower()
    if normalized in {"off", "none", "disabled"}:
        return None
    if normalized not in {"rapidocr", "rapidocr_if_available"}:
        raise ValueError(f"Unsupported scoreboard OCR backend: {backend}")
    try:
        return RapidOCRScoreboardReader(confidence_threshold=confidence_threshold)
    except Exception as exc:
        if normalized == "rapidocr":
            raise
        LOGGER.warning("RapidOCR scoreboard backend unavailable; using VLM fallback: %s", exc)
        return None
