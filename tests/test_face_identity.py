from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.analysis.face_identity import OpenCvSFaceIdentityAdapter, build_face_identity_adapter


def test_sface_adapter_aggregates_normalized_face_embeddings(tmp_path):
    detector_model = tmp_path / "yunet.onnx"
    recognizer_model = tmp_path / "sface.onnx"
    detector_model.touch()
    recognizer_model.touch()

    detector = MagicMock()
    detector.detect.return_value = (
        None,
        np.array(
            [[8.0, 4.0, 16.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90]],
            dtype=np.float32,
        ),
    )
    recognizer = MagicMock()
    recognizer.alignCrop.side_effect = lambda crop, face: crop
    recognizer.feature.side_effect = [
        np.array([[3.0, 4.0]], dtype=np.float32),
        np.array([[4.0, 3.0]], dtype=np.float32),
    ]

    with patch("app.analysis.face_identity.cv2.FaceDetectorYN.create", return_value=detector), patch(
        "app.analysis.face_identity.cv2.FaceRecognizerSF.create",
        return_value=recognizer,
    ):
        adapter = OpenCvSFaceIdentityAdapter(str(detector_model), str(recognizer_model))
        result = adapter.embed_player_crops(
            [np.full((80, 40, 3), 100, dtype=np.uint8) for _ in range(2)]
        )

    assert result is not None
    assert result.sample_count == 2
    assert result.quality > 0.90
    assert result.embedding.shape == (2,)
    assert np.linalg.norm(result.embedding) == pytest.approx(1.0)
    assert "sface" in result.model_id


def test_sface_adapter_rejects_face_below_player_upper_body(tmp_path):
    detector_model = tmp_path / "yunet.onnx"
    recognizer_model = tmp_path / "sface.onnx"
    detector_model.touch()
    recognizer_model.touch()

    detector = MagicMock()
    detector.detect.return_value = (
        None,
        np.array(
            [[8.0, 55.0, 16.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90]],
            dtype=np.float32,
        ),
    )

    with patch("app.analysis.face_identity.cv2.FaceDetectorYN.create", return_value=detector), patch(
        "app.analysis.face_identity.cv2.FaceRecognizerSF.create",
        return_value=MagicMock(),
    ):
        adapter = OpenCvSFaceIdentityAdapter(str(detector_model), str(recognizer_model))
        result = adapter.embed_player_crops([np.full((80, 40, 3), 100, dtype=np.uint8)])

    assert result is None


def test_sface_adapter_rejects_inconsistent_track_faces(tmp_path):
    detector_model = tmp_path / "yunet.onnx"
    recognizer_model = tmp_path / "sface.onnx"
    detector_model.touch()
    recognizer_model.touch()

    detector = MagicMock()
    detector.detect.return_value = (
        None,
        np.array(
            [[8.0, 4.0, 16.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90]],
            dtype=np.float32,
        ),
    )
    recognizer = MagicMock()
    recognizer.alignCrop.side_effect = lambda crop, face: crop
    recognizer.feature.side_effect = [
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0]], dtype=np.float32),
        np.array([[-1.0, 0.0]], dtype=np.float32),
    ]

    with patch("app.analysis.face_identity.cv2.FaceDetectorYN.create", return_value=detector), patch(
        "app.analysis.face_identity.cv2.FaceRecognizerSF.create",
        return_value=recognizer,
    ):
        adapter = OpenCvSFaceIdentityAdapter(str(detector_model), str(recognizer_model))
        result = adapter.embed_player_crops(
            [np.full((80, 40, 3), 100, dtype=np.uint8) for _ in range(3)]
        )

    assert result is None


def test_sface_if_available_falls_back_when_models_are_missing(tmp_path):
    adapter = build_face_identity_adapter(
        backend="opencv_sface_if_available",
        detector_model_path=str(tmp_path / "missing-yunet.onnx"),
        recognizer_model_path=str(tmp_path / "missing-sface.onnx"),
        allow_fallback=True,
    )

    assert adapter is None


def test_required_sface_backend_raises_when_models_are_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_face_identity_adapter(
            backend="opencv_sface",
            detector_model_path=str(tmp_path / "missing-yunet.onnx"),
            recognizer_model_path=str(tmp_path / "missing-sface.onnx"),
            allow_fallback=True,
        )
