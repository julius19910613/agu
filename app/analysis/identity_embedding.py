from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdentityEmbeddingResult:
    embedding: np.ndarray
    model_id: str
    method: str


class BaseIdentityEmbedder:
    model_id: str
    method: str

    def embed_crops(self, crops_bgr: Sequence[np.ndarray]) -> IdentityEmbeddingResult:
        raise NotImplementedError


class SidecarHsvHistogramEmbedder(BaseIdentityEmbedder):
    model_id = "sidecar_hsv_hist_embedding_v1"
    method = "sidecar_hsv_hist_embedding_v1"

    def embed_crops(self, crops_bgr: Sequence[np.ndarray]) -> IdentityEmbeddingResult:
        embeddings = [self._crop_embedding(crop) for crop in crops_bgr if crop is not None and crop.size > 0]
        if not embeddings:
            embedding = np.zeros((128,), dtype=np.float32)
        else:
            embedding = np.stack(embeddings, axis=0).mean(axis=0).astype(np.float32)
            embedding = _l2_normalize(embedding)
        return IdentityEmbeddingResult(embedding=embedding, model_id=self.model_id, method=self.method)

    def _crop_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        crop_small = cv2.resize(crop_bgr, (16, 16), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop_small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [8, 4, 4],
            [0, 180, 0, 256, 0, 256],
        ).astype(np.float32).reshape(-1)
        return _l2_normalize(hist)


class TorchvisionMobileNetV3SmallEmbedder(BaseIdentityEmbedder):
    method = "torchvision_mobilenet_v3_small_embedding_v1"

    def __init__(self, weights_name: str, device_name: str, batch_size: int = 16):
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        normalized_weights = (weights_name or "default").lower()
        if normalized_weights in {"default", "imagenet", "imagenet1k_v1"}:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            weight_label = "imagenet1k_v1"
        elif normalized_weights in {"none", "random", "untrained"}:
            weights = None
            weight_label = "none"
        else:
            raise ValueError(
                "identity_embedding_weights must be one of default, imagenet1k_v1, none; "
                f"got {weights_name!r}"
            )

        self.model_id = f"torchvision_mobilenet_v3_small_{weight_label}_embedding_v1"
        self.device = _resolve_torch_device(device_name)
        self.batch_size = max(1, int(batch_size or 16))
        model = mobilenet_v3_small(weights=weights)
        self.features = torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten()).to(self.device)
        self.features.eval()

    def embed_crops(self, crops_bgr: Sequence[np.ndarray]) -> IdentityEmbeddingResult:
        tensors = [self._preprocess(crop) for crop in crops_bgr if crop is not None and crop.size > 0]
        if not tensors:
            embedding = np.zeros((576,), dtype=np.float32)
            return IdentityEmbeddingResult(embedding=embedding, model_id=self.model_id, method=self.method)

        outputs: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(tensors), self.batch_size):
                batch = torch.stack(tensors[start : start + self.batch_size], dim=0).to(self.device)
                features = self.features(batch)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                outputs.append(features.detach().cpu().numpy().astype(np.float32))

        embedding = np.concatenate(outputs, axis=0).mean(axis=0).astype(np.float32)
        embedding = _l2_normalize(embedding)
        return IdentityEmbeddingResult(embedding=embedding, model_id=self.model_id, method=self.method)

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(crop_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std


def build_identity_embedder(
    backend: str,
    weights: str = "default",
    device: str = "mps_if_available",
    batch_size: int = 16,
    allow_fallback: bool = True,
) -> BaseIdentityEmbedder:
    normalized_backend = (backend or "torchvision_mobilenet_v3_small").lower()
    if normalized_backend in {"sidecar", "sidecar_hsv", "sidecar_hsv_hist"}:
        return SidecarHsvHistogramEmbedder()
    if normalized_backend in {"torchvision_mobilenet_v3_small", "mobilenet_v3_small", "mobilenetv3_small"}:
        try:
            return TorchvisionMobileNetV3SmallEmbedder(
                weights_name=weights,
                device_name=device,
                batch_size=batch_size,
            )
        except Exception as exc:
            if not allow_fallback:
                raise
            LOGGER.warning("Falling back to sidecar identity embedding after backend load failed: %s", exc)
            return SidecarHsvHistogramEmbedder()
    raise ValueError(f"Unsupported identity embedding backend: {backend}")


def _resolve_torch_device(device_name: str) -> torch.device:
    preference = (device_name or "mps_if_available").lower()
    if preference in {"auto", "mps_if_available"}:
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


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        return vector / norm
    return vector
