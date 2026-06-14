import numpy as np
import torch

from app.analysis.inference import inference_batch


def test_inference_batch_matches_v3_bgr_255_preprocessing():
    clip = np.zeros((16, 4, 4, 3), dtype=np.uint8)
    clip[..., 0] = 10
    clip[..., 1] = 20
    clip[..., 2] = 30

    batch = inference_batch(np.expand_dims(clip, axis=0))

    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (1, 3, 16, 112, 112)
    assert torch.all(batch[:, 0] == 10)
    assert torch.all(batch[:, 1] == 20)
    assert torch.all(batch[:, 2] == 30)
    assert batch.max().item() == 30
