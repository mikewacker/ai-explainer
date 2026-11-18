import torch

import ai_explainer.wine as aix


def test_load_data() -> None:
    data = aix.world.load_data()

    assert data.x.shape == (178, 2)
    assert data.y.shape == (178,)
    assert data.x.dtype == torch.float32
    assert data.y.dtype == torch.float32
