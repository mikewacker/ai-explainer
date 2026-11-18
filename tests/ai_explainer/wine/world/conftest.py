import pytest
import torch

from ai_explainer.wine.world import WineData, WineHyperparams


@pytest.fixture
def data() -> WineData:
    return WineData(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        y=torch.tensor([10.0, 20.0, 30.0]),
    )


@pytest.fixture
def hparams() -> WineHyperparams:
    return WineHyperparams(w_init=torch.tensor([4.0, 2.0]), lr=0.01, steps=3)
