import pytest
import torch

from ai_explainer.number.world import NumberData, NumberHyperparams


@pytest.fixture
def data() -> NumberData:
    return NumberData(y=torch.tensor(3.5))


@pytest.fixture
def hparams() -> NumberHyperparams:
    return NumberHyperparams(w_init=torch.tensor(5.0), lr=0.01, steps=3)
