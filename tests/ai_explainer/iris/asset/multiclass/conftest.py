import pytest
import torch

import ai_explainer.iris as aix
from ai_explainer.iris.world import IrisNn, IrisTrace


@pytest.fixture
def model() -> IrisNn:
    return IrisNn(task="multiclass", hidden_size=5, seed=42)


@pytest.fixture
def trace(model: IrisNn) -> IrisTrace:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor(2)
    return aix.world.trace(model, x, y)


@pytest.fixture(scope="session")
def act() -> str:
    return "iris-multiclass"


@pytest.fixture(scope="session")
def golden_type() -> str:
    return "scrap"
