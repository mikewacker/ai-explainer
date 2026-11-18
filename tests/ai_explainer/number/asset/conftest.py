from collections.abc import Callable

import pytest
import torch
from torch import Tensor

import ai_explainer.number as aix
from ai_explainer.number.asset import Viewport
from ai_explainer.number.world import NumberData, NumberModel, NumberTrace


@pytest.fixture
def data() -> NumberData:
    return NumberData(y=torch.tensor(1.5))


@pytest.fixture
def model() -> NumberModel:
    return NumberModel(w=torch.tensor(3.0))


@pytest.fixture
def trace(model: NumberModel, data: NumberData) -> NumberTrace:
    return aix.world.trace(model, data.y)


@pytest.fixture
def loss_fn(data: NumberData) -> Callable[[Tensor], Tensor]:
    return aix.asset.loss_fn(data.y)


@pytest.fixture
def loss_viewport() -> Viewport:
    return Viewport(x_var="w", xlim=(-2, 6), y_var="L", ylim=(-1, 7))


@pytest.fixture(scope="session")
def act() -> str:
    return "number"


@pytest.fixture(scope="session")
def golden_type() -> str:
    return "scrap"
