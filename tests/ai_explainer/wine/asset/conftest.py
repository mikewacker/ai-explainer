from collections.abc import Callable

import pytest
import torch
from torch import Tensor

import ai_explainer.wine as aix
from ai_explainer.wine.asset import LossCrossFn
from ai_explainer.wine.world import WineData, WineModel, WineTrace


@pytest.fixture
def data() -> WineData:
    return WineData(
        x=torch.tensor([[3.0, 1.0], [1.0, 3.0], [1.0, 1.0]]),
        y=torch.tensor([5.0, 9.0, 8.0]),
    )


@pytest.fixture
def model(model_factory: Callable[[], WineModel]) -> WineModel:
    return model_factory()


@pytest.fixture
def model_factory() -> Callable[[], WineModel]:
    return lambda: WineModel(w=torch.tensor([2.0, 5.0]))


@pytest.fixture
def trace_sample(model: WineModel, data: WineData) -> WineTrace:
    return aix.world.trace(model, data.x[0], data.y[0])


@pytest.fixture
def trace_data(model: WineModel, data: WineData) -> WineTrace:
    return aix.world.trace(model, data.x, data.y)


@pytest.fixture
def loss_mesh(data: WineData) -> Tensor:
    return aix.asset.loss_mesh(data.x, data.y, loss_delta=50)


@pytest.fixture
def loss_fn_w1(data: WineData, model_factory: Callable[[], WineModel]) -> LossCrossFn[Tensor]:
    return aix.asset.loss_cross_fn(data.x, data.y, model_factory, j=1)


@pytest.fixture
def loss_fn_w2(data: WineData, model_factory: Callable[[], WineModel]) -> LossCrossFn[Tensor]:
    return aix.asset.loss_cross_fn(data.x, data.y, model_factory, j=2)


@pytest.fixture(scope="session")
def act() -> str:
    return "wine"


@pytest.fixture(scope="session")
def golden_type() -> str:
    return "scrap"
