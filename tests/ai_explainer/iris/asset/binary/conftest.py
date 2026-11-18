from collections.abc import Callable

import pytest
import torch

import ai_explainer.iris as aix
from ai_explainer.iris.world import IrisData, IrisNn, IrisTrace


@pytest.fixture
def data() -> IrisData:
    generator = torch.Generator().manual_seed(42)
    return IrisData(
        x_train=torch.rand(8, 4, generator=generator),
        y_train=torch.randint(0, 2, (8,), dtype=torch.float32, generator=generator),
        x_test=torch.rand(2, 4, generator=generator),
        y_test=torch.randint(0, 2, (2,), dtype=torch.float32, generator=generator),
    )


@pytest.fixture
def model(model_factory: Callable[[], IrisNn]) -> IrisNn:
    return model_factory()


@pytest.fixture
def model_factory() -> Callable[[], IrisNn]:
    return lambda: IrisNn(hidden_size=5, seed=42)


@pytest.fixture
def trace(model: IrisNn) -> IrisTrace:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor(1.0)
    return aix.world.trace(model, x, y)


@pytest.fixture(scope="session")
def act() -> str:
    return "iris-binary"


@pytest.fixture(scope="session")
def golden_type() -> str:
    return "scrap"
