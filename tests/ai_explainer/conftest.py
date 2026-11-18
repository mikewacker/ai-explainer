from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch

from ai_explainer.common.asset import Outputs


@pytest.fixture(scope="module")
def outputs(tmp_path_factory: pytest.TempPathFactory, act: str) -> Outputs:
    root_dir = tmp_path_factory.mktemp("out")
    outputs = Outputs(root_dir).act(act)
    outputs.mkdir()
    return outputs


@pytest.fixture(scope="module")
def goldens(golden_type: str, act: str) -> Outputs:
    return Outputs(Path("golden", golden_type)).act(act)


@pytest.fixture(scope="module")
def video(request: pytest.FixtureRequest) -> bool:
    return any(
        item.path == request.path and item.get_closest_marker("video")
        for item in request.session.items
    )


@pytest.fixture(autouse=True)
def set_seed() -> None:
    torch.manual_seed(42)


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if isinstance(item, pytest.Function) and "outputs" in item.fixturenames:
            item.add_marker(pytest.mark.golden)
