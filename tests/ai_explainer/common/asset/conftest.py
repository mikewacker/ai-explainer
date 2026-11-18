from collections.abc import Callable

import pytest

import testing.example as aix
from testing.example.world import ExampleData, ExampleModel


@pytest.fixture
def data() -> ExampleData:
    return aix.world.load_data()


@pytest.fixture
def model_factory() -> Callable[[], ExampleModel]:
    return ExampleModel


@pytest.fixture(scope="session")
def act() -> str:
    return "common"


@pytest.fixture(scope="session")
def golden_type() -> str:
    return "scrap"
