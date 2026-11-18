import pytest

import testing.example as aix
from testing.example.world import ExampleData, ExampleHyperparams


@pytest.fixture
def data() -> ExampleData:
    return aix.world.load_data()


@pytest.fixture
def hparams() -> ExampleHyperparams:
    return ExampleHyperparams(lr=0.01, steps=3)
