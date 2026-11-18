import pytest
import torch

import ai_explainer.number as aix
import testing
from ai_explainer.number.asset import Outputs

pytestmark = pytest.mark.golden


@pytest.fixture(scope="session")
def act() -> str:
    return "number"


@pytest.fixture(scope="session")
def golden_type() -> str:
    return "model"


def test_trained_model(outputs: Outputs, goldens: Outputs) -> None:
    canon = aix.canon.get()

    testing.assert_matches_golden(outputs, goldens, canon.model_trained, "model")


def test_loss() -> None:
    canon = aix.canon.get()

    testing.assert_tensors_close(canon.history[0].loss, torch.tensor(2.25), decimals=2)
    testing.assert_tensors_close(canon.history[-1].loss, torch.tensor(0.04), decimals=2)


def test_sample() -> None:
    canon = aix.canon.get()
    x, y = canon.sample

    testing.assert_tensors_equal(x, torch.empty(0))
    testing.assert_tensors_equal(y, torch.tensor(3.5))
