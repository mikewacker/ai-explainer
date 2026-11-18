import pytest
import torch

import ai_explainer.wine as aix
import testing
from ai_explainer.wine.asset import Outputs

pytestmark = pytest.mark.golden


@pytest.fixture(scope="session")
def act() -> str:
    return "wine"


@pytest.fixture(scope="session")
def golden_type() -> str:
    return "model"


def test_trained_model(outputs: Outputs, goldens: Outputs) -> None:
    canon = aix.canon.get()

    testing.assert_matches_golden(outputs, goldens, canon.model_trained, "model")


def test_loss() -> None:
    canon = aix.canon.get()

    testing.assert_tensors_close(canon.history[0].loss, torch.tensor(7890.0), decimals=0)
    testing.assert_tensors_close(canon.history[-1].loss, torch.tensor(4215.0), decimals=0)


def test_sample() -> None:
    canon = aix.canon.get()
    x, y = canon.sample

    testing.assert_tensors_equal(x, torch.tensor([1.77, 2.03]))
    testing.assert_tensors_equal(y, torch.tensor(885.0))


def test_trained_model__alt(outputs: Outputs, goldens: Outputs) -> None:
    canon = aix.canon.get(alt=True)

    testing.assert_matches_golden(outputs, goldens, canon.model_trained, "model-alt")


def test_loss__alt() -> None:
    canon = aix.canon.get(alt=True)

    testing.assert_tensors_close(canon.history[0].loss, torch.tensor(5712.0), decimals=0)
    testing.assert_tensors_close(canon.history[-1].loss, torch.tensor(5391.0), decimals=0)
