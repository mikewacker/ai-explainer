import pytest
import torch

import ai_explainer.iris as aix
import testing
from ai_explainer.iris.asset.binary import Outputs

pytestmark = pytest.mark.golden


@pytest.fixture(scope="session")
def act() -> str:
    return "iris"


@pytest.fixture(scope="session")
def golden_type() -> str:
    return "model"


def test_trained_model(outputs: Outputs, goldens: Outputs) -> None:
    canon = aix.canon.get()

    testing.assert_matches_golden(outputs, goldens, canon.model_trained, "model")


def test_loss() -> None:
    canon = aix.canon.get()

    testing.assert_tensors_close(canon.history[0].loss, torch.tensor(0.847), decimals=3)
    testing.assert_tensors_close(canon.history[-1].loss, torch.tensor(0.101), decimals=3)


def test_accuracy() -> None:
    canon = aix.canon.get()

    testing.assert_tensors_close(canon.accuracy_train, torch.tensor(117 / 120))
    testing.assert_tensors_close(canon.accuracy_test, torch.tensor(30 / 30))


def test_sample() -> None:
    canon = aix.canon.get()
    x, y = canon.sample

    testing.assert_tensors_equal(x, torch.tensor([4.9, 2.5, 4.5, 1.7]))
    testing.assert_tensors_equal(y, torch.tensor(1.0))


def test_trained_model__multiclass(outputs: Outputs, goldens: Outputs) -> None:
    canon = aix.canon.get(task="multiclass")

    testing.assert_matches_golden(outputs, goldens, canon.model_trained, "model-multiclass")


def test_loss__multiclass() -> None:
    canon = aix.canon.get(task="multiclass")

    testing.assert_tensors_close(canon.history[0].loss, torch.tensor(1.700), decimals=3)
    testing.assert_tensors_close(canon.history[-1].loss, torch.tensor(0.090), decimals=3)


def test_accuracy__multiclass() -> None:
    canon = aix.canon.get(task="multiclass")

    testing.assert_tensors_close(canon.accuracy_train, torch.tensor(118 / 120))
    testing.assert_tensors_close(canon.accuracy_test, torch.tensor(30 / 30))


def test_sample__multiclass() -> None:
    canon = aix.canon.get(task="multiclass")
    x, y = canon.sample

    testing.assert_tensors_equal(x, torch.tensor([4.9, 2.5, 4.5, 1.7]))
    testing.assert_tensors_equal(y, torch.tensor(2))
