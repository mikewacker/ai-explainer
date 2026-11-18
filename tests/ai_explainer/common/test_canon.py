import pytest
import torch

import testing
import testing.example as aix
from testing.example.asset import Outputs

pytestmark = pytest.mark.golden


@pytest.fixture(scope="session")
def act() -> str:
    return "common"


@pytest.fixture(scope="session")
def golden_type() -> str:
    return "scrap"


def test_model_trained(outputs: Outputs, goldens: Outputs) -> None:
    canon = aix.canon.get()

    testing.assert_matches_golden(outputs, goldens, canon.model_trained, "model")


def test_loss() -> None:
    canon = aix.canon.get()

    testing.assert_tensors_close(canon.history[0].loss, torch.tensor(19.67), decimals=2)
    testing.assert_tensors_close(canon.history[-1].loss, torch.tensor(2.98), decimals=2)


def test_sample() -> None:
    canon = aix.canon.get()
    x, y = canon.sample

    testing.assert_tensors_equal(x, torch.tensor([1.0, 2.0]))
    testing.assert_tensors_equal(y, torch.tensor(10.0))


def test_model_untrained() -> None:
    canon = aix.canon.get()

    testing.assert_tensors_equal(canon.model_untrained.params(), torch.tensor([5.0, 2.0]))


def test_data() -> None:
    canon = aix.canon.get()

    testing.assert_tensors_equal(canon.data.x, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    testing.assert_tensors_equal(canon.data.y, torch.tensor([10.0, 20.0, 30.0]))


def test_hparams() -> None:
    canon = aix.canon.get()

    assert canon.hparams.lr == 0.01
    assert canon.hparams.steps == 3
