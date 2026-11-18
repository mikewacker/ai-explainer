from collections.abc import Callable

import torch

import ai_explainer.wine as aix
import testing
from ai_explainer.wine.world import WineData, WineModel


def test_loss_mesh(data: WineData) -> None:
    loss_mesh = aix.asset.loss_mesh(data.x, data.y, loss_delta=50)

    assert loss_mesh.shape == (101, 145, 3)
    testing.assert_tensors_close(loss_mesh[0, 0], torch.tensor([1.00, 3.00, 6.00]), decimals=2)
    testing.assert_tensors_close(loss_mesh[100, 0], torch.tensor([-3.33, 7.33, 56.00]), decimals=2)


def test_loss_cross_fn__w1(data: WineData, model_factory: Callable[[], WineModel]) -> None:
    loss_fn_w1 = aix.asset.loss_cross_fn(data.x, data.y, model_factory, j=1)
    w1 = torch.tensor([1.0, 2.0, 3.0])
    loss = loss_fn_w1(w1)

    testing.assert_tensors_close(loss, torch.tensor([62.0 / 3, 101.0 / 3, 162.0 / 3]))


def test_loss_cross_fn__w2(data: WineData, model_factory: Callable[[], WineModel]) -> None:
    loss_fn_w2 = aix.asset.loss_cross_fn(data.x, data.y, model_factory, j=2)
    w2 = torch.tensor([1.0, 2.0, 3.0])
    loss = loss_fn_w2(w2)

    testing.assert_tensors_close(loss, torch.tensor([45.0 / 3, 26.0 / 3, 29.0 / 3]))
