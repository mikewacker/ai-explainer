from collections.abc import Callable

import torch

import testing
import testing.example as aix
from testing.example.world import ExampleData, ExampleModel


class TestLossCrossFn:
    def test_params(self, data: ExampleData, model_factory: Callable[[], ExampleModel]) -> None:
        loss_fn_w1 = aix.asset.loss_cross_fn(data.x, data.y, model_factory, feature=1)

        testing.assert_tensors_equal(loss_fn_w1.params, torch.tensor([5.0, 2.0]))

    def test_call(self, data: ExampleData, model_factory: Callable[[], ExampleModel]) -> None:
        loss_fn_w2 = aix.asset.loss_cross_fn(data.x, data.y, model_factory, feature=2)
        w2 = torch.tensor([1.0, 2.0, 3.0])
        loss = loss_fn_w2(w2)

        testing.assert_tensors_close(loss, torch.tensor([11.0 / 3, 59.0 / 3, 73.0]))

    def test_call__0d(self, data: ExampleData, model_factory: Callable[[], ExampleModel]) -> None:
        loss_fn_w2 = aix.asset.loss_cross_fn(data.x, data.y, model_factory, feature=2)
        w2 = torch.tensor(1.0)
        loss = loss_fn_w2(w2)

        testing.assert_tensors_close(loss, torch.tensor(11.0 / 3))
