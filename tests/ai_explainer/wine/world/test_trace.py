import torch

import ai_explainer.wine as aix
import testing
from ai_explainer.wine.world import WineData, WineHyperparams, WineModel


def test_trace(data: WineData, hparams: WineHyperparams) -> None:
    model = WineModel(w=hparams.w_init)
    trace = aix.world.trace(model, data.x, data.y)

    testing.assert_tensors_equal(trace.w, torch.tensor([4.0, 2.0]))
    testing.assert_tensors_equal(trace.x, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    testing.assert_tensors_close(trace.y_hat, torch.tensor([8.0, 20.0, 32.0]))
    testing.assert_tensors_equal(trace.y, torch.tensor([10.0, 20.0, 30.0]))
    testing.assert_tensors_close(trace.ell, torch.tensor([4.0, 0.0, 4.0]))
    testing.assert_tensors_close(trace.loss, torch.tensor(8.0 / 3))
    testing.assert_tensors_close(trace.w_grad, torch.tensor([16.0 / 3, 16.0 / 3.0]))
    assert trace.x is not data.x
    assert trace.y is not data.y
    assert not trace.ell.requires_grad


def test_trace__sample(data: WineData, hparams: WineHyperparams) -> None:
    model = WineModel(w=hparams.w_init)
    trace = aix.world.trace(model, data.x[0], data.y[0])

    testing.assert_tensors_equal(trace.w, torch.tensor([4.0, 2.0]))
    testing.assert_tensors_equal(trace.x, torch.tensor([1.0, 2.0]))
    testing.assert_tensors_close(trace.y_hat, torch.tensor(8.0))
    testing.assert_tensors_equal(trace.y, torch.tensor(10.0))
    testing.assert_tensors_close(trace.ell, torch.tensor(4.0))
    testing.assert_tensors_close(trace.loss, torch.tensor(4.0))
    testing.assert_tensors_close(trace.w_grad, torch.tensor([-4.0, -8.0]))
