import torch

import testing
import testing.example as aix
from testing.example.world import ExampleData, ExampleModel


def test_trace(data: ExampleData) -> None:
    model = ExampleModel()
    model.eval()  # trace() should call model.train()
    trace = aix.world.trace(model, data.x, data.y)

    testing.assert_tensors_equal(trace.w, torch.tensor([5.0, 2.0]))
    testing.assert_tensors_equal(trace.x, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    testing.assert_tensors_close(trace.y_hat, torch.tensor([9.0, 23.0, 37.0]))
    testing.assert_tensors_equal(trace.y, torch.tensor([10.0, 20.0, 30.0]))
    testing.assert_tensors_close(trace.loss, torch.tensor(59.0 / 3))
    testing.assert_tensors_close(trace.y_hat_grad, torch.tensor([-2.0 / 3, 6.0 / 3, 14.0 / 3]))
    testing.assert_tensors_close(trace.w_grad, torch.tensor([86.0 / 3, 104.0 / 3]))
    assert not trace.w.requires_grad
    assert not trace.x.requires_grad
    assert not trace.y_hat.requires_grad
    assert not trace.y.requires_grad
    assert not trace.loss.requires_grad
    assert not trace.y_hat_grad.requires_grad
    assert not trace.w_grad.requires_grad
    assert model.training
