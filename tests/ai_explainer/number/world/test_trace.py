import torch

import ai_explainer.number as aix
import testing
from ai_explainer.number.world import NumberData, NumberHyperparams, NumberModel


def test_trace(data: NumberData, hparams: NumberHyperparams) -> None:
    model = NumberModel(w=hparams.w_init)
    trace = aix.world.trace(model, data.y)

    testing.assert_tensors_equal(trace.w, torch.tensor(5.0))
    testing.assert_tensors_equal(trace.y_hat, torch.tensor(5.0))
    testing.assert_tensors_equal(trace.y, torch.tensor(3.5))
    testing.assert_tensors_close(trace.loss, torch.tensor(2.25))
    testing.assert_tensors_close(trace.w_grad, torch.tensor(3.0))
    assert trace.y is not data.y
