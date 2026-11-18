import torch

import ai_explainer.number as aix
import testing
from ai_explainer.number.world import NumberData


def test_loss_fn(data: NumberData) -> None:
    loss_fn = aix.asset.loss_fn(data.y)
    w = torch.tensor([1.0, 2.0, 3.0])
    loss = loss_fn(w)

    testing.assert_tensors_close(loss, torch.tensor([0.25, 0.25, 2.25]))
