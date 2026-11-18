from collections.abc import Iterator
from typing import override

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from ai_explainer.common import world as common_world
from ai_explainer.common.world import Trace

from ._model import NumberModel, get_loss_criterion


class NumberTrace(Trace[NumberModel]):
    """Trace through the number model, forward and backward."""

    w: Tensor
    y_hat: Tensor
    y: Tensor
    loss: Tensor
    w_grad: Tensor

    def __init__(self, model: NumberModel, _x: Tensor, y: Tensor) -> None:
        self.w = model.params()
        self.y = y.detach().clone()

    @override
    def _register_hooks(
        self, model: NumberModel, criterion: nn.Module
    ) -> Iterator[RemovableHandle]:
        def capture_y_hat(y_hat: Tensor) -> None:
            self.y_hat = y_hat

        def capture_loss(loss: Tensor) -> None:
            self.loss = loss

        def capture_w_grad(w_grad: Tensor) -> None:
            self.w_grad = w_grad

        yield self._register_out_hook(model, capture_y_hat)
        yield self._register_out_hook(criterion, capture_loss)
        yield self._register_param_grad_hook(model.w, capture_w_grad)


def trace(model: NumberModel, y: Tensor) -> NumberTrace:
    """Trace a step through the number model, forward and backward."""
    criterion = get_loss_criterion()
    x = torch.empty(0)
    return common_world.trace(model, criterion, x, y, trace_factory=NumberTrace)
