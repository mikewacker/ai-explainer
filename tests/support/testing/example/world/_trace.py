from collections.abc import Iterator
from typing import override

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from ai_explainer.common import world as common_world
from ai_explainer.common.world import Trace

from ._model import ExampleModel, get_loss_criterion


class ExampleTrace(Trace[ExampleModel]):
    """Trace through the example model, forward and backward."""

    w: Tensor
    x: Tensor
    y_hat: Tensor
    y: Tensor
    loss: Tensor
    y_hat_grad: Tensor
    w_grad: Tensor

    def __init__(self, model: ExampleModel, x: Tensor, y: Tensor) -> None:
        self.w = model.params()
        self.x = x.detach().clone()
        self.y = y.detach().clone()

    @override
    def _register_hooks(
        self, model: ExampleModel, criterion: nn.Module
    ) -> Iterator[RemovableHandle]:
        def capture_y_hat(y_hat: Tensor) -> None:
            self.y_hat = y_hat

        def capture_loss(loss: Tensor) -> None:
            self.loss = loss

        def capture_y_hat_grad(y_hat_grad: Tensor) -> None:
            self.y_hat_grad = y_hat_grad

        def capture_w_grad(w_grad: Tensor) -> None:
            self.w_grad = w_grad.squeeze(0)

        yield self._register_out_hook(model, capture_y_hat)
        yield self._register_out_hook(criterion, capture_loss)
        yield self._register_in_grad_hook(criterion, capture_y_hat_grad)
        yield self._register_param_grad_hook(model.linear.weight, capture_w_grad)


def trace(model: ExampleModel, x: Tensor, y: Tensor) -> ExampleTrace:
    """Trace a step through the example model, forward and backward."""
    criterion = get_loss_criterion()
    return common_world.trace(model, criterion, x, y, trace_factory=ExampleTrace)
