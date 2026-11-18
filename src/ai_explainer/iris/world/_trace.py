from collections.abc import Iterator
from typing import override

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from ai_explainer.common import world as common_world
from ai_explainer.common.world import Trace

from ._model import IrisNn, get_loss_criterion


class IrisTrace(Trace[IrisNn]):
    """Trace through the iris neural network, forward and backward."""

    w1: Tensor
    b1: Tensor
    w2: Tensor
    b2: Tensor
    x: Tensor
    z1: Tensor
    a: Tensor
    z2: Tensor
    y_hat: Tensor
    y: Tensor
    loss: Tensor
    z2_grad: Tensor
    w2_grad: Tensor
    b2_grad: Tensor
    a_grad: Tensor
    z1_grad: Tensor
    w1_grad: Tensor
    b1_grad: Tensor

    def __init__(self, model: IrisNn, x: Tensor, y: Tensor) -> None:
        params = model.params()
        self.w1 = params.w1
        self.b1 = params.b1
        self.w2 = params.w2
        self.b2 = params.b2
        self.x = x.detach().clone()
        if model.task == "multiclass":
            y = nn.functional.one_hot(y, model.out_size).to(torch.float32)
        self.y = y.detach().clone()

    @override
    def _register_hooks(self, model: IrisNn, criterion: nn.Module) -> Iterator[RemovableHandle]:
        def capture_z1(z1: Tensor) -> None:
            self.z1 = z1

        def capture_a(a: Tensor) -> None:
            self.a = a

        def capture_z2_y_hat(z2: Tensor) -> None:
            self.z2 = z2
            if model.task == "binary":
                self.y_hat = torch.sigmoid(z2)
            else:
                self.y_hat = torch.softmax(z2, dim=-1)

        def capture_loss(loss: Tensor) -> None:
            self.loss = loss

        def capture_z2_grad(z2_grad: Tensor) -> None:
            self.z2_grad = z2_grad

        def capture_w2_grad(w2_grad: Tensor) -> None:
            if model.task == "binary":
                w2_grad = w2_grad.squeeze(0)
            self.w2_grad = w2_grad

        def capture_b2_grad(b2_grad: Tensor) -> None:
            if model.task == "binary":
                b2_grad = b2_grad.squeeze(0)
            self.b2_grad = b2_grad

        def capture_a_grad(a_grad: Tensor) -> None:
            self.a_grad = a_grad

        def capture_z1_grad(z1_grad: Tensor) -> None:
            self.z1_grad = z1_grad

        def capture_w1_grad(w1_grad: Tensor) -> None:
            self.w1_grad = w1_grad

        def capture_b1_grad(b1_grad: Tensor) -> None:
            self.b1_grad = b1_grad

        yield self._register_out_hook(model.hidden.linear, capture_z1)
        yield self._register_out_hook(model.hidden.relu, capture_a)
        yield self._register_out_hook(model, capture_z2_y_hat)
        yield self._register_out_hook(criterion, capture_loss)
        yield self._register_in_grad_hook(criterion, capture_z2_grad)
        yield self._register_param_grad_hook(model.output.weight, capture_w2_grad)
        yield self._register_param_grad_hook(model.output.bias, capture_b2_grad)
        yield self._register_in_grad_hook(model.output, capture_a_grad)
        yield self._register_in_grad_hook(model.hidden.relu, capture_z1_grad)
        yield self._register_param_grad_hook(model.hidden.linear.weight, capture_w1_grad)
        yield self._register_param_grad_hook(model.hidden.linear.bias, capture_b1_grad)


def trace(model: IrisNn, x: Tensor, y: Tensor) -> IrisTrace:
    """Trace a step through the iris neural network, forward and backward."""
    criterion = get_loss_criterion(task=model.task)
    return common_world.trace(model, criterion, x, y, trace_factory=IrisTrace)
