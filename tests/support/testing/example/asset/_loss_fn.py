from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor, nn

import testing.example as aix
from testing.example.asset import LossCrossFn
from testing.example.world import ExampleModel


def loss_cross_fn(
    x: Tensor, y: Tensor, model_factory: Callable[[], ExampleModel], *, feature: Literal[1, 2]
) -> LossCrossFn[Tensor]:
    """Create a 2D cross-section of the loss function for one parameter."""
    criterion = aix.world.get_loss_criterion()

    def set_param(model: ExampleModel, w: Tensor) -> None:
        mask = torch.ones_like(model.linear.weight, dtype=torch.bool)
        mask[0, feature - 1] = False
        model.linear.weight = nn.Parameter(torch.where(mask, model.linear.weight, w))

    return LossCrossFn(x, y, model_factory, criterion, param_setter=set_param)
