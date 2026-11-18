from collections.abc import Callable

import torch
from torch import Tensor, nn

import ai_explainer.number as aix
from ai_explainer.number.asset import LossCrossFn
from ai_explainer.number.world import NumberModel


def loss_fn(y: Tensor) -> Callable[[Tensor], Tensor]:
    """Create the loss function for the parameter."""
    x = torch.empty(0)

    def create_model() -> NumberModel:
        return NumberModel(w=torch.tensor(0.0))

    criterion = aix.world.get_loss_criterion()

    def set_param(model: NumberModel, w: Tensor) -> None:
        model.w = nn.Parameter(w)

    return LossCrossFn(x, y, create_model, criterion, param_setter=set_param)
