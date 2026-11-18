from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn

from ai_explainer.common import world as common_world
from ai_explainer.common.world import Snapshot


class NumberModel(nn.Module):
    """Model to guess a number."""

    w: nn.Parameter

    def __init__(self, *, w: Tensor) -> None:
        """Create a model from the parameter for the initial guess."""
        super().__init__()
        self._check_w(w)
        w = w.detach().clone()
        self.w = nn.Parameter(w)

    def params(self) -> Tensor:
        """Copy the parameters."""
        return self.w.detach().clone()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.w

    @staticmethod
    def _check_w(w: Tensor) -> None:
        if w.shape != ():
            raise ValueError(f"w.shape must be (): {tuple(w.shape)}")


@dataclass(frozen=True, kw_only=True)
class NumberHyperparams:
    """Hyperparameters for the number model."""

    w_init: Tensor
    lr: float
    steps: int


NumberSnapshot = Snapshot[Tensor]


def train(model: NumberModel, hparams: NumberHyperparams, y: Tensor) -> list[NumberSnapshot]:
    """Train the number model."""
    criterion = get_loss_criterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=hparams.lr)
    x = torch.empty(0)
    return common_world.train(model, criterion, optimizer, x, y, steps=hparams.steps)


def get_loss_criterion() -> nn.Module:
    """Get the loss criterion for comparing predictions to targets."""
    return nn.MSELoss()
