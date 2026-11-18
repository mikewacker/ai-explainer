from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn

from ai_explainer.common import world as common_world
from ai_explainer.common.world import Snapshot


class WineModel(nn.Module):
    """Model for predicting proline in wine."""

    linear: nn.Linear

    def __init__(self, *, w: Tensor) -> None:
        """Create a model from initial parameters for malic acid and proanthocyanidins."""
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        self._check_w(w)
        w = w.unsqueeze(0).detach().clone()
        self.linear.weight = nn.Parameter(w)

    def params(self) -> Tensor:
        """Copy the parameters."""
        return self.linear.weight.squeeze(0).detach().clone()

    @override
    def forward(self, x: Tensor) -> Tensor:
        self._check_x(x)
        return self.linear(x).squeeze(-1)

    @staticmethod
    def _check_w(w: Tensor) -> None:
        if w.shape != (2,):
            raise ValueError(f"w.shape must be (2,): {tuple(w.shape)}")

    @staticmethod
    def _check_x(x: Tensor) -> None:
        if not 1 <= x.ndim <= 2 or x.shape[-1] != 2:
            raise ValueError(f"x.shape must be (m, 2) or (2,): {tuple(x.shape)}")


@dataclass(frozen=True, kw_only=True)
class WineHyperparams:
    """Hyperparameters for the wine model."""

    w_init: Tensor
    lr: float
    steps: int


WineSnapshot = Snapshot[Tensor]


def train(model: WineModel, hparams: WineHyperparams, x: Tensor, y: Tensor) -> list[WineSnapshot]:
    """Train the wine model."""
    criterion = get_loss_criterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=hparams.lr)
    return common_world.train(model, criterion, optimizer, x, y, steps=hparams.steps)


def get_loss_criterion() -> nn.Module:
    """Get the loss criterion for comparing predictions to targets."""
    return nn.MSELoss()
