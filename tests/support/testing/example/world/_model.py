from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn

from ai_explainer.common import world as common_world
from ai_explainer.common.world import Snapshot


class ExampleModel(nn.Module):
    """Example model for testing."""

    def __init__(self) -> None:
        """Create a model with seeded parameters."""
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        self.linear.weight = nn.Parameter(torch.tensor([[5.0, 2.0]]))

    def params(self) -> Tensor:
        """Copy the parameters."""
        return self.linear.weight.squeeze(0).detach().clone()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x).squeeze(-1)


@dataclass(frozen=True, kw_only=True)
class ExampleHyperparams:
    """Hyperparameters for the example model."""

    lr: float
    steps: int


ExampleSnapshot = Snapshot[Tensor]


def train(
    model: ExampleModel, hparams: ExampleHyperparams, x: Tensor, y: Tensor
) -> list[ExampleSnapshot]:
    """Train the example model."""
    criterion = get_loss_criterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=hparams.lr)
    return common_world.train(model, criterion, optimizer, x, y, steps=hparams.steps)


def test(model: ExampleModel, x: Tensor, y: Tensor) -> Tensor:
    """Test the example model, returning the mean squared error."""
    metric = nn.MSELoss()
    return common_world.test(model, metric, x, y)


def get_loss_criterion() -> nn.Module:
    """Get the loss criterion for comparing predictions to targets."""
    return nn.MSELoss()
