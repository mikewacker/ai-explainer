from dataclasses import dataclass
from typing import Protocol, Self

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


class Model[P](Protocol):
    """Model with parameters."""

    def params(self) -> P:
        """Copy the parameters."""

    def train(self) -> Self:
        """Set the model in training mode."""

    def __call__(self, x: Tensor, /) -> Tensor:
        """Feed forward."""


@dataclass(frozen=True)
class Snapshot[P]:
    """Snapshot of the loss and parameters."""

    loss: Tensor
    params: P


def train[P](
    model: Model[P],
    criterion: nn.Module,
    optimizer: Optimizer,
    x: Tensor,
    y: Tensor,
    *,
    steps: int,
) -> list[Snapshot[P]]:
    """Train the model, returning a list of snapshots."""
    model.train()
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        snapshot = Snapshot(loss.detach().clone(), model.params())
        history.append(snapshot)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        y_hat = model(x)
        loss = criterion(y_hat, y)
        snapshot = Snapshot(loss.detach().clone(), model.params())
        history.append(snapshot)
    return history


def test(model: nn.Module, metric: nn.Module, x: Tensor, y: Tensor) -> Tensor:
    """Test the model, returning the metric."""
    model.eval()
    with torch.no_grad():
        y_hat = model(x)
        result = metric(y_hat, y)
        return result.detach().clone()
