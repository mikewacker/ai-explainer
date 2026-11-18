from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, override

import torch
import torchmetrics
from torch import Tensor, nn

from ai_explainer.common import world as common_world
from ai_explainer.common.world import Snapshot


class IrisNn(nn.Module):
    """Neural network to classify iris flowers."""

    task: IrisTask
    hidden: HiddenLayer
    output: nn.Linear

    def __init__(
        self, *, task: IrisTask = "binary", hidden_size: int, seed: int | None = None
    ) -> None:
        """Create a neural network with one hidden layer."""
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.task = task
        out_size = 1 if task == "binary" else 3
        self.hidden = HiddenLayer(4, hidden_size)
        self.output = nn.Linear(hidden_size, out_size)

    @property
    def in_size(self) -> int:
        """Size of the input layer."""
        return self.hidden.linear.in_features

    @property
    def hidden_size(self) -> int:
        """Size of the hidden layer."""
        return self.hidden.linear.out_features

    @property
    def out_size(self) -> int:
        """Size of the output layer."""
        return self.output.out_features

    def params(self) -> IrisParams:
        """Copy the parameters."""
        w1, b1 = self.hidden.linear.weight, self.hidden.linear.bias
        w2, b2 = self.output.weight, self.output.bias
        if self.task == "binary":
            w2, b2 = w2.squeeze(0), b2.squeeze(0)
        w1, b1, w2, b2 = [w.detach().clone() for w in [w1, b1, w2, b2]]
        return IrisParams(w1=w1, b1=b1, w2=w2, b2=b2)

    @override
    def forward(self, x: Tensor) -> Tensor:
        self._check_x(x)
        a = self.hidden(x)
        logits = self.output(a)
        if self.task == "binary":
            logits = logits.squeeze(-1)
        return logits

    @staticmethod
    def _check_x(x: Tensor) -> None:
        if not 1 <= x.ndim <= 2 or x.shape[-1] != 4:
            raise ValueError(f"x.shape must be (m, 4) or (4,): {tuple(x.shape)}")


type IrisTask = Literal["binary", "multiclass"]


@dataclass(frozen=True, kw_only=True)
class IrisParams:
    """Parameters of the iris neural network."""

    w1: Tensor
    b1: Tensor
    w2: Tensor
    b2: Tensor


class HiddenLayer(nn.Module):
    """Hidden layer that uses ReLU."""

    linear: nn.Linear
    relu: nn.ReLU

    def __init__(self, in_size: int, out_size: int) -> None:
        """Create a hidden layer."""
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU()

    @override
    def forward(self, x: Tensor) -> Tensor:
        z = self.linear(x)
        return self.relu(z)


@dataclass(frozen=True, kw_only=True)
class IrisHyperparams:
    """Hyperparameters for the iris neural network."""

    hidden_size: int
    seed: int | None = None
    lr: float
    steps: int


IrisSnapshot = Snapshot[IrisParams]


def train(model: IrisNn, hparams: IrisHyperparams, x: Tensor, y: Tensor) -> list[IrisSnapshot]:
    """Train the iris neural network."""
    criterion = get_loss_criterion(task=model.task)
    optimizer = torch.optim.SGD(model.parameters(), lr=hparams.lr)
    return common_world.train(model, criterion, optimizer, x, y, steps=hparams.steps)


def test(model: IrisNn, x: Tensor, y: Tensor) -> Tensor:
    """Test the iris neural network, returning the accuracy."""
    metric = _get_metric(task=model.task)
    return common_world.test(model, metric, x, y)


def get_loss_criterion(*, task: IrisTask = "binary") -> nn.Module:
    """Get the loss criterion for comparing predictions to targets."""
    return nn.BCEWithLogitsLoss() if task == "binary" else nn.CrossEntropyLoss()


def _get_metric(*, task: IrisTask = "binary") -> nn.Module:
    num_classes = 3 if task == "multiclass" else None
    return torchmetrics.Accuracy(task=task, num_classes=num_classes)
