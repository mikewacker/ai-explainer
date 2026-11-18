from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class IrisData:
    """Sepal length and width, petal length and width, and species (target) of iris flowers."""

    x_train: Tensor
    x_test: Tensor
    y_train: Tensor
    y_test: Tensor


class IrisSpecies(IntEnum):
    """Species of an iris flower."""

    SETOSA = 0
    VERSICOLOR = 1
    VIRGINICA = 2


def load_data(variant: IrisSpecies | Literal["multiclass"], *, seed: int | None = None) -> IrisData:
    """Load the iris dataset."""
    x_np, y_np = datasets.load_iris(return_X_y=True)
    x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(
        x_np, y_np, test_size=0.2, stratify=y_np, random_state=seed
    )
    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    x_test = torch.tensor(x_test_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    y_test = torch.tensor(y_test_np, dtype=torch.long)
    if variant != "multiclass":
        y_train = _to_binary(y_train, variant)
        y_test = _to_binary(y_test, variant)
    return IrisData(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def _to_binary(y: Tensor, species: IrisSpecies) -> Tensor:
    return (y == int(species)).to(torch.float32)
