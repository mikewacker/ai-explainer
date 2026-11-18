from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class ExampleData:
    """Example data for testing."""

    x: Tensor
    y: Tensor


def load_data() -> ExampleData:
    """Load the example data."""
    return ExampleData(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        y=torch.tensor([10.0, 20.0, 30.0]),
    )
