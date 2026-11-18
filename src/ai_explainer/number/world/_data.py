from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class NumberData:
    """Target for the number-guessing game."""

    y: Tensor


def load_data() -> NumberData:
    """Load the target for the number-guessing game."""
    return NumberData(y=torch.tensor(3.5))
