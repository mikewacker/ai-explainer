from dataclasses import dataclass

import torch
from sklearn import datasets
from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class WineData:
    """Malic acid, proanthocyanidins, and proline (target) of wines."""

    x: Tensor
    y: Tensor


def load_data() -> WineData:
    """Load the wine dataset."""
    x_np, _ = datasets.load_wine(return_X_y=True)
    x = torch.tensor(x_np[:, [1, 8]], dtype=torch.float32)  # malic acid, proanthocyanidins
    y = torch.tensor(x_np[:, 12], dtype=torch.float32)  # proline
    return WineData(x=x, y=y)
