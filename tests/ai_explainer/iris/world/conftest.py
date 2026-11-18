import pytest
import torch

from ai_explainer.iris.world import IrisData, IrisHyperparams


@pytest.fixture
def data() -> IrisData:
    generator = torch.Generator().manual_seed(42)
    return IrisData(
        x_train=torch.rand(8, 4, generator=generator),
        x_test=torch.rand(2, 4, generator=generator),
        y_train=torch.randint(0, 2, (8,), dtype=torch.float32, generator=generator),
        y_test=torch.randint(0, 2, (2,), dtype=torch.float32, generator=generator),
    )


@pytest.fixture
def data_multiclass() -> IrisData:
    generator = torch.Generator().manual_seed(42)
    return IrisData(
        x_train=torch.rand(8, 4, generator=generator),
        x_test=torch.rand(2, 4, generator=generator),
        y_train=torch.randint(0, 3, (8,), dtype=torch.long, generator=generator),
        y_test=torch.randint(0, 3, (2,), dtype=torch.long, generator=generator),
    )


@pytest.fixture
def hparams() -> IrisHyperparams:
    return IrisHyperparams(hidden_size=5, seed=42, lr=0.01, steps=3)
