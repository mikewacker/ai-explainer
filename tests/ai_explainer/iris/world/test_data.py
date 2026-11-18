import torch

import ai_explainer.iris as aix
import testing
from ai_explainer.iris.world import IrisSpecies


def test_load_data() -> None:
    data = aix.world.load_data(IrisSpecies.VIRGINICA, seed=42)

    assert data.x_train.shape == (120, 4)
    assert data.x_test.shape == (30, 4)
    assert data.y_train.shape == (120,)
    assert data.y_test.shape == (30,)
    assert data.x_train.dtype == torch.float32
    assert data.x_test.dtype == torch.float32
    assert data.y_train.dtype == torch.float32
    assert data.y_test.dtype == torch.float32
    data_multiclass = aix.world.load_data(variant="multiclass", seed=42)
    testing.assert_tensors_equal(
        data.y_train == 1.0, data_multiclass.y_train == int(IrisSpecies.VIRGINICA)
    )
    testing.assert_tensors_equal(
        data.y_test == 1.0, data_multiclass.y_test == int(IrisSpecies.VIRGINICA)
    )


def test_load_data__multiclass() -> None:
    data = aix.world.load_data(variant="multiclass", seed=42)

    assert data.x_train.shape == (120, 4)
    assert data.x_test.shape == (30, 4)
    assert data.y_train.shape == (120,)
    assert data.y_test.shape == (30,)
    assert data.x_train.dtype == torch.float32
    assert data.x_test.dtype == torch.float32
    assert data.y_train.dtype == torch.long
    assert data.y_test.dtype == torch.long
