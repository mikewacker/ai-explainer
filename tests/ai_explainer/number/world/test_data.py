import torch

import ai_explainer.number as aix
import testing


def test_load_data() -> None:
    data = aix.world.load_data()

    testing.assert_tensors_equal(data.y, torch.tensor(3.5))
