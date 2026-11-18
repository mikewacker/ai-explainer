import pytest

import testing
from ai_explainer.iris.asset.binary import Diagram, Outputs, drawio


@pytest.mark.parametrize(
    "name",
    [
        "loss_sample",
        "loss_sample-neurons",
        "loss_data",
        "loss_sample-matrix",
    ],
)
def test_drawio(name: str, outputs: Outputs, goldens: Outputs) -> None:
    diag = Diagram(drawio(name))

    testing.assert_matches_golden(outputs, goldens, diag, f"loss_drawio_{name}")
