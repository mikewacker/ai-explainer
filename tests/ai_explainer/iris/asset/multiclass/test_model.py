import pytest

import testing
from ai_explainer.iris.asset.multiclass import Diagram, Outputs, drawio


@pytest.mark.parametrize(
    "name",
    [
        "model_nn",
        "model_output",
    ],
)
def test_drawio(name: str, outputs: Outputs, goldens: Outputs) -> None:
    diag = Diagram(drawio(name))

    testing.assert_matches_golden(outputs, goldens, diag, f"model_drawio_{name}")
