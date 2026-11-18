import pytest
import torch

import ai_explainer.iris as aix
import testing
from ai_explainer.iris.asset.binary import Diagram, Outputs, drawio


@pytest.mark.parametrize(
    "name",
    [
        "model_nn",
        "model_input",
        "model_hidden",
        "model_hidden-no-layer",
        "model_nn-hidden1",
        "model_hidden-a1",
        "model_hidden1",
        "model_hidden1-scalar",
        "model_output",
    ],
)
def test_drawio(name: str, outputs: Outputs, goldens: Outputs) -> None:
    diag = Diagram(drawio(name))

    testing.assert_matches_golden(outputs, goldens, diag, f"model_drawio_{name}")


def test_latex_linear1(outputs: Outputs, goldens: Outputs) -> None:
    w = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor(4.0)
    x = torch.tensor([5.0, 6.0, 7.0])
    linear1_steps = aix.asset.binary.latex_linear1(w, b, x)
    linear1 = aix.asset.binary.equation(linear1_steps, multiline=True, align_terms=True)

    testing.assert_matches_golden(outputs, goldens, [linear1], "model_latex_linear1")


def test_latex_linear(outputs: Outputs, goldens: Outputs) -> None:
    linear_steps = aix.asset.binary.latex_linear()
    linear = aix.asset.binary.equation(linear_steps, multiline=True, align_terms=True)

    testing.assert_matches_golden(outputs, goldens, [linear], "model_latex_linear")
