import pytest

import ai_explainer.iris as aix
import testing
from ai_explainer.iris.asset.multiclass import Diagram, Outputs, drawio
from ai_explainer.iris.world import IrisTrace


@pytest.mark.parametrize(
    "name",
    [
        "slope_output",
    ],
)
def test_drawio(name: str, outputs: Outputs, goldens: Outputs) -> None:
    diag = Diagram(drawio(name))

    testing.assert_matches_golden(outputs, goldens, diag, f"slope_drawio_{name}")


def test_latex_z2_grad(trace: IrisTrace, outputs: Outputs, goldens: Outputs) -> None:
    grad_steps = aix.asset.multiclass.latex_z2_grad(trace)
    grad = aix.asset.multiclass.equation(grad_steps, approx_steps=2)

    testing.assert_matches_golden(outputs, goldens, [grad], "slope_latex_z2_grad")


def test_latex_a_grad(trace: IrisTrace, outputs: Outputs, goldens: Outputs) -> None:
    link_steps, backlink_steps, grad_steps = aix.asset.multiclass.latex_a_grad(trace, j=3)
    link = aix.asset.multiclass.equation(link_steps)
    backlink = aix.asset.multiclass.equation(backlink_steps, approx_steps=1)
    grad = aix.asset.multiclass.equation(
        grad_steps, approx_steps=2, multiline=True, align_terms=True
    )

    testing.assert_matches_golden(outputs, goldens, [link, backlink, grad], "slope_latex_a_grad")
