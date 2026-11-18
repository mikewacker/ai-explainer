import ai_explainer.wine as aix
import testing
from ai_explainer.wine.asset import Outputs
from ai_explainer.wine.world import WineTrace


def test_latex_grad(trace_data: WineTrace, outputs: Outputs, goldens: Outputs) -> None:
    grad_steps = aix.asset.latex_grad(trace_data)
    grad = aix.asset.equation(grad_steps, approx_steps=1, multiline=True)

    testing.assert_matches_golden(outputs, goldens, [grad], "slope_latex_grad")
