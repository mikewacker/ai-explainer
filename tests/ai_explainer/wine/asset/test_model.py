import pytest

import ai_explainer.wine as aix
import testing
from ai_explainer.wine.asset import Diagram, Outputs, drawio
from ai_explainer.wine.world import WineTrace


@pytest.mark.parametrize(
    "name",
    [
        "model_blackbox-wine",
        "model_blackbox",
        "model_hand",
        "model_ai",
    ],
)
def test_drawio(name: str, outputs: Outputs, goldens: Outputs) -> None:
    diag = Diagram(drawio(name))

    testing.assert_matches_golden(outputs, goldens, diag, f"model_drawio_{name}")


def test_latex_prediction_hand(trace_sample: WineTrace, outputs: Outputs, goldens: Outputs) -> None:
    prediction_steps = aix.asset.latex_prediction_hand(trace_sample)
    prediction = aix.asset.equation(prediction_steps, multiline=True, align_terms=True)

    testing.assert_matches_golden(outputs, goldens, [prediction], "model_latex_prediction_hand")


def test_latex_prediction_no_params(
    trace_sample: WineTrace, outputs: Outputs, goldens: Outputs
) -> None:
    prediction_steps = aix.asset.latex_prediction_no_params(trace_sample)
    prediction = aix.asset.equation(prediction_steps, multiline=True, align_terms=True)

    testing.assert_matches_golden(
        outputs, goldens, [prediction], "model_latex_prediction_no_params"
    )
