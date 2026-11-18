from collections.abc import Callable

from torch import Tensor

import ai_explainer.number as aix
import testing
from ai_explainer.number.asset import Outputs, Viewport
from ai_explainer.number.world import NumberTrace


def test_plot_tangent(
    trace: NumberTrace,
    loss_fn: Callable[[Tensor], Tensor],
    loss_viewport: Viewport,
    outputs: Outputs,
    goldens: Outputs,
) -> None:
    fig, ax = aix.asset.figure()
    aix.asset.frame_graph(ax, loss_viewport, title="Loss")
    aix.asset.plot_fn(ax, loss_fn)
    aix.asset.plot_tangent(ax, trace.w, trace.loss, trace.w_grad)

    testing.assert_matches_golden(outputs, goldens, fig, "slope_plot_tangent")


def test_latex_grad(trace: NumberTrace, outputs: Outputs, goldens: Outputs) -> None:
    grad_steps = aix.asset.latex_grad(trace)
    grad = aix.asset.equation(grad_steps)

    testing.assert_matches_golden(outputs, goldens, [grad], "slope_latex_grad")
