from collections.abc import Callable

from torch import Tensor

import ai_explainer.number as aix
import testing
from ai_explainer.number.asset import Outputs, Viewport
from ai_explainer.number.world import NumberTrace


def test_frame_dark(
    trace: NumberTrace, loss_viewport: Viewport, outputs: Outputs, goldens: Outputs
) -> None:
    fig, ax = aix.asset.figure()
    aix.asset.frame_dark(ax, loss_viewport, title="Loss")
    aix.asset.plot_point(ax, trace.w, trace.loss)

    testing.assert_matches_golden(outputs, goldens, fig, "loss_frame_dark")


def test_plot_hill(
    trace: NumberTrace,
    loss_fn: Callable[[Tensor], Tensor],
    loss_viewport: Viewport,
    outputs: Outputs,
    goldens: Outputs,
) -> None:
    fig, ax = aix.asset.figure()
    aix.asset.frame_graph(ax, loss_viewport, title="Loss")
    aix.asset.plot_hill(ax, loss_fn)
    aix.asset.plot_point(ax, trace.w, trace.loss)

    testing.assert_matches_golden(outputs, goldens, fig, "loss_plot_hill")


def test_latex_loss(trace: NumberTrace, outputs: Outputs, goldens: Outputs) -> None:
    loss_steps = aix.asset.latex_loss(trace)
    loss = aix.asset.equation(loss_steps)

    testing.assert_matches_golden(outputs, goldens, [loss], "loss_latex_loss")


def test_latex_se(trace: NumberTrace, outputs: Outputs, goldens: Outputs) -> None:
    e_steps, se_steps = aix.asset.latex_se(trace)
    e = aix.asset.equation(e_steps)
    se = aix.asset.equation(se_steps)

    testing.assert_matches_golden(outputs, goldens, [e, se], "loss_latex_se")
