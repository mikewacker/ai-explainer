from collections.abc import Callable

import pytest
import torch
from torch import Tensor

import ai_explainer.number as aix
import testing
from ai_explainer.number.asset import NumberVisualSnapshot, Outputs, Viewport
from ai_explainer.number.world import NumberSnapshot, NumberTrace


@pytest.fixture
def snapshot1() -> NumberSnapshot:
    return NumberSnapshot(loss=torch.tensor(2.25), params=torch.tensor(3.0))


@pytest.fixture
def snapshot2() -> NumberSnapshot:
    return NumberSnapshot(loss=torch.tensor(1.0), params=torch.tensor(2.5))


class TestNumberVisualSnapshot:
    def test_plot(
        self, loss_viewport: Viewport, snapshot1: NumberSnapshot, outputs: Outputs, goldens: Outputs
    ) -> None:
        fig, ax = aix.asset.figure()
        NumberVisualSnapshot(ax, loss_viewport, snapshot1)

        testing.assert_matches_golden(outputs, goldens, fig, "learn_plot_visual_snapshot")

    def test_animate(
        self,
        loss_viewport: Viewport,
        snapshot1: NumberSnapshot,
        snapshot2: NumberSnapshot,
        outputs: Outputs,
        goldens: Outputs,
    ) -> None:
        fig, ax = aix.asset.figure()
        vsnapshot = NumberVisualSnapshot(ax, loss_viewport, snapshot1)
        artists = vsnapshot.animate(snapshot2)

        testing.assert_matches_golden(outputs, goldens, fig, "learn_animate_visual_snapshot")
        assert len(artists) == 2


def test_plot_step(
    trace: NumberTrace,
    loss_fn: Callable[[Tensor], Tensor],
    loss_viewport: Viewport,
    outputs: Outputs,
    goldens: Outputs,
) -> None:
    fig, ax = aix.asset.figure()
    aix.asset.plot_step(ax, loss_viewport, loss_fn, trace.w, trace.w + 0.02)

    testing.assert_matches_golden(outputs, goldens, fig, "learn_plot_step")


def test_latex_sgd_step(trace: NumberTrace, outputs: Outputs, goldens: Outputs) -> None:
    grad_steps, delta_steps, update_steps = aix.asset.latex_sgd_step(trace, 0.01)
    grad = aix.asset.equation(grad_steps)
    delta = aix.asset.equation(delta_steps)
    update = aix.asset.equation(update_steps)

    testing.assert_matches_golden(outputs, goldens, [grad, delta, update], "learn_latex_sgd_step")
