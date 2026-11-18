from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

import testing
import testing.example as aix
from testing.example.asset import Outputs, Viewport


@dataclass
class FnTrace:
    x: Tensor
    y: Tensor
    x_grad: Tensor

    def __init__(self, *, x: Tensor) -> None:
        self.x = x
        self.y = x**2
        self.x_grad = 2 * x


@pytest.fixture
def trace1() -> FnTrace:
    return FnTrace(x=torch.tensor(1.5))


@pytest.fixture
def trace2() -> FnTrace:
    return FnTrace(x=torch.tensor(-1.0))


@pytest.fixture
def trace3() -> FnTrace:
    return FnTrace(x=torch.tensor(0.0))


@pytest.fixture
def fn() -> Callable[[Tensor], Tensor]:
    return lambda x: x**2


@pytest.fixture
def fn_viewport() -> Viewport:
    return Viewport(x_var="x", xlim=(-4, 4), y_var="y", ylim=(-1, 7))


def test_plot_fn_and_point(
    trace1: FnTrace,
    fn: Callable[[Tensor], Tensor],
    fn_viewport: Viewport,
    outputs: Outputs,
    goldens: Outputs,
) -> None:
    fig, ax = aix.asset.figure()
    aix.asset.frame_graph(ax, fn_viewport)
    aix.asset.plot_fn(ax, fn)
    aix.asset.plot_point(ax, trace1.x, trace1.y)

    testing.assert_matches_golden(outputs, goldens, fig, "plot_fn_and_point")


def test_plot_fn_and_point__show_values(
    trace1: FnTrace,
    trace2: FnTrace,
    fn: Callable[[Tensor], Tensor],
    fn_viewport: Viewport,
    outputs: Outputs,
    goldens: Outputs,
) -> None:
    fig, ax = aix.asset.figure()
    aix.asset.frame_graph(ax, fn_viewport)
    aix.asset.plot_fn(ax, fn)
    aix.asset.plot_point(ax, trace1.x, trace1.y, show_values=True)
    aix.asset.plot_point(ax, trace2.x, trace2.y, show_values=True, ha="right", va="top")

    testing.assert_matches_golden(outputs, goldens, fig, "plot_fn_and_point-show_values")


def test_plot_point__show_approx_values(trace1: FnTrace, fn_viewport: Viewport) -> None:
    _, ax = aix.asset.figure()
    aix.asset.frame_graph(ax, fn_viewport)
    aix.asset.plot_point(ax, trace1.x, trace1.y, show_values=True, decimals=2)

    assert ax.texts[0].get_text() == r"$(x \approx 1.50, y \approx 2.25)$"


def test_plot_fn_and_slope(
    trace1: FnTrace,
    trace2: FnTrace,
    trace3: FnTrace,
    fn: Callable[[Tensor], Tensor],
    fn_viewport: Viewport,
    outputs: Outputs,
    goldens: Outputs,
) -> None:
    fig, ax = aix.asset.figure()
    aix.asset.frame_graph(ax, fn_viewport)
    aix.asset.plot_fn(ax, fn)
    aix.asset.plot_slope(ax, trace1.x, trace1.y, trace1.x_grad)
    aix.asset.plot_slope(ax, trace2.x, trace2.y, trace2.x_grad)
    aix.asset.plot_slope(ax, trace3.x, trace3.y, trace3.x_grad)

    testing.assert_matches_golden(outputs, goldens, fig, "plot_fn_and_slope")


def test_plot_fn_and_slope__show_values(
    trace1: FnTrace,
    trace2: FnTrace,
    fn: Callable[[Tensor], Tensor],
    fn_viewport: Viewport,
    outputs: Outputs,
    goldens: Outputs,
) -> None:
    fig, ax = aix.asset.figure()
    aix.asset.frame_graph(ax, fn_viewport)
    aix.asset.plot_fn(ax, fn)
    aix.asset.plot_slope(ax, trace1.x, trace1.y, trace1.x_grad, show_values=True)
    aix.asset.plot_slope(
        ax, trace2.x, trace2.y, trace2.x_grad, show_values=True, ha="right", va="top"
    )

    testing.assert_matches_golden(outputs, goldens, fig, "plot_fn_and_slope-show_values")


def test_plot_slope__show_approx_values(trace1: FnTrace, fn_viewport: Viewport) -> None:
    _, ax = aix.asset.figure()
    aix.asset.frame_graph(ax, fn_viewport)
    aix.asset.plot_slope(ax, trace1.x, trace1.y, trace1.x_grad, show_values=True, decimals=2)

    assert ax.texts[1].get_text() == (
        r"$(x \approx 1.50, \frac{\partial y}{\partial x} \approx 3.00)$"
    )


def test_sample_fn(fn: Callable[[Tensor], Tensor]) -> None:
    x, y = aix.asset.sample_fn(fn, (-5, 5))

    testing.assert_tensors_equal(x[[0, 1, -1]], torch.tensor([-5.0, -4.9, 5.0]))
    assert x.shape == (101,)
    testing.assert_tensors_equal(y, fn(x))
