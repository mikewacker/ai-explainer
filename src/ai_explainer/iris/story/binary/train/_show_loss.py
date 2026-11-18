from functools import partial

import torch
from matplotlib.figure import Figure
from torch import Tensor, nn

import ai_explainer.iris as aix
from ai_explainer.iris.asset.binary import Diagram, Viewport, drawio

from ._trace import get_trace_sample


def show_loss_sample_neurons() -> Diagram:
    return Diagram(drawio("loss_sample-neurons"))


def show_loss_sample(*, phase: int = 5) -> Diagram:
    trace = get_trace_sample()
    diag = Diagram(drawio("loss_sample"))
    if phase >= 1:
        diag.fill_tensor("w1", trace.w1, decimals=2)
        diag.fill_tensor("b1", trace.b1, decimals=2)
        diag.fill_tensor("w2", trace.w2, decimals=2)
        diag.fill_tensor("b2", trace.b2, decimals=2)
    if phase == 1:
        diag.highlight(fill="param_color")
    if phase >= 2:
        diag.fill_tensor("x", trace.x, decimals=1)
        diag.fill_tensor("yhat", trace.y_hat, decimals=2)
    if phase == 2:
        diag.highlight(fill="prediction_color")
    if phase >= 3:
        diag.fill_tensor("y", trace.y)
    if phase == 3:
        diag.highlight(fill="target_color")
    if phase >= 4:
        diag.fill_tensor("L", trace.loss, decimals=2)
    if phase == 4:
        diag.highlight(fill="loss_color")
    return diag


def show_bce() -> Figure:
    trace = get_trace_sample()
    viewport = Viewport(x_var=r"\hat{y}", xlim=(0, 1), y_var=r"L", ylim=(0, 3))
    fig = aix.asset.binary.figure_empty()
    for y in torch.tensor([0.0, 1.0]):
        bce_y = partial(_bce, y=y)
        title = f"BCE ($y = {y:g}$)"
        ax = fig.add_subplot(1, 2, int(y + 1))
        aix.asset.binary.frame_graph(ax, viewport, title=title, show_labels=True)
        aix.asset.binary.plot_fn(ax, bce_y)
        if y == trace.y:
            aix.asset.binary.plot_point(ax, trace.y_hat, trace.loss, show_values=True, decimals=2)
    return fig


def show_loss_sample_matrix() -> Diagram:
    return Diagram(drawio("loss_sample-matrix"))


def show_loss(*, show_io: bool = False) -> Diagram:
    diag = Diagram(drawio("loss_data"))
    if show_io:
        diag.highlight_io()
    return diag


def show_linear() -> list[str]:
    linear_steps = aix.asset.binary.latex_linear()
    linear = aix.asset.binary.equation(linear_steps)
    return [linear]


def _bce(y_hat: Tensor, y: Tensor) -> Tensor:
    y = y.expand_as(y_hat)
    return nn.functional.binary_cross_entropy(y_hat, y, reduction="none")
