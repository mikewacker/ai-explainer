from functools import partial

import torch
from matplotlib.figure import Figure
from torch import Tensor, nn

import ai_explainer.iris as aix
from ai_explainer.iris.asset.binary import Diagram, Viewport, drawio

from ._trace import get_trace_sample


def show_z2_grad() -> list[str]:
    trace = get_trace_sample()
    grad_steps = aix.asset.binary.latex_z2_grad(trace)
    grad_fn = aix.asset.binary.equation(grad_steps[:2])
    grad = aix.asset.binary.equation(grad_steps, approx_steps=2)
    return [grad_fn, grad]


def show_bce_logit() -> Figure:
    trace = get_trace_sample()
    viewport = Viewport(x_var="z^{(2)}", xlim=(-4, 4), y_var="L", ylim=(0, 4))
    fig = aix.asset.binary.figure_empty()
    for y in torch.tensor([0.0, 1.0]):
        bce_logits_y = partial(_bce_logits, y=y)
        title = f"Sigmoid + Binary Cross-Entropy ($y = {y:g}$)"
        ax = fig.add_subplot(2, 1, int(y + 1))
        aix.asset.binary.frame_graph(ax, viewport, title=title, show_labels=True)
        aix.asset.binary.plot_fn(ax, bce_logits_y)
        if y == trace.y:
            aix.asset.binary.plot_slope(
                ax, trace.z2, trace.loss, trace.z2_grad, show_values=True, decimals=2
            )
    return fig


def show_output_grad(*, phase: int = 6) -> Diagram:
    trace = get_trace_sample()
    diag = Diagram(drawio("slope_output"))
    if phase >= 1:
        diag.fill_tensor("z", trace.z2_grad, decimals=2)
    if phase == 1:
        diag.highlight(fill="z_color")
    if phase == 2:
        diag.fill_tensor("w1", trace.w2_grad[0], decimals=2)
    elif phase >= 3:
        diag.fill_tensor("w", trace.w2_grad, decimals=2)
        diag.fill_tensor("b", trace.b2_grad, decimals=2)
    if 2 <= phase <= 3:
        diag.highlight(fill="param_color")
    if phase == 4:
        diag.fill_tensor("a1", trace.a_grad[0], decimals=2)
    elif phase >= 5:
        diag.fill_tensor("a", trace.a_grad, decimals=2)
    if 4 <= phase <= 5:
        diag.highlight(fill="in_color")
    return diag


def show_w21_grad() -> list[str]:
    trace = get_trace_sample()
    link_steps, backlink_steps, grad_steps = aix.asset.binary.latex_w2_grad(trace, j=1)
    link_fn = aix.asset.binary.equation(link_steps)
    backlink_fn = aix.asset.binary.equation(backlink_steps[:2])
    backlink = aix.asset.binary.equation(backlink_steps, approx_steps=1)
    grad_fn = aix.asset.binary.equation(grad_steps[:2])
    grad = aix.asset.binary.equation(grad_steps, approx_steps=2)
    return [link_fn, backlink_fn, backlink, grad_fn, grad]


def show_a1_grad() -> list[str]:
    trace = get_trace_sample()
    link_steps, backlink_steps, grad_steps = aix.asset.binary.latex_a_grad(trace, j=1)
    link_fn = aix.asset.binary.equation(link_steps)
    backlink_fn = aix.asset.binary.equation(backlink_steps[:2])
    backlink = aix.asset.binary.equation(backlink_steps, approx_steps=1)
    grad = aix.asset.binary.equation(grad_steps, approx_steps=2)
    return [link_fn, backlink_fn, backlink, grad]


def show_hidden_a1_grad(*, phase: int = 3) -> Diagram:
    trace = get_trace_sample()
    diag = Diagram(drawio("slope_hidden-a1"))
    diag.fill_tensor("a", trace.a_grad, decimals=2)
    if phase == 0:
        diag.highlight(fill="out_color")
    if phase >= 1:
        diag.fill_tensor("z", trace.z1_grad, decimals=2)
    if phase == 1:
        diag.highlight(fill="z_color")
    if phase >= 2:
        diag.fill_tensor("w", trace.w1_grad, decimals=2)
        diag.fill_tensor("b", trace.b1_grad, decimals=2)
    if phase == 2:
        diag.highlight(fill="param_color")
    return diag


def show_z11_grad() -> list[str]:
    trace = get_trace_sample()
    link_steps, backlink_steps, grad_steps = aix.asset.binary.latex_z1_grad(trace, j=1)
    link_fn = aix.asset.binary.equation(link_steps)
    backlink_fn = aix.asset.binary.equation(backlink_steps[:2])
    backlink = aix.asset.binary.equation(backlink_steps)
    grad = aix.asset.binary.equation(grad_steps, approx_steps=2)
    return [link_fn, backlink_fn, backlink, grad]


def show_hidden_grad() -> Diagram:
    trace = get_trace_sample()
    diag = Diagram(drawio("slope_hidden"))
    diag.fill_tensor("a", trace.a_grad, decimals=2)
    diag.fill_tensor("z", trace.z1_grad, decimals=2)
    diag.fill_tensor("w", trace.w1_grad, decimals=2)
    diag.fill_tensor("b", trace.b1_grad, decimals=2)
    return diag


def _bce_logits(y_hat: Tensor, y: Tensor) -> Tensor:
    y = y.expand_as(y_hat)
    return nn.functional.binary_cross_entropy_with_logits(y_hat, y, reduction="none")
