import torch
from matplotlib.figure import Figure

import ai_explainer.number as aix

from ._loss_fn import loss_fn, loss_viewport
from ._trace import get_trace


def show_tangent() -> Figure:
    trace = get_trace()
    fig = aix.asset.figure_empty(figsize=(7, 3.5))
    ax_light = fig.add_subplot(1, 2, 1)
    aix.asset.frame_graph(ax_light, loss_viewport(), title="Light")
    aix.asset.plot_hill(ax_light, loss_fn())
    aix.asset.plot_tangent(ax_light, trace.w, trace.loss, trace.w_grad)
    ax_dark = fig.add_subplot(1, 2, 2)
    aix.asset.frame_dark(ax_dark, loss_viewport(), title="Dark")
    aix.asset.plot_tangent(ax_dark, trace.w, trace.loss, trace.w_grad)
    return fig


def show_grad() -> list[str]:
    trace = get_trace()
    grad_steps = aix.asset.latex_grad(trace)
    grad_sym = aix.asset.equation(grad_steps[:1])
    grad_fn = aix.asset.equation(grad_steps[:2])
    grad = aix.asset.equation(grad_steps)
    return [grad_sym, grad_fn, grad]


def show_up() -> Figure:
    traces = [get_trace(w) for w in torch.tensor([2.0, 3.0, 4.0, 5.0])]
    fig, ax = aix.asset.figure()
    aix.asset.frame_graph(ax, loss_viewport(), title="Loss")
    aix.asset.plot_fn(ax, loss_fn())
    for trace in traces:
        ha = "left" if trace.w >= trace.y else "right"
        aix.asset.plot_slope(ax, trace.w, trace.loss, trace.w_grad, show_values=True, ha=ha)
    return fig
