from matplotlib.figure import Figure

import ai_explainer.number as aix

from ._loss_fn import loss_fn, loss_viewport
from ._trace import get_trace


def show_se() -> list[str]:
    trace = get_trace()
    e_steps, se_steps = aix.asset.latex_se(trace)
    e = aix.asset.equation(e_steps)
    se = aix.asset.equation(se_steps)
    return [e, se]


def show_loss() -> Figure:
    trace = get_trace()
    loss_steps = aix.asset.latex_loss(trace)
    loss_title = aix.asset.equation(loss_steps[:2])
    fig, ax = aix.asset.figure()
    aix.asset.frame_graph(ax, loss_viewport(), title=f"${loss_title}$", show_labels=True)
    aix.asset.plot_fn(ax, loss_fn())
    aix.asset.plot_point(ax, trace.w, trace.loss, show_values=True)
    return fig


def show_hill() -> Figure:
    trace = get_trace()
    fig, ax = aix.asset.figure()
    aix.asset.frame_graph(ax, loss_viewport(), title="Loss")
    aix.asset.plot_hill(ax, loss_fn())
    aix.asset.plot_point(ax, trace.w, trace.loss)
    return fig


def show_dark() -> Figure:
    trace = get_trace()
    fig, ax = aix.asset.figure()
    aix.asset.frame_dark(ax, loss_viewport(), title="Loss")
    aix.asset.plot_point(ax, trace.w, trace.loss)
    return fig
