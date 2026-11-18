from matplotlib.figure import Figure

import ai_explainer.wine as aix
from ai_explainer.wine.asset import Diagram, Table, drawio

from ._loss_fn import loss_fn_w1, loss_fn_w2, loss_mesh
from ._trace import get_trace_data, get_trace_sample


def show_sample() -> Table:
    canon = aix.canon.get()
    return aix.asset.table_training_data(*canon.sample)


def show_se() -> list[str]:
    trace = get_trace_sample()
    e_steps, se_steps = aix.asset.latex_se(trace)
    e = aix.asset.equation(e_steps)
    se = aix.asset.equation(se_steps, approx_steps=1)
    return [e, se]


def show_loss_sample() -> Diagram:
    trace = get_trace_sample()
    diag = Diagram(drawio("loss_sample"))
    diag.fill_tensor("w", trace.w)
    diag.fill_tensor("x", trace.x)
    diag.fill_tensor("yhat", trace.y_hat)
    diag.fill_tensor("y", trace.y)
    diag.fill_tensor("L", trace.loss, decimals=0)
    return diag


def show_training_data(*, alt: bool = False) -> Table:
    canon = aix.canon.get(alt=alt)
    table = aix.asset.table_training_data(canon.data.x, canon.data.y)
    if alt:
        table.highlight(3, 1)
    return table


def show_loss(*, phase: int = 6) -> Diagram:
    trace = get_trace_data()
    diag = Diagram(drawio("loss_data"))
    if phase >= 1:
        diag.fill_tensor("w", trace.w)
    if phase == 1:
        diag.highlight(fill="in_color")
    if phase >= 2:
        diag.fill_tensor("x", trace.x)
        diag.fill_tensor("y", trace.y)
    if phase == 2:
        diag.highlight(fill="data_color")
    if phase >= 3:
        diag.fill_tensor("yhat", trace.y_hat)
    if phase == 3:
        diag.highlight(fill="prediction_color")
    if phase >= 4:
        diag.fill_tensor("l", trace.ell, decimals=0)
    if phase == 4:
        diag.highlight(fill="l_color")
    if phase >= 5:
        diag.fill_tensor("L", trace.loss, decimals=0)
    if phase == 5:
        diag.highlight(fill="out_color")
    return diag


def show_loss_io(*, alt: bool = False) -> Diagram:
    canon = aix.canon.get(alt=alt)
    name = "loss_alt" if alt else "loss_data"
    diag = Diagram(drawio(name))
    diag.highlight_io()
    diag.fill_tensor("x", canon.data.x)
    diag.fill_tensor("y", canon.data.y)
    if alt:
        diag.highlight(fill="alt_color")
    return diag


def show_hill() -> Figure:
    fig, ax = aix.asset.figure(is_3d=True)
    aix.asset.plot_loss_fn(ax, loss_mesh())
    return fig


def show_hill_w1() -> Figure:
    fig, ax = aix.asset.figure(is_3d=True)
    aix.asset.plot_loss_fn(ax, loss_mesh(), show_labels=["w1", "L"])
    aix.asset.plot_loss_fn_w1(ax, loss_fn_w1())
    return fig


def show_hill_w2() -> Figure:
    fig, ax = aix.asset.figure(is_3d=True)
    aix.asset.plot_loss_fn(ax, loss_mesh(), show_labels=["w2", "L"])
    aix.asset.plot_loss_fn_w2(ax, loss_fn_w2())
    return fig


def show_hill_alt() -> Figure:
    fig = aix.asset.figure_empty(figsize=(5, 3.5))
    ax_old = fig.add_subplot(1, 2, 1)
    aix.asset.plot_loss_contours(ax_old, loss_mesh())
    ax_old.set_title("Loss (Old)")
    ax_new = fig.add_subplot(1, 2, 2)
    aix.asset.plot_loss_contours(ax_new, loss_mesh(alt=True))
    ax_new.set_title("Loss (New)")
    ax_old.set_xlim(ax_new.get_xlim())
    ax_old.set_ylim(ax_new.get_ylim())
    return fig
