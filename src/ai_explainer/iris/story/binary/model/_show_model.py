from typing import Literal

import torch
from matplotlib.figure import Figure

import ai_explainer.iris as aix
from ai_explainer.iris.asset.binary import Diagram, Table, Viewport, drawio

from ._trace import get_trace_sample


def show_nn(*, highlight: Literal["in", "hidden", "out"] | None = None) -> Diagram:
    diag = Diagram(drawio("model_nn"))
    if highlight is not None:
        diag.highlight(fill=f"{highlight}_fill", stroke=f"{highlight}_stroke")
    return diag


def show_sample() -> Table:
    canon = aix.canon.get()
    x, _ = canon.sample
    columns = [
        r"Sepal Length \[$x_1$\]",
        r"Sepal Width \[$x_2$\]",
        r"Petal Length \[$x_3$\]",
        r"Petal Width \[$x_4$\]",
    ]
    return Table(columns, x)


def show_input() -> Diagram:
    trace = get_trace_sample()
    diag = Diagram(drawio("model_input"))
    diag.highlight_io()
    diag.fill_tensor("x", trace.x, decimals=1)
    return diag


def show_hidden(*, phase: int = 4) -> Diagram:
    trace = get_trace_sample()
    diag = Diagram(drawio("model_hidden-no-layer"))
    if phase == 0:
        diag.highlight_io()
    if phase >= 1:
        diag.fill_tensor("w", trace.w1, decimals=2)
        diag.fill_tensor("b", trace.b1, decimals=2)
    if phase == 1:
        diag.highlight(fill="param_color")
    if phase >= 2:
        diag.fill_tensor("x", trace.x)
        diag.fill_tensor("z", trace.z1, decimals=2)
    if phase == 2:
        diag.highlight(fill="in_color")
        diag.highlight(fill="z_color")
    if phase >= 3:
        diag.fill_tensor("a", trace.a, decimals=2)
    if phase == 3:
        diag.highlight(fill="out_color")
    return diag


def show_nn_hidden1() -> Diagram:
    diag = Diagram(drawio("model_nn-hidden1"))
    diag.highlight(fill="hidden_fill", stroke="hidden_stroke")
    return diag


def show_hidden1_scalar() -> Diagram:
    diag = Diagram(drawio("model_hidden1-scalar"))
    diag.highlight_io()
    return diag


def show_hidden1(*, phase: int = 4) -> Diagram:
    trace = get_trace_sample()
    diag = Diagram(drawio("model_hidden1"))
    if phase == 0:
        diag.highlight_io()
    if phase >= 1:
        diag.fill_tensor("w", trace.w1, decimals=2)
        diag.fill_tensor("b", trace.b1, decimals=2)
    if phase == 1:
        diag.highlight(fill="param_color")
    if phase >= 2:
        diag.fill_tensor("x", trace.x)
        diag.fill_tensor("z", trace.z1, decimals=2)
    if phase == 2:
        diag.highlight(fill="in_color")
        diag.highlight(fill="z_color")
    if phase >= 3:
        diag.fill_tensor("a", trace.a, decimals=2)
    if phase == 3:
        diag.highlight(fill="out_color")
    return diag


def show_relu() -> Figure:
    trace = get_trace_sample()
    viewport = Viewport(x_var="z", xlim=(-4, 4), y_var="a", ylim=(-4, 4))
    fig, ax = aix.asset.binary.figure()
    aix.asset.binary.frame_graph(ax, viewport, title="ReLU", show_labels=True)
    aix.asset.binary.plot_fn(ax, torch.relu)
    aix.asset.binary.plot_point(
        ax, trace.z1[0], trace.a[0], show_values=True, decimals=2, ha="right"
    )
    return fig


def show_linear1() -> list[str]:
    trace = get_trace_sample()
    params_table = [
        (torch.tensor([0.0, 0.0, 0.0, 0.0]), torch.tensor(-4.0)),
        (torch.tensor([2.0, 0.0, 0.0, 0.0]), torch.tensor(-4.0)),
        (torch.tensor([2.0, -3.0, 0.0, 0.0]), torch.tensor(-4.0)),
    ]
    linear1_table = [aix.asset.binary.latex_linear1(w, b, trace.x) for w, b in params_table]
    return [
        aix.asset.binary.equation(steps, multiline=True, align_terms=True)
        for steps in linear1_table
    ]


def show_output(*, phase: int = 4) -> Diagram:
    trace = get_trace_sample()
    diag = Diagram(drawio("model_output"))
    if phase == 0:
        diag.highlight_io()
    if phase >= 1:
        diag.fill_tensor("w", trace.w2, decimals=2)
        diag.fill_tensor("b", trace.b2, decimals=2)
    if phase == 1:
        diag.highlight(fill="param_color")
    if phase >= 2:
        diag.fill_tensor("a", trace.a, decimals=2)
        diag.fill_tensor("z", trace.z2, decimals=2)
    if phase == 2:
        diag.highlight(fill="in_color")
        diag.highlight(fill="z_color")
    if phase >= 3:
        diag.fill_tensor("yhat", trace.y_hat, decimals=2)
    if phase == 3:
        diag.highlight(fill="out_color")
    return diag


def show_sigmoid() -> Figure:
    trace = get_trace_sample()
    viewport = Viewport(x_var="z^{(2)}", xlim=(-4, 4), y_var=r"\hat{y}", ylim=(0, 1))
    fig, ax = aix.asset.binary.figure(figsize=(5, 1.25))
    aix.asset.binary.frame_graph(ax, viewport, title="Sigmoid", show_labels=True)
    aix.asset.binary.plot_fn(ax, torch.sigmoid)
    aix.asset.binary.plot_point(
        ax, trace.z2, trace.y_hat, show_values=True, decimals=2, ha="right", va="bottom"
    )
    return fig
