import torch
from matplotlib.figure import Figure
from torch import Tensor, nn

import ai_explainer.iris as aix
from ai_explainer.iris.asset.multiclass import Diagram, Viewport, drawio

from ._trace import get_trace_sample


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


def show_ce() -> Figure:
    trace = get_trace_sample()
    j = int(trace.y.argmax().item()) + 1
    viewport = Viewport(x_var=rf"\hat{{y}}_{j}", xlim=(0, 1), y_var=r"L", ylim=(0, 3))
    fig, ax = aix.asset.multiclass.figure(figsize=(2.5, 5))
    aix.asset.multiclass.frame_graph(ax, viewport, title="Cross-Entropy", show_labels=True)
    aix.asset.multiclass.plot_fn(ax, _ce_target)
    aix.asset.multiclass.plot_point(
        ax, trace.y_hat[j - 1], trace.loss, show_values=True, decimals=2
    )
    return fig


def show_loss() -> Diagram:
    return Diagram(drawio("loss_data"))


def _ce_target(y_hat: Tensor) -> Tensor:
    y = torch.ones_like(y_hat)
    return nn.functional.binary_cross_entropy(y_hat, y, reduction="none")
