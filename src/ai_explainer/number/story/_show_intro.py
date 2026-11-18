import torch
from matplotlib.figure import Figure
from torch import Tensor

import ai_explainer.number as aix
from ai_explainer.number.asset import Viewport


def show_function() -> Figure:
    x = torch.tensor(3.0)
    fig, ax = aix.asset.figure()
    aix.asset.frame_graph(ax, _fn_viewport(), title="$y = (x - 1)^2$", show_labels=True)
    aix.asset.plot_fn(ax, _fn)
    aix.asset.plot_point(ax, x, _fn(x), show_values=True)
    return fig


def _fn(x: Tensor) -> Tensor:
    return (x - 1) ** 2


def _fn_viewport() -> Viewport:
    return Viewport(x_var="x", xlim=(-3, 5), y_var="y", ylim=(-1, 7))
