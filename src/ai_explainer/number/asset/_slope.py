import torch
from matplotlib.axes import Axes
from torch import Tensor

import ai_explainer.number as aix
from ai_explainer.number.asset import pd
from ai_explainer.number.world import NumberTrace


def plot_tangent(ax: Axes, x0: Tensor, y0: Tensor, m: Tensor) -> None:
    """Plot the tangent line at the point."""
    denom = torch.sqrt(1 + m**2)
    x1 = x0 - 1 / denom
    y1 = y0 - m / denom
    x2 = x0 + 1 / denom
    y2 = y0 + m / denom
    aix.asset.plot_point(ax, x0, y0)
    ax.plot([x1, x2], [y1, y2], linewidth=2, linestyle="--", color="firebrick", zorder=12)


def latex_grad(trace: NumberTrace) -> list[str]:
    """Get the steps to compute the gradient."""
    return [
        pd("L", "w"),
        f"2(w - {trace.y:g})",
        f"2({trace.w:g} - {trace.y:g})",
        f"{trace.w_grad:g}",
    ]
