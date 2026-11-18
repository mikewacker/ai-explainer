from collections.abc import Callable

from matplotlib.axes import Axes
from torch import Tensor

import ai_explainer.number as aix
from ai_explainer.number.asset import Viewport
from ai_explainer.number.world import NumberTrace


def frame_dark(ax: Axes, viewport: Viewport, *, title: str = "") -> None:
    """Frame the figure with a dark viewport."""
    if title:
        ax.set_title(title, fontsize=16, pad=8)
    viewport.apply(ax)

    # Set up a dark background.
    ax.set_facecolor("black")

    # Remove chartjunk.
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_hill(ax: Axes, f: Callable[[Tensor], Tensor]) -> None:
    """Plot the graph for a function as a grassy hill."""
    x, y = aix.asset.sample_fn(f, ax.get_xlim())
    ax.plot(x, y, linewidth=2, color="forestgreen", zorder=10)
    ax.fill_between(x, y, ax.get_ylim()[0], color="saddlebrown")


def latex_loss(trace: NumberTrace) -> list[str]:
    """Get the steps to compute loss."""
    return [
        "L",
        f"(w - {trace.y:g})^2",
        f"({trace.w:g} - {trace.y:g})^2",
        f"{trace.loss:g}",
    ]


def latex_se(trace: NumberTrace) -> tuple[list[str], list[str]]:
    """Get the steps to compute squared error."""
    e = trace.y_hat - trace.y
    e_steps = [
        f"{trace.y_hat:g} - {trace.y:g}",
        f"{e:g}",
    ]
    se_steps = [
        f"{e:g}^2" if e >= 0 else f"({e:g})^2",
        f"{trace.loss}",
    ]
    return e_steps, se_steps
