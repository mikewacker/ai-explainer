from collections.abc import Callable

import torch
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.text import Text
from torch import Tensor

import ai_explainer.number as aix
from ai_explainer.number.asset import Viewport, pd
from ai_explainer.number.world import NumberSnapshot, NumberTrace


class NumberVisualSnapshot:
    """Visual snapshot for the number example that can be animated."""

    _loss_title: Text
    _param_dot: PathCollection

    def __init__(self, ax: Axes, loss_viewport: Viewport, snapshot: NumberSnapshot) -> None:
        """Create and plot a visual snapshot."""
        self._loss_title = self._frame(ax, loss_viewport)
        self._set_loss(snapshot.loss)
        self._param_dot = aix.asset.plot_point(ax, snapshot.params, snapshot.loss)

    def animate(self, snapshot: NumberSnapshot) -> list[Artist]:
        """Animate this snapshot."""
        self._set_loss(snapshot.loss)
        self._param_dot.set_offsets(torch.stack([snapshot.params, snapshot.loss]))
        return [self._loss_title, self._param_dot]

    @staticmethod
    def _frame(ax: Axes, loss_viewport: Viewport) -> Text:
        aix.asset.frame_dark(ax, loss_viewport, title="Loss")
        ax.title.set_fontfamily("monospace")
        return ax.title

    def _set_loss(self, loss: Tensor) -> None:
        self._loss_title.set_text(f"Loss: {loss:.2f}")


def plot_step(
    ax: Axes,
    loss_viewport: Viewport,
    loss_fn: Callable[[Tensor], Tensor],
    w: Tensor,
    w_next: Tensor,
) -> None:
    """Plot a single step."""
    aix.asset.frame_graph(ax, loss_viewport, title="Loss")
    aix.asset.plot_fn(ax, loss_fn)
    ax.scatter(w.item(), loss_fn(w).item(), s=32, color="#D99191", zorder=11)
    aix.asset.plot_point(ax, w_next, loss_fn(w_next))


def latex_sgd_step(trace: NumberTrace, lr: float) -> tuple[list[str], list[str], list[str]]:
    """Get the steps to compute a single step using gradient descent."""
    w_delta = lr * trace.w_grad
    w_next = trace.w - w_delta
    grad_steps = aix.asset.latex_grad(trace)
    delta_steps = [
        f"{lr:g} {pd('L', 'w')}",
        rf"{lr:g} \cdot {trace.w_grad:g}",
        f"{w_delta:g}",
    ]
    update_steps = [
        rf"w - {lr:g} {pd('L', 'w')}",
        f"{trace.w:g} - {w_delta:g}",
        f"{w_next:g}",
    ]
    return grad_steps, delta_steps, update_steps
