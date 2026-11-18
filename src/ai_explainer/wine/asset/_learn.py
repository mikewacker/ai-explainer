import torch
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.text import Text
from torch import Tensor

import ai_explainer.wine as aix
from ai_explainer.wine.asset import Table, pd
from ai_explainer.wine.world import WineData, WineModel, WineSnapshot, WineTrace


class WineVisualSnapshot:
    """Visual snapshot for the wine example that can be animated."""

    _loss_title: Text
    _params_dot: PathCollection

    def __init__(
        self, ax: Axes, loss_mesh: Tensor, snapshot: WineSnapshot, *, show_values: bool = False
    ) -> None:
        """Create and plot a visual snapshot.

        The show_values option does not support animation.
        """
        self._loss_title = self._frame(ax, loss_mesh)
        self._set_loss(snapshot.loss)
        if show_values:
            self._params_dot = aix.asset.plot_point(
                ax, *snapshot.params, show_values=True, decimals=2, ha="center", va="top"
            )
        else:
            self._params_dot = aix.asset.plot_point(ax, *snapshot.params)

    def animate(self, snapshot: WineSnapshot) -> list[Artist]:
        """Animate this snapshot."""
        self._set_loss(snapshot.loss)
        self._params_dot.set_offsets(snapshot.params)
        return [self._loss_title, self._params_dot]

    @staticmethod
    def _frame(ax: Axes, loss_mesh: Tensor) -> Text:
        aix.asset.plot_loss_contours(ax, loss_mesh)
        ax.title.set_fontfamily("monospace")
        return ax.title

    def _set_loss(self, loss: Tensor) -> None:
        self._loss_title.set_text(f"Loss: {loss:4.0f}")


def table_predictions(data: WineData, model_old: WineModel, model_new: WineModel) -> Table:
    """Get a table of predictions for both models."""
    with torch.no_grad():
        y_hat_untrained = torch.round(model_old(data.x))
        y_hat_trained = torch.round(model_new(data.x))
    predictions = Table("Proline [$y$]", data.y)
    predictions.add_group(r"Prediction (Old) [$hat(y)$]", y_hat_untrained)
    predictions.add_group(r"Prediction (New) [$hat(y)$]", y_hat_trained)
    return predictions


def latex_sgd_step(
    trace: WineTrace, lr: float
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Get the steps to compute a single step using gradient descent."""
    d_in = trace.x.shape[-1]
    w_delta = lr * trace.w_grad
    w_next = trace.w - w_delta
    grad_steps = aix.asset.latex_grad(trace)
    delta_steps = [
        [
            f"{lr:g} {pd('L', f'w_{j}')}",
            rf"{lr:g} \cdot {trace.w_grad[j - 1]:.2f}",
            f"{w_delta[j - 1]:.2f}",
        ]
        for j in range(1, d_in + 1)
    ]
    update_steps = [
        [
            f"w_{j} - {lr:g} {pd('L', f'w_{j}')}",
            f"{trace.w[j - 1]:g} - {w_delta[j - 1]:.2f}",
            f"{w_next[j - 1]:.2f}",
        ]
        for j in range(1, d_in + 1)
    ]
    return grad_steps, delta_steps, update_steps
