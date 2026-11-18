from typing import cast

import torch
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.text import Text
from torch import Tensor

import ai_explainer.iris as aix
from ai_explainer.iris.asset.binary import Viewport
from ai_explainer.iris.world import IrisSnapshot


class IrisVisualSnapshot:
    """Visual snapshot for the iris example that can be animated."""

    _loss_title: Text
    _hidden_bars: list[BarContainer]
    _output_bars: list[BarContainer]

    def __init__(
        self,
        fig: Figure,
        param_lim: tuple[float, float],
        snapshot: IrisSnapshot,
        *,
        show_values: bool = False,
    ) -> None:
        """Create and plot a visual snapshot.

        The show_values option does not support animation.
        """
        self._loss_title, hidden_axs, output_axs = self._frame(fig, param_lim, snapshot)
        self._set_loss(snapshot.loss)
        self._hidden_bars = self._plot_layer_params(
            hidden_axs, snapshot.params.w1, snapshot.params.b1, show_values=show_values
        )
        self._output_bars = self._plot_layer_params(
            output_axs, snapshot.params.w2, snapshot.params.b2, show_values=show_values
        )

    def animate(self, snapshot: IrisSnapshot) -> list[Artist]:
        """Animate this snapshot."""
        return [
            self._set_loss(snapshot.loss),
            *self._animate_layer_params(self._hidden_bars, snapshot.params.w1, snapshot.params.b1),
            *self._animate_layer_params(self._output_bars, snapshot.params.w2, snapshot.params.b2),
        ]

    @staticmethod
    def _frame(
        fig: Figure, param_lim: tuple[float, float], snapshot: IrisSnapshot
    ) -> tuple[Text, list[Axes], list[Axes]]:
        loss_title = fig.suptitle("Loss", family="monospace", fontsize=16)
        in_size = snapshot.params.w1.shape[1]
        hidden_size = snapshot.params.w1.shape[0]
        out_size = snapshot.params.w2.shape[0] if snapshot.params.w2.ndim == 2 else 1
        grid = fig.add_gridspec(max(hidden_size, out_size), 2)
        hidden_axs = IrisVisualSnapshot._frame_layer_params(
            grid, param_lim, 1, in_size, hidden_size
        )
        output_axs = IrisVisualSnapshot._frame_layer_params(
            grid, param_lim, 2, hidden_size, out_size
        )
        return loss_title, hidden_axs, output_axs

    @staticmethod
    def _frame_layer_params(
        grid: GridSpec, param_lim: tuple[float, float], layer: int, d_in: int, d_out: int
    ) -> list[Axes]:
        layer_axs = IrisVisualSnapshot._axs_layer(grid, layer, d_out)
        viewport = Viewport(xlim=(-0.5, d_in + 0.5), ylim=param_lim, scale="auto")
        for k, neuron_ax in enumerate(layer_axs, start=1):
            aix.asset.binary.frame_empty(neuron_ax, viewport)

            # Add the labels for the parameters.
            if d_out > 1:
                w_labels = [f"$w^{{({layer})}}_{{{k}{j}}}$" for j in range(1, d_in + 1)]
                b_label = f"$b^{{({layer})}}_{k}$"
            else:
                w_labels = [f"$w^{{({layer})}}_{j}$" for j in range(1, d_in + 1)]
                b_label = f"$b^{{({layer})}}_1$"  # subscript is needed for vertical alignment
            neuron_ax.set_xticks(range(d_in + 1), [*w_labels, b_label], fontsize=10)
            if d_out == 1:
                IrisVisualSnapshot._remove_b_subscript(neuron_ax)

            # Add a zero line for the parameter values.
            neuron_ax.axhline(y=0, linewidth=0.5, color="silver", zorder=11)

            # Remove chartjunk.
            neuron_ax.set_yticks([])
            neuron_ax.tick_params(which="both", length=0)
        return layer_axs

    @staticmethod
    def _axs_layer(grid: GridSpec, layer: int, d_out: int) -> list[Axes]:
        start = (grid.nrows - d_out) // 2
        stop = start + d_out
        fig = cast(Figure, grid.figure)
        return [fig.add_subplot(grid[row, layer - 1]) for row in range(start, stop)]

    @staticmethod
    def _remove_b_subscript(neuron_ax: Axes) -> None:
        b_label = neuron_ax.get_xticklabels()[-1]
        b_label.set_alpha(0)  # hide the label, but keep its bounding box
        b_text = b_label.get_text().replace("_1", "")
        neuron_ax.annotate(
            b_text,
            xycoords=b_label,
            xy=(0.5, 0.5),
            ha="center",
            va="center",
            fontproperties=b_label.get_fontproperties(),
        )

    def _set_loss(self, loss: Tensor) -> Text:
        self._loss_title.set_text(f"Loss: {loss:.2f}")
        return self._loss_title

    @staticmethod
    def _plot_layer_params(
        layer_axs: list[Axes], w: Tensor, b: Tensor, *, show_values: bool
    ) -> list[BarContainer]:
        layer_params = IrisVisualSnapshot._stack_linear_params(w, b)
        layer_bars = []
        for neuron_ax, neuron_params in zip(layer_axs, layer_params, strict=True):
            idxs = range(neuron_params.shape[0])
            param_min = neuron_ax.get_ylim()[0]
            neuron_heights = [param.item() - param_min for param in neuron_params]
            neuron_bars = neuron_ax.bar(
                idxs, neuron_heights, bottom=param_min, width=1, color="#8A7F80", zorder=10
            )
            layer_bars.append(neuron_bars)
            if not show_values:
                continue

            neuron_values = [f"{param:.2f}" for param in neuron_params]
            neuron_ax.bar_label(neuron_bars, neuron_values, fontsize=10, zorder=10)
        return layer_bars

    @staticmethod
    def _animate_layer_params(layer_bars: list[BarContainer], w: Tensor, b: Tensor) -> list[Artist]:
        layer_params = IrisVisualSnapshot._stack_linear_params(w, b)
        layer_artists = []
        for neuron_bars, neuron_params in zip(layer_bars, layer_params, strict=True):
            for param_bar, param in zip(neuron_bars, neuron_params, strict=True):
                param_height = param.item() - param_bar.get_y()
                param_bar.set_height(param_height)
                layer_artists.append(param_bar)
        return layer_artists

    @staticmethod
    def _stack_linear_params(w: Tensor, b: Tensor) -> Tensor:
        params = torch.hstack([w, b.unsqueeze(-1)])
        return torch.atleast_2d(params)
