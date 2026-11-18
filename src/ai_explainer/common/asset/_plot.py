from collections.abc import Callable
from typing import Literal

import torch
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from torch import Tensor

from ._equation import pd


def plot_fn(ax: Axes, f: Callable[[Tensor], Tensor]) -> None:
    """Plot the graph for the function."""
    x, y = sample_fn(f, ax.get_xlim())
    ax.plot(x, y, linewidth=2, color="steelblue", zorder=10)


def plot_point(
    ax: Axes,
    x0: Tensor,
    y0: Tensor,
    *,
    show_values: bool = False,
    decimals: int | None = None,
    ha: Literal["left", "center", "right"] = "left",
    va: Literal["top", "center", "bottom"] = "center",
) -> PathCollection:
    """Plot the point."""
    point = ax.scatter(x0.item(), y0.item(), s=32, color="firebrick", zorder=12)
    if not show_values:
        return point

    x_var = ax.get_xlabel()
    y_var = ax.get_ylabel()
    text = _create_values_text(x_var, x0, y_var, y0, decimals=decimals)
    _show_values(ax, x0, y0, text, ha=ha, va=va)
    return point


def plot_slope(
    ax: Axes,
    x0: Tensor,
    y0: Tensor,
    m: Tensor,
    *,
    show_values: bool = False,
    decimals: int | None = None,
    ha: Literal["left", "center", "right"] = "left",
    va: Literal["top", "center", "bottom"] = "center",
) -> None:
    """Plot the point and an upward-facing arrow for the slope at that point."""
    if m != 0:
        denom = torch.sqrt(1 / m**2 + 1)
        x1 = x0 + (1 / m) / denom
        y1 = y0 + 1 / denom
    else:
        x1 = x0 + 1
        y1 = y0
    plot_point(ax, x0, y0)
    ax.annotate(
        "",
        xy=(x1.item(), y1.item()),
        xytext=(x0.item(), y0.item()),
        arrowprops={"arrowstyle": "->", "linewidth": 2, "color": "firebrick"},
        zorder=12,
    )
    if not show_values:
        return

    x_var = ax.get_xlabel()
    y_var = ax.get_ylabel()
    m_var = pd(y_var, x_var)
    text = _create_values_text(x_var, x0, m_var, m, decimals=decimals)
    _show_values(ax, x0, y0, text, ha=ha, va=va)


def sample_fn(
    f: Callable[[Tensor], Tensor], xlim: tuple[float, float], *, steps: int = 101
) -> tuple[Tensor, Tensor]:
    """Sample points to plot the function."""
    x = torch.linspace(*xlim, steps)
    y = f(x)
    return x, y


def _create_values_text(
    x_var: str, x0: Tensor, y_var: str, y0: Tensor, *, decimals: int | None
) -> str:
    values = zip([x_var, y_var], [x0, y0], strict=True)
    if decimals is None:
        coords = [f"{name} = {value:g}" for name, value in values]
    else:
        coords = [rf"{name} \approx {value:.{decimals}f}" for name, value in values]
    return f"$({coords[0]}, {coords[1]})$"


def _show_values(
    ax: Axes,
    x0: Tensor,
    y0: Tensor,
    text: str,
    *,
    ha: Literal["left", "center", "right"] = "left",
    va: Literal["top", "center", "bottom"] = "center",
) -> None:
    if ha == "left":
        x_offset = 8
    elif ha == "center":
        x_offset = 0
    else:
        x_offset = -8
    if va == "top":
        y_offset = -8
    elif va == "center":
        y_offset = 0
    else:
        y_offset = 8
    ax.annotate(
        text,
        xy=(x0.item(), y0.item()),
        textcoords="offset points",
        xytext=(x_offset, y_offset),
        ha=ha,
        va=va,
        fontsize=10,
        zorder=12,
    )
