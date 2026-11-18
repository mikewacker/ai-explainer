from pathlib import Path
from typing import Literal, overload

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from ._png import quantize_png


@overload
def figure(
    *, figsize: tuple[float, float] = (5, 5), is_3d: Literal[False] = False
) -> tuple[Figure, Axes]: ...


@overload
def figure(
    *, figsize: tuple[float, float] = (5, 5), is_3d: Literal[True]
) -> tuple[Figure, Axes3D]: ...


def figure(*, figsize: tuple[float, float] = (5, 5), is_3d: bool = False) -> tuple[Figure, Axes]:
    """Create a figure with a default size of 5x5 and a constrained layout."""
    subplot_kw = {"projection": "3d", "computed_zorder": False} if is_3d else None
    return plt.subplots(figsize=figsize, layout="constrained", subplot_kw=subplot_kw)


def figure_empty(*, figsize: tuple[float, float] = (5, 5)) -> Figure:
    """Create an empty figure with a default size of 5x5 and a constrained layout."""
    return plt.figure(figsize=figsize, layout="constrained")


def save_figure(fig: Figure, path: Path) -> None:
    """Save the figure as a PNG file."""
    fig.savefig(path, dpi=300, format="png")
    quantize_png(path)
