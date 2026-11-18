import math
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D


@dataclass(frozen=True, kw_only=True)
class Viewport:
    """Viewport for a figure.

    A scale of (1, 2) means that 1 x-unit and 2 y-units have the same length.
    """

    x_var: str = ""
    xlim: tuple[float, float]
    y_var: str = ""
    ylim: tuple[float, float]
    scale: tuple[float, float] | Literal["auto"] = (1, 1)

    def apply(self, ax: Axes) -> None:
        """Apply the viewport to the axes."""
        if self.x_var:
            ax.set_xlabel(self.x_var, visible=False)
        ax.set_xlim(self.xlim)
        if self.y_var:
            ax.set_ylabel(self.y_var, visible=False)
        ax.set_ylim(self.ylim)
        aspect = self.scale[0] / self.scale[1] if self.scale != "auto" else "auto"
        ax.set_aspect(aspect)


@dataclass(frozen=True, kw_only=True)
class Viewport3D:
    """3D viewport for a figure.

    A scale of (1, 2, 3) means that 1 x-unit, 2 y-units, and 3 z-units have the same length.
    """

    x_var: str = ""
    xlim: tuple[float, float]
    y_var: str = ""
    ylim: tuple[float, float]
    z_var: str = ""
    zlim: tuple[float, float]
    scale: tuple[float, float, float] = (1, 1, 1)

    def apply(self, ax: Axes3D) -> None:
        """Apply the viewport to the axes."""
        if self.x_var:
            ax.set_xlabel(self.x_var, visible=False)
        ax.set_xlim(self.xlim)
        if self.y_var:
            ax.set_ylabel(self.y_var, visible=False)
        ax.set_ylim(self.ylim)
        if self.z_var:
            ax.set_zlabel(self.z_var, visible=False)
        ax.set_zlim(self.zlim)
        lims = [self.xlim, self.ylim, self.zlim]
        box_aspect = tuple((lim[1] - lim[0]) / s for lim, s in zip(lims, self.scale, strict=True))
        ax.set_box_aspect(box_aspect)

    def xy(self) -> Viewport:
        """Get the 2D viewport for the x- and y-axes."""
        return Viewport(
            x_var=self.x_var,
            xlim=self.xlim,
            y_var=self.y_var,
            ylim=self.ylim,
            scale=(self.scale[0], self.scale[1]),
        )

    def xz(self) -> Viewport:
        """Get the 2D viewport for the x- and z-axes."""
        return Viewport(
            x_var=self.x_var,
            xlim=self.xlim,
            y_var=self.z_var,
            ylim=self.zlim,
            scale=(self.scale[0], self.scale[2]),
        )

    def yz(self) -> Viewport:
        """Get the 2D viewport for the y- and z-axes."""
        return Viewport(
            x_var=self.y_var,
            xlim=self.ylim,
            y_var=self.z_var,
            ylim=self.zlim,
            scale=(self.scale[1], self.scale[2]),
        )


def frame_graph(
    ax: Axes, viewport: Viewport, *, title: str = "", show_labels: bool = False
) -> None:
    """Frame the figure as a graph with a 1x1 grid and optional tick labels at the edges."""
    viewport.apply(ax)
    if title:
        pad = 16 if show_labels else 8
        ax.set_title(title, fontsize=16, pad=pad)

    # Set up the x-axis and y-axis.
    for spine in [ax.spines["left"], ax.spines["bottom"]]:
        spine.set_position("zero")
        spine.set_linewidth(0.5)

    # Set up a 1x1 grid.
    for lim, axis in zip([viewport.xlim, viewport.ylim], [ax.xaxis, ax.yaxis], strict=True):
        locs = range(math.ceil(lim[0]), math.floor(lim[1]) + 1)
        axis.set_minor_locator(plt.FixedLocator(locs))
    ax.grid(True, which="both", linewidth=0.5, alpha=0.5, clip_on=False)

    # Set up tick labels at the edges of the graph.
    if show_labels:
        x_labels = [f"${ax.get_xlabel()} = {x:g}$" for x in ax.get_xlim()]
        y_labels = [f"${ax.get_ylabel()} = {y:g}$" for y in ax.get_ylim()]
        ax.set_xticks(viewport.xlim, labels=x_labels, fontsize=10)
        ax.set_yticks(viewport.ylim, labels=y_labels, fontsize=10)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove chartjunk.
    for spine in [ax.spines["right"], ax.spines["top"]]:
        spine.set_visible(False)
    ax.tick_params(which="both", length=0)


def frame_empty(ax: Axes, viewport: Viewport, *, title: str = "") -> None:
    """Frame the figure with a border and an empty viewport."""
    viewport.apply(ax)
    if title:
        ax.set_title(title, fontsize=16, pad=8)

    # Put the frame in front.
    for spine in ax.spines.values():
        spine.set_zorder(100)

    # Remove chartjunk.
    ax.set_xticks([])
    ax.set_yticks([])
