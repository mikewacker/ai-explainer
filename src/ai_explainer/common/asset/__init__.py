"""Tools for building hand-crafted assets."""

import matplotlib.pyplot as plt

from ._animation import Animation
from ._diagram import Diagram
from ._equation import equation, pd
from ._figure import figure, figure_empty
from ._file import Outputs
from ._frame import Viewport, Viewport3D, frame_empty, frame_graph
from ._loss_fn import LossCrossFn
from ._plot import plot_fn, plot_point, plot_slope, sample_fn
from ._save import save
from ._table import Table

__all__ = [
    "Animation",
    "Diagram",
    "LossCrossFn",
    "Outputs",
    "Table",
    "Viewport",
    "Viewport3D",
    "equation",
    "figure",
    "figure_empty",
    "frame_empty",
    "frame_graph",
    "pd",
    "plot_fn",
    "plot_point",
    "plot_slope",
    "sample_fn",
    "save",
]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["STIX", *plt.rcParams["font.serif"]]
plt.rcParams["mathtext.fontset"] = "stix"
