from pathlib import Path

from matplotlib.figure import Figure

from ._equation import save_equations
from ._figure import save_figure


def save(asset: Figure | list[str], path: Path) -> None:
    """Save the asset."""
    if isinstance(asset, Figure):
        save_figure(asset, path)
    else:
        save_equations(asset, path)
