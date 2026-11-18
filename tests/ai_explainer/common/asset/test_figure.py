from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import testing.example as aix
from testing.example.asset import Outputs, save


def test_figure() -> None:
    fig, ax = aix.asset.figure()

    assert tuple(fig.get_size_inches()) == (5, 5)
    assert fig.get_constrained_layout()
    assert len(fig.axes) == 1
    assert isinstance(ax, Axes)
    assert not isinstance(ax, Axes3D)


def test_figure__3d() -> None:
    fig, ax = aix.asset.figure(is_3d=True)

    assert tuple(fig.get_size_inches()) == (5, 5)
    assert fig.get_constrained_layout()
    assert len(fig.axes) == 1
    assert isinstance(ax, Axes3D)


def test_figure_empty() -> None:
    fig = aix.asset.figure_empty()

    assert tuple(fig.get_size_inches()) == (5, 5)
    assert fig.get_constrained_layout()
    assert len(fig.axes) == 0


def test_save(outputs: Outputs) -> None:
    fig = aix.asset.figure_empty(figsize=(0.5, 1))
    save(fig, outputs.png("figure"))

    with Image.open(outputs.png("figure")) as img:
        assert img.format == "PNG"
        assert img.size == (150, 300)
        assert img.mode == "P"
