import pytest

import testing
import testing.example as aix
from testing.example.asset import Outputs, Viewport, Viewport3D


@pytest.fixture
def viewport() -> Viewport:
    return Viewport(x_var="x", xlim=(-4, 4), y_var="y", ylim=(-1, 7))


class TestViewport:
    def test_apply(self) -> None:
        viewport = Viewport(x_var="x", xlim=(0, 1), y_var="y", ylim=(0, 3))
        _, ax = aix.asset.figure()
        viewport.apply(ax)

        assert ax.get_xlabel() == "x"
        assert not ax.xaxis.label.get_visible()
        assert ax.get_xlim() == (0, 1)
        assert ax.get_ylabel() == "y"
        assert not ax.yaxis.label.get_visible()
        assert ax.get_ylim() == (0, 3)
        assert ax.get_aspect() == 1

    def test_apply__no_vars(self) -> None:
        viewport = Viewport(xlim=(0, 1), ylim=(0, 3))
        _, ax = aix.asset.figure()
        viewport.apply(ax)

        assert ax.get_xlabel() == ""
        assert ax.xaxis.label.get_visible()
        assert ax.get_xlim() == (0, 1)
        assert ax.get_ylabel() == ""
        assert ax.yaxis.label.get_visible()
        assert ax.get_ylim() == (0, 3)

    def test_apply__scale(self) -> None:
        viewport = Viewport(xlim=(0, 1), ylim=(0, 3), scale=(1, 3))
        _, ax = aix.asset.figure()
        viewport.apply(ax)

        assert ax.get_aspect() == 1 / 3

    def test_apply__auto_scale(self) -> None:
        viewport = Viewport(xlim=(0, 1), ylim=(0, 3), scale="auto")
        _, ax = aix.asset.figure()
        viewport.apply(ax)

        assert ax.get_aspect() == "auto"


class TestViewport3D:
    def test_apply(self) -> None:
        viewport = Viewport3D(
            x_var="x", xlim=(-1, 1), y_var="y", ylim=(-2, 2), z_var="z", zlim=(-3, 3)
        )
        _, ax = aix.asset.figure(is_3d=True)
        viewport.apply(ax)

        assert ax.get_xlabel() == "x"
        assert not ax.xaxis.label.get_visible()
        assert ax.get_xlim() == (-1, 1)
        assert ax.get_ylabel() == "y"
        assert not ax.yaxis.label.get_visible()
        assert ax.get_ylim() == (-2, 2)
        assert ax.get_zlabel() == "z"
        assert not ax.zaxis.label.get_visible()
        assert ax.get_zlim() == (-3, 3)
        box_aspect = ax.get_box_aspect()
        assert box_aspect[1] / box_aspect[0] == 2  # type: ignore  # Axes3D
        assert box_aspect[2] / box_aspect[0] == 3  # type: ignore  # Axes3D

    def test_apply__no_vars(self) -> None:
        viewport = Viewport3D(xlim=(-1, 1), ylim=(-2, 2), zlim=(-3, 3))
        _, ax = aix.asset.figure(is_3d=True)
        viewport.apply(ax)

        assert ax.get_xlabel() == ""
        assert ax.xaxis.label.get_visible()
        assert ax.get_xlim() == (-1, 1)
        assert ax.get_ylabel() == ""
        assert ax.yaxis.label.get_visible()
        assert ax.get_ylim() == (-2, 2)
        assert ax.get_zlabel() == ""
        assert ax.zaxis.label.get_visible()
        assert ax.get_zlim() == (-3, 3)

    def test_apply__scale(self) -> None:
        viewport = Viewport3D(xlim=(-1, 1), ylim=(-2, 2), zlim=(-3, 3), scale=(1, 2, 3))
        _, ax = aix.asset.figure(is_3d=True)
        viewport.apply(ax)

        box_aspect = ax.get_box_aspect()
        assert box_aspect[1] / box_aspect[0] == 1  # type: ignore  # Axes3D
        assert box_aspect[2] / box_aspect[0] == 1  # type: ignore  # Axes3D

    def test_2d_viewports(self) -> None:
        viewport = Viewport3D(
            x_var="x",
            xlim=(-1, 1),
            y_var="y",
            ylim=(-2, 2),
            z_var="z",
            zlim=(-3, 3),
            scale=(1, 2, 3),
        )
        xy = viewport.xy()
        xz = viewport.xz()
        yz = viewport.yz()

        assert xy.x_var == "x"
        assert xy.xlim == (-1, 1)
        assert xy.y_var == "y"
        assert xy.ylim == (-2, 2)
        assert xy.scale == (1, 2)
        assert xz.x_var == "x"
        assert xz.xlim == (-1, 1)
        assert xz.y_var == "z"
        assert xz.ylim == (-3, 3)
        assert xz.scale == (1, 3)
        assert yz.x_var == "y"
        assert yz.xlim == (-2, 2)
        assert yz.y_var == "z"
        assert yz.ylim == (-3, 3)
        assert yz.scale == (2, 3)


def test_frame_graph(viewport: Viewport, outputs: Outputs, goldens: Outputs) -> None:
    fig, ax = aix.asset.figure()
    aix.asset.frame_graph(ax, viewport, title="Title")

    testing.assert_matches_golden(outputs, goldens, fig, "frame_graph")


def test_frame_graph__show_labels(viewport: Viewport, outputs: Outputs, goldens: Outputs) -> None:
    fig, ax = aix.asset.figure()
    aix.asset.frame_graph(ax, viewport, title="Title", show_labels=True)

    testing.assert_matches_golden(outputs, goldens, fig, "frame_graph-show_labels")


def test_frame_empty(viewport: Viewport, outputs: Outputs, goldens: Outputs) -> None:
    fig, ax = aix.asset.figure()
    aix.asset.frame_empty(ax, viewport, title="Title")

    testing.assert_matches_golden(outputs, goldens, fig, "frame_empty")
    for spine in ax.spines.values():
        assert spine.get_zorder() == 100
