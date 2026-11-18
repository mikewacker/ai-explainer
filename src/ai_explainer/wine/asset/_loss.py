import math
from typing import Literal

import torch
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from torch import Tensor

import ai_explainer.wine as aix
from ai_explainer.wine.asset import LossCrossFn, Table, Viewport3D
from ai_explainer.wine.world import WineTrace


def plot_loss_fn(
    ax: Axes3D, loss_mesh: Tensor, *, show_labels: list[Literal["w1", "w2", "L"]] | None = None
) -> None:
    """Plot the graph for the loss function."""
    show_labels = show_labels if show_labels is not None else ["w1", "w2", "L"]
    viewport = _get_viewport(loss_mesh)
    _frame_loss_fn(ax, viewport, show_labels=show_labels)
    ax.plot_surface(*loss_mesh.unbind(dim=-1), cmap="Blues", alpha=0.5, zorder=11)
    ax.scatter(*loss_mesh[0, 0], s=0.5, c="silver", zorder=12)
    for contour in loss_mesh[10::10]:
        ax.plot(*contour.unbind(dim=-1), linewidth=0.5, color="silver", zorder=12)


def plot_loss_contours(ax: Axes, loss_mesh: Tensor) -> None:
    """Plot the contours for the loss function from a top-down perspective."""
    viewport = _get_viewport(loss_mesh)
    loss_mesh = loss_mesh[..., :2]
    aix.asset.frame_empty(ax, viewport.xy(), title="Loss")
    ax.scatter(*loss_mesh[0, 0], s=0.5, c="silver", zorder=10)
    for contour in loss_mesh[10::10]:
        ax.plot(*contour.unbind(dim=-1), linewidth=0.5, color="silver", zorder=10)


def plot_loss_fn_w1(ax: Axes3D, loss_fn_w1: LossCrossFn[Tensor]) -> None:
    """Plot the graph for a 2D cross-section of the loss function for w_1."""
    w2 = loss_fn_w1.params[1]
    _frame_loss_fn_w1(ax, w2)
    w1, loss = aix.asset.sample_fn(loss_fn_w1, ax.get_xlim())
    w1, loss = _clip_2d(ax, w1, loss)
    ax.plot(w1, w2, loss, color="black", linewidth=1, zorder=13)


def plot_loss_fn_w2(ax: Axes3D, loss_fn_w2: LossCrossFn[Tensor]) -> None:
    """Plot the graph for a 2D cross-section of the loss function for w_2."""
    w1 = loss_fn_w2.params[0]
    _frame_loss_fn_w2(ax, w1)
    w2, loss = aix.asset.sample_fn(loss_fn_w2, ax.get_ylim())
    w2, loss = _clip_2d(ax, w2, loss)
    ax.plot(w1, w2, loss, color="black", linewidth=1, zorder=13)


def table_training_data(x: Tensor, y: Tensor) -> Table:
    """Get a table for the training data."""
    table = Table([r"Malic Acid \[$x_1$\]", r"Proanthocyanidins \[$x_2$\]"], x)
    table.add_group(r"Proline \[$y$\]", y)
    return table


def latex_se(trace: WineTrace) -> tuple[list[str], list[str]]:
    """Get the steps to compute squared error."""
    e = trace.y_hat - trace.y
    e_steps = [
        r"\hat{y} - y",
        f"{trace.y_hat:g} - {trace.y:g}",
        f"{e:g}",
    ]
    se_steps = [
        r"(\hat{y} - y)^2",
        f"{e:g}^2" if e >= 0 else f"({e:g})^2",
        f"{trace.loss:.0f}",
    ]
    return e_steps, se_steps


def _get_viewport(loss_mesh: Tensor) -> Viewport3D:
    bottom_contour, top_contour = loss_mesh[0], loss_mesh[-1]
    w1_min = math.floor(torch.min(top_contour[:, 0]).item()) - 1
    w1_max = math.ceil(torch.max(top_contour[:, 0]).item()) + 1
    w2_min = math.floor(torch.min(top_contour[:, 1]).item()) - 1
    w2_max = math.ceil(torch.max(top_contour[:, 1]).item()) + 1
    loss_min = bottom_contour[0, 2].item()
    loss_max = top_contour[0, 2].item()
    loss_scale = (loss_max - loss_min) / max(w1_max - w1_min, w2_max - w2_min)
    return Viewport3D(
        x_var="w_1",
        xlim=(w1_min, w1_max),
        y_var="w_2",
        ylim=(w2_min, w2_max),
        z_var="L",
        zlim=(loss_min, loss_max),
        scale=(1, 1, loss_scale),
    )


def _frame_loss_fn(
    ax: Axes3D, viewport: Viewport3D, *, show_labels: list[Literal["w1", "w2", "L"]]
) -> None:
    ax.set_title("Loss", fontsize=16, pad=0)
    viewport.apply(ax)
    ax.set_box_aspect(ax.get_box_aspect(), zoom=0.96)

    # Set the orientation.
    ax.view_init(elev=30, azim=-120)

    # Draw the back of the box wireframe.
    w1_min, w1_max = ax.get_xlim()
    w2_min, w2_max = ax.get_ylim()
    loss_min, loss_max = ax.get_zlim()
    for w1 in ax.get_xlim():
        _frame_line(ax, w1, (w2_min, w2_max), loss_min, zorder=10)
    for w2 in ax.get_ylim():
        _frame_line(ax, (w1_min, w1_max), w2, loss_min, zorder=10)
    for w1, w2 in [(w1_min, w2_max), (w1_max, w2_min), (w1_max, w2_max)]:
        _frame_line(ax, w1, w2, (loss_min, loss_max), zorder=10)
    _frame_line(ax, (w1_min, w1_max), w2_max, loss_max, zorder=10)
    _frame_line(ax, w1_max, (w2_min, w2_max), loss_max, zorder=10)

    # Draw the front of the box wireframe.
    _frame_line(ax, (w1_min, w1_max), w2_min, loss_max, zorder=13)
    _frame_line(ax, w1_min, (w2_min, w2_max), loss_max, zorder=13)
    _frame_line(ax, w1_min, w2_min, (loss_min, loss_max), zorder=13)

    # Label the edges.
    if "w1" in show_labels:
        for w1 in ax.get_xlim():
            _label_w1(ax, w1)
    if "w2" in show_labels:
        for w2 in ax.get_ylim():
            _label_w2(ax, w2)
    if "L" in show_labels:
        for loss in ax.get_zlim():
            _label_loss(ax, loss)

    # Remove chartjunk.
    ax.axis("off")


def _frame_loss_fn_w1(ax: Axes3D, w2: Tensor) -> None:
    # Draw the back of the square wireframe.
    w1_min, w1_max = ax.get_xlim()
    loss_min, loss_max = ax.get_zlim()
    _frame_line(ax, (w1_min, w1_max), w2.item(), loss_min, linewidth=1, zorder=10)
    _frame_line(ax, w1_max, w2.item(), (loss_min, loss_max), linewidth=1, zorder=10)

    # Draw the front of the square wireframe.
    _frame_line(ax, (w1_min, w1_max), w2.item(), loss_max, linewidth=1, zorder=13)
    _frame_line(ax, w1_min, w2.item(), (loss_min, loss_max), linewidth=1, zorder=13)

    # Label w2.
    _label_w2(ax, w2.item())


def _frame_loss_fn_w2(ax: Axes3D, w1: Tensor) -> None:
    # Draw the back of the square wireframe.
    w2_min, w2_max = ax.get_ylim()
    loss_min, loss_max = ax.get_zlim()
    _frame_line(ax, w1.item(), (w2_min, w2_max), loss_min, linewidth=1, zorder=10)
    _frame_line(ax, w1.item(), w2_max, (loss_min, loss_max), linewidth=1, zorder=10)

    # Draw the front of the square wireframe.
    _frame_line(ax, w1.item(), (w2_min, w2_max), loss_max, linewidth=1, zorder=13)
    _frame_line(ax, w1.item(), w2_min, (loss_min, loss_max), linewidth=1, zorder=13)

    # Label w1.
    _label_w1(ax, w1.item())


def _frame_line(
    ax: Axes3D,
    w1: float | tuple[float, float],
    w2: float | tuple[float, float],
    loss: float | tuple[float, float],
    *,
    linewidth: float = 0.5,
    zorder: int,
) -> None:
    ax.plot(w1, w2, loss, linewidth=linewidth, color="black", zorder=zorder)


def _label_w1(ax: Axes3D, w1: float) -> None:
    pad = _get_label_pad(ax)
    w2 = ax.get_ylim()[0] - pad
    loss = ax.get_zlim()[0]
    ax.text(w1, w2, loss, f"$w_1 = {w1:g}$", ha="left", va="top")


def _label_w2(ax: Axes3D, w2: float) -> None:
    pad = _get_label_pad(ax)
    w1 = ax.get_xlim()[0] - pad
    loss = ax.get_zlim()[0]
    ax.text(w1, w2, loss, f"$w_2 = {w2:g}$", ha="right", va="top")


def _label_loss(ax: Axes3D, loss: float) -> None:
    pad = _get_label_pad(ax)
    w1 = ax.get_xlim()[0]
    w2 = ax.get_ylim()[1] + pad
    ax.text(w1, w2, loss, rf"$L \approx {loss:.0f}$", ha="right", va="bottom")


def _get_label_pad(ax: Axes3D) -> float:
    w1_lim = ax.get_xlim()
    w2_lim = ax.get_ylim()
    w_len = max(w2_lim[1] - w2_lim[0], w1_lim[1] - w1_lim[0])
    return w_len * 0.02


def _clip_2d(ax: Axes3D, w: Tensor, loss: Tensor) -> tuple[Tensor, Tensor]:
    loss_max = ax.get_zlim()[1]
    start, stop = _get_clip_2d_range(loss, loss_max)
    w, loss = w[start:stop], loss[start:stop]
    w_min = _interpolate_2d(w[:2], loss[:2], loss_max)
    w[0], loss[0] = w_min, loss_max
    w_max = _interpolate_2d(w[-2:], loss[-2:], loss_max)
    w[-1], loss[-1] = w_max, loss_max
    return w, loss


def _get_clip_2d_range(loss: Tensor, loss_max: float) -> tuple[int, int]:
    in_idxs = torch.nonzero(loss <= loss_max).flatten()
    start = int(in_idxs[0].item()) - 1
    stop = int(in_idxs[-1].item()) + 2
    return start, stop


def _interpolate_2d(w: Tensor, loss: Tensor, loss_max: float) -> Tensor:
    m = (loss[1] - loss[0]) / (w[1] - w[0])
    return w[0] + (loss_max - loss[0]) / m
