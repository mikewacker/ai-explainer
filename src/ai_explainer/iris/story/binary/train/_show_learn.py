from __future__ import annotations

from dataclasses import dataclass

import torch
from matplotlib.artist import Artist
from matplotlib.figure import Figure

import ai_explainer.iris as aix
from ai_explainer.iris.asset.binary import Animation, IrisVisualSnapshot


def show_sgd() -> Animation:
    fig, vsnapshot = _show_sgd_snapshot(0)
    animator = _IrisSgdAnimator(vsnapshot)
    return Animation(fig, animator, init_frames=20, active_frames=100, fps=30)


def show_params(*, init: bool = False) -> Figure:
    step = 0 if init else -1
    fig, _ = _show_sgd_snapshot(step, show_values=True)
    return fig


def show_predictions() -> list[str]:
    canon = aix.canon.get()
    x, _ = canon.sample
    with torch.no_grad():
        y_hat = torch.sigmoid(canon.model_trained(x))
    return [f"{y_hat:.2f}", f"{canon.accuracy_train:.2%}", f"{canon.accuracy_test:.2%}"]


def _show_sgd_snapshot(
    step: int, *, show_values: bool = False
) -> tuple[Figure, IrisVisualSnapshot]:
    canon = aix.canon.get()
    snapshot = canon.history[step]
    fig = aix.asset.binary.figure_empty(figsize=(5, 7))
    vsnapshot = IrisVisualSnapshot(fig, (-3, 3), snapshot, show_values=show_values)
    return fig, vsnapshot


@dataclass(frozen=True)
class _IrisSgdAnimator:
    vsnapshot: IrisVisualSnapshot

    def animate(self, frame: int) -> list[Artist]:
        canon = aix.canon.get()
        step = 30 * frame
        snapshot = canon.history[step]
        return self.vsnapshot.animate(snapshot)
