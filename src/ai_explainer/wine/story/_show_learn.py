from __future__ import annotations

from dataclasses import dataclass

from matplotlib.artist import Artist
from matplotlib.figure import Figure

import ai_explainer.wine as aix
from ai_explainer.wine.asset import Animation, Table, WineVisualSnapshot

from ._loss_fn import loss_mesh
from ._trace import get_trace_step


def show_sgd_step() -> list[str]:
    canon = aix.canon.get()
    trace = get_trace_step(0)
    grad_steps, delta_steps, update_steps = aix.asset.latex_sgd_step(trace, canon.hparams.lr)
    grad = aix.asset.equation(grad_steps, approx_steps=1, multiline=True)
    delta = aix.asset.equation(delta_steps, approx_steps=2)
    update = aix.asset.equation(update_steps, approx_steps=2)
    return [grad, delta, update]


def show_sgd() -> Animation:
    fig, vsnapshot = _show_sgd_snapshot(0)
    animator = _WineSgdAnimator(vsnapshot)
    return Animation(fig, animator, init_frames=20, active_frames=100, fps=30)


def show_params(*, alt: bool = False) -> Figure:
    fig, _ = _show_sgd_snapshot(-1, show_values=True, alt=alt)
    return fig


def show_predictions() -> Table:
    canon = aix.canon.get()
    return aix.asset.table_predictions(canon.data, canon.model_untrained, canon.model_trained)


def show_predictions_alt() -> Table:
    canon = aix.canon.get()
    canon_alt = aix.canon.get(alt=True)
    return aix.asset.table_predictions(canon_alt.data, canon.model_trained, canon_alt.model_trained)


def _show_sgd_snapshot(
    step: int, show_values: bool = False, alt: bool = False
) -> tuple[Figure, WineVisualSnapshot]:
    canon = aix.canon.get(alt=alt)
    snapshot = canon.history[step]
    figsize = (3.5, 5) if alt else (3, 5)
    fig, ax = aix.asset.figure(figsize=figsize)
    vsnapshot = WineVisualSnapshot(ax, loss_mesh(alt=alt), snapshot, show_values=show_values)
    return fig, vsnapshot


@dataclass(frozen=True)
class _WineSgdAnimator:
    vsnapshot: WineVisualSnapshot

    def animate(self, frame: int) -> list[Artist]:
        canon = aix.canon.get()
        snapshot = canon.history[frame]
        return self.vsnapshot.animate(snapshot)
