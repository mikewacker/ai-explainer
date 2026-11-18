from __future__ import annotations

from dataclasses import dataclass

from matplotlib.artist import Artist
from matplotlib.figure import Figure

import ai_explainer.number as aix
from ai_explainer.number.asset import Animation, NumberVisualSnapshot

from ._loss_fn import loss_fn, loss_viewport
from ._trace import get_trace_step


def show_step_large() -> Figure:
    trace = get_trace_step(0)
    w_next = trace.w - trace.w_grad
    fig, ax = aix.asset.figure()
    aix.asset.plot_step(ax, loss_viewport(), loss_fn(), trace.w, w_next)
    return fig


def show_step_uphill() -> Figure:
    canon = aix.canon.get()
    trace = get_trace_step(0)
    w_next = trace.w + canon.hparams.lr * trace.w_grad
    fig, ax = aix.asset.figure()
    aix.asset.plot_step(ax, loss_viewport(), loss_fn(), trace.w, w_next)
    return fig


def show_step_downhill() -> Figure:
    canon = aix.canon.get()
    trace = get_trace_step(0)
    w_next = trace.w - canon.hparams.lr * trace.w_grad
    fig, ax = aix.asset.figure()
    aix.asset.plot_step(ax, loss_viewport(), loss_fn(), trace.w, w_next)
    return fig


def show_sgd_steps() -> list[str]:
    return [*_show_sgd_step(0), *_show_sgd_step(1)]


def show_sgd() -> Animation:
    fig, vsnapshot = _show_sgd_snapshot(0)
    animator = _NumberSgdAnimator(vsnapshot)
    return Animation(fig, animator, init_frames=20, active_frames=100, fps=30)


def _show_sgd_snapshot(step: int) -> tuple[Figure, NumberVisualSnapshot]:
    canon = aix.canon.get()
    snapshot = canon.history[step]
    fig, ax = aix.asset.figure()
    vsnapshot = NumberVisualSnapshot(ax, loss_viewport(), snapshot)
    return fig, vsnapshot


@dataclass(frozen=True)
class _NumberSgdAnimator:
    vsnapshot: NumberVisualSnapshot

    def animate(self, frame: int) -> list[Artist]:
        canon = aix.canon.get()
        snapshot = canon.history[frame]
        return self.vsnapshot.animate(snapshot)


def _show_sgd_step(step: int) -> list[str]:
    canon = aix.canon.get()
    trace = get_trace_step(step)
    grad_steps, delta_steps, update_steps = aix.asset.latex_sgd_step(trace, canon.hparams.lr)
    grad = aix.asset.equation(grad_steps)
    delta = aix.asset.equation(delta_steps)
    update = aix.asset.equation(update_steps)
    return [grad, delta, update]
