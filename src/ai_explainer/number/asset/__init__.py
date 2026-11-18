"""Tools for building hand-crafted assets."""

from ai_explainer.common.asset import *  # noqa: F403
from ai_explainer.common.asset import __all__ as _common_all

from ._learn import NumberVisualSnapshot, latex_sgd_step, plot_step
from ._loss import frame_dark, latex_loss, latex_se, plot_hill
from ._loss_fn import loss_fn
from ._slope import latex_grad, plot_tangent

__all__ = [
    *_common_all,
    "NumberVisualSnapshot",
    "frame_dark",
    "latex_grad",
    "latex_loss",
    "latex_se",
    "latex_sgd_step",
    "loss_fn",
    "plot_hill",
    "plot_step",
    "plot_tangent",
]
