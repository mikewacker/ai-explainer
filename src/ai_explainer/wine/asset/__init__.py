"""Tools for building hand-crafted assets."""

from ai_explainer.common.asset import *  # noqa: F403
from ai_explainer.common.asset import __all__ as _common_all

from ._drawio import drawio
from ._learn import WineVisualSnapshot, latex_sgd_step, table_predictions
from ._loss import (
    latex_se,
    plot_loss_contours,
    plot_loss_fn,
    plot_loss_fn_w1,
    plot_loss_fn_w2,
    table_training_data,
)
from ._loss_fn import loss_cross_fn, loss_mesh
from ._model import latex_prediction_hand, latex_prediction_no_params
from ._slope import latex_grad

__all__ = [
    *_common_all,
    "WineVisualSnapshot",
    "drawio",
    "latex_grad",
    "latex_prediction_hand",
    "latex_prediction_no_params",
    "latex_se",
    "latex_sgd_step",
    "loss_cross_fn",
    "loss_mesh",
    "plot_loss_contours",
    "plot_loss_fn",
    "plot_loss_fn_w1",
    "plot_loss_fn_w2",
    "table_predictions",
    "table_training_data",
]
