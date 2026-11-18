from collections.abc import Callable
from functools import cache

from torch import Tensor

import ai_explainer.number as aix
from ai_explainer.number.asset import Viewport


@cache
def loss_fn() -> Callable[[Tensor], Tensor]:
    canon = aix.canon.get()
    return aix.asset.loss_fn(canon.data.y)


@cache
def loss_viewport() -> Viewport:
    return Viewport(x_var="w", xlim=(-1, 7), y_var="L", ylim=(-1, 7))
