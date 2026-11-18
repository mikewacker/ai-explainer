from functools import cache

from torch import Tensor

import ai_explainer.wine as aix
from ai_explainer.wine.asset import LossCrossFn


@cache
def loss_mesh(alt: bool = False) -> Tensor:
    """Get the contour mesh for the loss function."""
    canon = aix.canon.get(alt=alt)
    return aix.asset.loss_mesh(canon.data.x, canon.data.y, loss_delta=5000)


@cache
def loss_fn_w1() -> LossCrossFn[Tensor]:
    canon = aix.canon.get()
    return aix.asset.loss_cross_fn(canon.data.x, canon.data.y, canon.create_model, j=1)


@cache
def loss_fn_w2() -> LossCrossFn[Tensor]:
    canon = aix.canon.get()
    return aix.asset.loss_cross_fn(canon.data.x, canon.data.y, canon.create_model, j=2)
