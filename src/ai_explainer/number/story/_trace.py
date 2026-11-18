from functools import cache

from torch import Tensor

import ai_explainer.number as aix
from ai_explainer.number.world import NumberModel, NumberTrace


@cache
def get_trace(w: Tensor | None = None) -> NumberTrace:
    canon = aix.canon.get()
    model = canon.model_untrained if w is None else NumberModel(w=w)
    return aix.world.trace(model, canon.data.y)


@cache
def get_trace_step(step: int) -> NumberTrace:
    canon = aix.canon.get()
    model = NumberModel(w=canon.history[step].params)
    return aix.world.trace(model, canon.data.y)
