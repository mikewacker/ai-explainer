from functools import cache

import ai_explainer.wine as aix
from ai_explainer.wine.world import WineModel, WineTrace


@cache
def get_trace_sample() -> WineTrace:
    canon = aix.canon.get()
    return aix.world.trace(canon.model_untrained, *canon.sample)


@cache
def get_trace_data() -> WineTrace:
    canon = aix.canon.get()
    return aix.world.trace(canon.model_untrained, canon.data.x, canon.data.y)


@cache
def get_trace_step(step: int) -> WineTrace:
    canon = aix.canon.get()
    model = WineModel(w=canon.history[step].params)
    return aix.world.trace(model, canon.data.x, canon.data.y)
