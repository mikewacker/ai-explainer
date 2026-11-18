from functools import cache

import ai_explainer.iris as aix
from ai_explainer.iris.world import IrisTrace


@cache
def get_trace_sample() -> IrisTrace:
    canon = aix.canon.get()
    return aix.world.trace(canon.model_untrained, *canon.sample)
