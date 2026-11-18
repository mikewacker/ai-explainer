import ai_explainer.wine as aix
from ai_explainer.wine.asset import Diagram, drawio

from ._trace import get_trace_sample


def show_model_blackbox(*, show_names: bool = False) -> Diagram:
    name = "model_blackbox-wine" if show_names else "model_blackbox"
    return Diagram(drawio(name))


def show_model_hand() -> Diagram:
    canon = aix.canon.get()
    diag = Diagram(drawio("model_hand"))
    diag.fill_tensor("w", canon.model_untrained.params())
    return diag


def show_prediction_hand() -> list[str]:
    trace = get_trace_sample()
    prediction_steps = aix.asset.latex_prediction_hand(trace)
    prediction = aix.asset.equation(prediction_steps, multiline=True, align_terms=True)
    return [prediction]


def show_model(*, phase: int = 3) -> Diagram:
    trace = get_trace_sample()
    diag = Diagram(drawio("model_ai"))
    if phase >= 1:
        diag.fill_tensor("w", trace.w)
    if phase == 1:
        diag.highlight(fill="param_color")
    if phase >= 2:
        diag.fill_tensor("x", trace.x)
        diag.fill_tensor("yhat", trace.y_hat)
    if phase == 2:
        diag.highlight(fill="in_color")
        diag.highlight(fill="out_color")
    return diag


def show_prediction_no_params() -> list[str]:
    trace = get_trace_sample()
    prediction_steps = aix.asset.latex_prediction_no_params(trace)
    prediction = aix.asset.equation(prediction_steps, multiline=True, align_terms=True)
    return [prediction]


def show_model_io() -> Diagram:
    canon = aix.canon.get()
    diag = Diagram(drawio("model_ai"))
    diag.highlight_io()
    diag.fill_tensor("w", canon.model_untrained.params())
    return diag
