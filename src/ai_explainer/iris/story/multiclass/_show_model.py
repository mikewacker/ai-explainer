from ai_explainer.iris.asset.multiclass import Diagram, drawio

from ._trace import get_trace_sample


def show_nn() -> Diagram:
    return Diagram(drawio("model_nn"))


def show_output(*, phase: int = 4) -> Diagram:
    trace = get_trace_sample()
    diag = Diagram(drawio("model_output"))
    if phase == 0:
        diag.highlight_io()
    if phase >= 1:
        diag.fill_tensor("w", trace.w2, decimals=2)
        diag.fill_tensor("b", trace.b2, decimals=2)
    if phase == 1:
        diag.highlight(fill="param_color")
    if phase >= 2:
        diag.fill_tensor("a", trace.a, decimals=2)
        diag.fill_tensor("z", trace.z2, decimals=2)
    if phase == 2:
        diag.highlight(fill="in_color")
        diag.highlight(fill="z_color")
    if phase >= 3:
        diag.fill_tensor("yhat", trace.y_hat, decimals=2)
    if phase == 3:
        diag.highlight(fill="out_color")
    return diag
