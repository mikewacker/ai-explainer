import ai_explainer.iris as aix
from ai_explainer.iris.asset.multiclass import Diagram, drawio

from ._trace import get_trace_sample


def show_z2_grad() -> list[str]:
    trace = get_trace_sample()
    grad_steps = aix.asset.multiclass.latex_z2_grad(trace)
    grad = aix.asset.multiclass.equation(grad_steps, approx_steps=2)
    return [grad]


def show_output_grad(*, phase: int = 5) -> Diagram:
    trace = get_trace_sample()
    diag = Diagram(drawio("slope_output"))
    if phase >= 1:
        diag.fill_tensor("z", trace.z2_grad, decimals=2)
    if phase == 1:
        diag.highlight(fill="z_color")
    if phase >= 2:
        diag.fill_tensor("w", trace.w2_grad, decimals=2)
        diag.fill_tensor("b", trace.b2_grad, decimals=2)
    if phase == 2:
        diag.highlight(fill="param_color")
    if phase == 3:
        diag.fill_tensor("a1", trace.a_grad[0], decimals=2)
    elif phase >= 4:
        diag.fill_tensor("a", trace.a_grad, decimals=2)
    if 3 <= phase <= 4:
        diag.highlight(fill="in_color")
    return diag


def show_a1_grad() -> list[str]:
    trace = get_trace_sample()
    link_steps, backlink_steps, grad_steps = aix.asset.multiclass.latex_a_grad(trace, j=1)
    grad_fn = aix.asset.multiclass.equation(grad_steps[:2])
    link_fn = aix.asset.multiclass.equation(link_steps)
    backlink_fn = aix.asset.multiclass.equation([psteps[:2] for psteps in backlink_steps])
    backlink = aix.asset.multiclass.equation(backlink_steps, approx_steps=1)
    grad = aix.asset.multiclass.equation(
        grad_steps, approx_steps=2, multiline=True, align_terms=True
    )
    return [grad_fn, link_fn, backlink_fn, backlink, grad]
