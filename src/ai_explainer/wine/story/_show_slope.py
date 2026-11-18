import ai_explainer.wine as aix

from ._trace import get_trace_data


def show_grad() -> list[str]:
    trace = get_trace_data()
    grad_steps = aix.asset.latex_grad(trace)
    grad_sym = aix.asset.equation([psteps[:1] for psteps in grad_steps])
    grad_fn = aix.asset.equation([psteps[:2] for psteps in grad_steps])
    grad = aix.asset.equation(grad_steps, approx_steps=1, multiline=True)
    return [grad_sym, grad_fn, grad]
