from ai_explainer.iris.asset.binary._slope import latex_backsteps
from ai_explainer.iris.asset.multiclass import pd
from ai_explainer.iris.world import IrisTrace


def latex_z2_grad(trace: IrisTrace) -> list[list[str]]:
    """Get the steps to compute the gradient of the logits in the output layer."""
    d_in = trace.z2.shape[-1]
    return [
        [
            pd("L", f"z^{{(2)}}_{j}"),
            rf"\hat{{y}}_{j} - y_{j}",
            f"{trace.y_hat[j - 1]:.2f} - {trace.y[j - 1]:g}",
            f"{trace.z2_grad[j - 1]:.2f}",
        ]
        for j in range(1, d_in + 1)
    ]


def latex_a_grad(trace: IrisTrace, *, j: int) -> tuple[list[list[str]], list[list[str]], list[str]]:
    """Get the steps to compute the gradient of an activation in the hidden layer."""
    d_out = trace.z2.shape[-1]
    link_steps = [_latex_z2_link(trace, k) for k in range(1, d_out + 1)]
    backlink_steps = [_latex_a_backlink(trace, j, k) for k in range(1, d_out + 1)]
    grad_steps = latex_backsteps(
        [f"z^{{(2)}}_{k}" for k in range(1, d_out + 1)],
        f"a_{j}",
        trace.z2_grad,
        [psteps[-1] for psteps in backlink_steps],
        trace.a_grad[j - 1],
    )
    return link_steps, backlink_steps, grad_steps


def _latex_z2_link(trace: IrisTrace, k: int) -> list[str]:
    d_in = trace.a.shape[-1]
    steps_terms = [
        [f"z^{{(2)}}_{k}"],
        [*[f"w^{{(2)}}_{{{k}{j}}} a_{j}" for j in range(1, d_in + 1)], f"b^{{(2)}}_{k}"],
    ]
    return [" + ".join(terms) for terms in steps_terms]


def _latex_a_backlink(trace: IrisTrace, j: int, k: int) -> list[str]:
    return [
        pd(f"z^{{(2)}}_{k}", f"a_{j}"),
        f"w^{{(2)}}_{{{k}{j}}}",
        f"{trace.w2[k - 1, j - 1]:.2f}",
    ]
