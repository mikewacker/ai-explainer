from torch import Tensor

from ai_explainer.iris.asset.binary import pd
from ai_explainer.iris.world import IrisTrace


def latex_z2_grad(trace: IrisTrace) -> list[str]:
    """Get the steps to compute the gradient of the logit in the output layer."""
    return [
        pd("L", "z^{(2)}"),
        r"\hat{y} - y",
        f"{trace.y_hat:.2f} - {trace.y:g}",
        f"{trace.z2_grad:.2f}",
    ]


def latex_w2_grad(trace: IrisTrace, *, j: int) -> tuple[list[str], list[str], list[str]]:
    """Get the steps to compute the gradient of a weight in the output layer."""
    link_steps = _latex_z2_link(trace)
    backlink_steps = _latex_w2_backlink(trace, j)
    grad_steps = latex_backstep(
        "z^{(2)}", f"w^{{(2)}}_{j}", trace.z2_grad, backlink_steps[-1], trace.w2_grad[j - 1]
    )
    return link_steps, backlink_steps, grad_steps


def latex_a_grad(trace: IrisTrace, *, j: int) -> tuple[list[str], list[str], list[str]]:
    """Get the steps to compute the gradient of an activation in the hidden layer."""
    link_steps = _latex_z2_link(trace)
    backlink_steps = _latex_a_backlink(trace, j)
    grad_steps = latex_backstep(
        "z^{(2)}", f"a_{j}", trace.z2_grad, backlink_steps[-1], trace.a_grad[j - 1]
    )
    return link_steps, backlink_steps, grad_steps


def latex_z1_grad(trace: IrisTrace, *, j: int) -> tuple[list[str], list[str], list[str]]:
    """Get the steps to compute the gradient of a preactivation in the hidden layer."""
    link_steps = _latex_a_link(j)
    backlink_steps = _latex_z1_backlink(trace, j)
    grad_steps = latex_backstep(
        f"a_{j}", f"z^{{(1)}}_{j}", trace.a_grad[j - 1], backlink_steps[-1], trace.z1_grad[j - 1]
    )
    return link_steps, backlink_steps, grad_steps


def latex_backsteps(
    out_vars: list[str], in_var: str, out_grad: Tensor, backlinks: list[str], in_grad: Tensor
) -> list[str]:
    steps_terms = [
        [pd("L", in_var)],
        [f"{pd('L', out_var)} {pd(out_var, in_var)}" for out_var in out_vars],
        [
            rf"{out_pd:.2f} \cdot {backlink}"
            for _, out_pd, backlink in zip(out_vars, out_grad, backlinks, strict=True)
        ],
        [f"{in_grad:.2f}"],
    ]
    return [" + ".join(terms) for terms in steps_terms]


def latex_backstep(
    out_var: str, in_var: str, out_grad: Tensor, backlink: str, in_grad: Tensor
) -> list[str]:
    return latex_backsteps([out_var], in_var, out_grad.unsqueeze(0), [backlink], in_grad)


def _latex_z2_link(trace: IrisTrace) -> list[str]:
    d_in = trace.a.shape[-1]
    steps_terms = [
        ["z^{(2)}"],
        [*[f"w^{{(2)}}_{j} a_{j}" for j in range(1, d_in + 1)], "b^{(2)}"],
    ]
    return [" + ".join(terms) for terms in steps_terms]


def _latex_w2_backlink(trace: IrisTrace, j: int) -> list[str]:
    return [
        pd("z^{(2)}", f"w^{{(2)}}_{j}"),
        f"a_{j}",
        f"{trace.a[j - 1]:.2f}",
    ]


def _latex_a_backlink(trace: IrisTrace, j: int) -> list[str]:
    return [
        pd("z^{(2)}", f"a_{j}"),
        f"w^{{(2)}}_{j}",
        f"{trace.w2[j - 1]:.2f}",
    ]


def _latex_a_link(j: int) -> list[str]:
    return [
        f"a_{j}",
        rf"\operatorname{{ReLU}}(z^{{(1)}}_{j})",
    ]


def _latex_z1_backlink(trace: IrisTrace, j: int) -> list[str]:
    return [
        pd(f"a_{j}", f"z^{{(1)}}_{j}"),
        *[
            (
                r"\begin{cases} "
                rf"1, & \text{{if }} {z} > 0 \\ "
                rf"0, & \text{{if }} {z} \le 0 "
                r"\end{cases}"
            )
            for z in [f"z^{{(1)}}_{j}", f"{trace.z1[j - 1]:.2f}"]
        ],
        f"{1 if trace.z1[j - 1] > 0 else 0:g}",
    ]
