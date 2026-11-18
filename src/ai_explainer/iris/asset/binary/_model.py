import torch
from torch import Tensor


def latex_linear1(w: Tensor, b: Tensor, x: Tensor) -> list[str]:
    """Get the steps to compute a single linear combination in the hidden layer."""
    d_in = x.shape[-1]
    z = torch.inner(x, w) + b
    steps_terms = [
        ["z"],
        [*[f"w_{j} x_{j}" for j in range(1, d_in + 1)], "b"],
        [*[rf"{w[j - 1]:g} \cdot {x[j - 1]:g}" for j in range(1, d_in + 1)], f"{b:g}"],
        [*[rf"{w[j - 1] * x[j - 1]:g}" for j in range(1, d_in + 1)], f"{b:g}"],
        [f"{z:g}"],
    ]
    return [" + ".join(terms) for terms in steps_terms]


def latex_linear() -> list[str]:
    """Get the steps to compute the linear combinations in the hidden layer."""
    return [
        r"\mathbf{Z}^{(1)}",
        r"\mathbf{X} \mathbf{W}^{(1) \mathsf{T}} + \mathbf{b}^{(1) \mathsf{T}}",
    ]
