from collections.abc import Iterable

from ai_explainer.wine.asset import pd
from ai_explainer.wine.world import WineTrace


def latex_grad(trace: WineTrace) -> list[list[str]]:
    """Get the steps to compute the gradient."""
    m, d_in = trace.x.shape
    return [
        [
            pd("L", f"w_{j}"),
            _latex_mean(
                rf"{2 * trace.x[i - 1, j - 1]:g}(\hat{{y}}_{i} - {trace.y[i - 1]:g})"
                for i in range(1, m + 1)
            ),
            _latex_mean(
                rf"{2 * trace.x[i - 1, j - 1]:g}({trace.y_hat[i - 1]:g} - {trace.y[i - 1]:g})"
                for i in range(1, m + 1)
            ),
            f"{trace.w_grad[j - 1]:.2f}",
        ]
        for j in range(1, d_in + 1)
    ]


def _latex_mean(terms: Iterable[str]) -> str:
    terms = list(terms)
    num = " + ".join(terms)
    denom = f"{len(terms)}"
    return rf"\frac{{{num}}}{{{denom}}}"
