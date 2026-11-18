"""Story sequence for computing the gradient."""

from ai_explainer.number.asset import Outputs, save

from ._show_slope import show_grad, show_tangent, show_up


def produce(outputs: Outputs) -> None:
    """Produce the outputs for computing the gradient."""
    outputs = outputs.sequence("slope")
    tangent = show_tangent()
    save(tangent, outputs.png("tangent"))
    grad = show_grad()
    save(grad, outputs.txt("grad"))
    up = show_up()
    save(up, outputs.png("up"))
