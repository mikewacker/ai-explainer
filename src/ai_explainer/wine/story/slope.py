"""Story sequence for computing the gradient."""

from ai_explainer.wine.asset import Outputs, save

from ._show_loss import show_hill, show_hill_w1, show_hill_w2, show_loss
from ._show_slope import show_grad


def produce(outputs: Outputs) -> None:
    """Produce the outputs for computing the gradient."""
    outputs = outputs.sequence("slope")
    hill = show_hill()
    save(hill, outputs.png("hill"))
    hill_w1 = show_hill_w1()
    save(hill_w1, outputs.png("hill-w1"))
    hill_w2 = show_hill_w2()
    save(hill_w2, outputs.png("hill-w2"))
    grad = show_grad()
    save(grad, outputs.txt("grad"))
    loss = show_loss()
    loss.save(outputs.png("loss"))
