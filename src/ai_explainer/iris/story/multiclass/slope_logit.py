"""Story sequence for computing the gradient with respect to the logits."""

from ai_explainer.iris.asset.multiclass import Outputs, save

from ._show_slope import show_z2_grad


def produce(outputs: Outputs) -> None:
    """Produce the outputs for computing the gradient with respect to the logits."""
    outputs = outputs.sequence("slope1")
    z2_grad = show_z2_grad()
    save(z2_grad, outputs.txt("z2-grad"))
