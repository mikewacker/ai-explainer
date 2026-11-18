"""Story sequence for computing the gradient with respect to the logit."""

from ai_explainer.iris.asset.binary import Outputs, save

from ._show_slope import show_bce_logit, show_z2_grad


def produce(outputs: Outputs) -> None:
    """Produce the outputs for computing the gradient with respect to the logit."""
    outputs = outputs.sequence("slope1")
    z2_grad = show_z2_grad()
    save(z2_grad, outputs.txt("z2-grad"))
    bce_logits = show_bce_logit()
    save(bce_logits, outputs.png("bce-logit"))
