"""Story sequence for computing loss for the training data."""

from ai_explainer.iris.asset.multiclass import Outputs

from ._show_loss import show_loss


def produce(outputs: Outputs) -> None:
    """Produce the outputs for computing loss for the training data."""
    outputs = outputs.sequence("loss2")
    loss = show_loss()
    loss.save(outputs.png("loss"))
