"""Story sequence for computing loss for the training data."""

from ai_explainer.iris.asset.binary import Outputs, save

from ._show_loss import show_linear, show_loss, show_loss_sample_matrix


def produce(outputs: Outputs) -> None:
    """Produce the outputs for computing loss for the training data."""
    outputs = outputs.sequence("loss2")
    outputs.skip()  # external asset
    loss_sample_matrix = show_loss_sample_matrix()
    loss_sample_matrix.save(outputs.png("loss-sample-matrix"))
    loss = show_loss()
    loss.save(outputs.png("loss"))
    linear = show_linear()
    save(linear, outputs.txt("linear"))
    loss_io = show_loss(show_io=True)
    loss_io.save(outputs.png("loss-io"))
