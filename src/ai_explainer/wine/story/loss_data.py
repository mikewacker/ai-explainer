"""Story sequence for computing loss for the training data."""

from ai_explainer.wine.asset import Outputs

from ._show_loss import show_loss, show_loss_io, show_training_data
from ._show_model import show_model_io


def produce(outputs: Outputs) -> None:
    """Produce the outputs for computing loss for the training data."""
    outputs = outputs.sequence("loss2")
    training_data = show_training_data()
    training_data.save(outputs.png("training-data"))
    loss_phase0 = show_loss(phase=0)
    loss_phase0.save(outputs.png("loss-phase0"))
    loss_phase1 = show_loss(phase=1)
    loss_phase1.save(outputs.png("loss-phase1"))
    loss_phase2 = show_loss(phase=2)
    loss_phase2.save(outputs.png("loss-phase2"))
    loss_phase3 = show_loss(phase=3)
    loss_phase3.save(outputs.png("loss-phase3"))
    loss_phase4 = show_loss(phase=4)
    loss_phase4.save(outputs.png("loss-phase4"))
    loss_phase5 = show_loss(phase=5)
    loss_phase5.save(outputs.png("loss-phase5"))
    model_io = show_model_io()
    model_io.save(outputs.png("model-io"))
    loss_io = show_loss_io()
    loss_io.save(outputs.png("loss-io"))
