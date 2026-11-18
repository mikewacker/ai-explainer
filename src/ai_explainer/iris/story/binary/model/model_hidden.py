"""Story sequence for feeding forward through the hidden layer."""

from ai_explainer.iris.asset.binary import Outputs

from ._show_model import show_hidden


def produce(outputs: Outputs) -> None:
    """Produce the outputs for feeding forward through the hidden layer."""
    outputs = outputs.sequence("model4")
    hidden_phase1 = show_hidden(phase=1)
    hidden_phase1.save(outputs.png("hidden-phase1"))
    hidden_phase2 = show_hidden(phase=2)
    hidden_phase2.save(outputs.png("hidden-phase2"))
    hidden_phase3 = show_hidden(phase=3)
    hidden_phase3.save(outputs.png("hidden-phase3"))
