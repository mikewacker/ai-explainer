"""Story sequence for feeding forward through a hidden layer with one neuron."""

from ai_explainer.iris.asset.binary import Outputs, save

from ._show_model import (
    show_hidden,
    show_hidden1,
    show_hidden1_scalar,
    show_linear1,
    show_nn,
    show_nn_hidden1,
    show_relu,
)


def produce(outputs: Outputs) -> None:
    """Produce the outputs for feeding forward through a hidden layer with one neuron."""
    outputs = outputs.sequence("model3")
    nn_hidden = show_nn(highlight="hidden")
    nn_hidden.save(outputs.png("nn-hidden"))
    hidden_phase0 = show_hidden(phase=0)
    hidden_phase0.save(outputs.png("hidden-phase0"))
    nn_hidden1 = show_nn_hidden1()
    nn_hidden1.save(outputs.png("nn-hidden1"))
    hidden1_scalar = show_hidden1_scalar()
    hidden1_scalar.save(outputs.png("hidden1-scalar"))
    hidden1_phase0 = show_hidden1(phase=0)
    hidden1_phase0.save(outputs.png("hidden1-phase0"))
    hidden1_phase1 = show_hidden1(phase=1)
    hidden1_phase1.save(outputs.png("hidden1-phase1"))
    hidden1_phase2 = show_hidden1(phase=2)
    hidden1_phase2.save(outputs.png("hidden1-phase2"))
    relu = show_relu()
    save(relu, outputs.png("relu"))
    hidden1_phase3 = show_hidden1(phase=3)
    hidden1_phase3.save(outputs.png("hidden1-phase3"))
    linear1 = show_linear1()
    save(linear1, outputs.txt("linear1"))
