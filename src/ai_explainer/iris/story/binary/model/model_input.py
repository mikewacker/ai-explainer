"""Story sequence for feeding forward through the input layer."""

from ai_explainer.iris.asset.binary import Outputs

from ._show_model import show_input, show_nn


def produce(outputs: Outputs) -> None:
    """Produce the outputs for feeding forward through the input layer."""
    outputs = outputs.sequence("model2")
    nn_input = show_nn(highlight="in")
    nn_input.save(outputs.png("nn-input"))
    input_ = show_input()
    input_.save(outputs.png("input"))
