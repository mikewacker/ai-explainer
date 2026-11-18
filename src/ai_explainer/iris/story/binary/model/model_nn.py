"""Story sequence for building the neural network."""

from ai_explainer.iris.asset.binary import Outputs

from ._show_model import show_nn, show_sample


def produce(outputs: Outputs) -> None:
    """Produce the outputs for building the neural network."""
    outputs = outputs.sequence("model1")
    nn = show_nn()
    nn.save(outputs.png("nn"))
    sample = show_sample()
    sample.save(outputs.png("sample"))
    outputs.skip()  # external asset
