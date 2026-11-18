"""Story sequence for feeding forward through the output layer."""

from ai_explainer.iris.asset.binary import Outputs, save

from ._show_model import show_nn, show_output, show_sigmoid


def produce(outputs: Outputs) -> None:
    """Produce the outputs for feeding forward through the output layer."""
    outputs = outputs.sequence("model5")
    nn_output = show_nn(highlight="out")
    nn_output.save(outputs.png("nn-output"))
    output_phase0 = show_output(phase=0)
    output_phase0.save(outputs.png("output-phase0"))
    output_phase1 = show_output(phase=1)
    output_phase1.save(outputs.png("output-phase1"))
    output_phase2 = show_output(phase=2)
    output_phase2.save(outputs.png("output-phase2"))
    sigmoid = show_sigmoid()
    save(sigmoid, outputs.png("sigmoid"))
    output_phase3 = show_output(phase=3)
    output_phase3.save(outputs.png("output-phase3"))
