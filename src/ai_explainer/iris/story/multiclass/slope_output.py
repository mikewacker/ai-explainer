"""Story sequence for backpropagating through the output layer."""

from ai_explainer.iris.asset.multiclass import Outputs, save

from ._show_slope import show_a1_grad, show_output_grad


def produce(outputs: Outputs) -> None:
    """Produce the outputs for backpropagating through the output layer."""
    outputs = outputs.sequence("slope2")
    output_grad_phase0 = show_output_grad(phase=0)
    output_grad_phase0.save(outputs.png("output-grad-phase0"))
    output_grad_phase1 = show_output_grad(phase=1)
    output_grad_phase1.save(outputs.png("output-grad-phase1"))
    output_grad_phase2 = show_output_grad(phase=2)
    output_grad_phase2.save(outputs.png("output-grad-phase2"))
    a1_grad = show_a1_grad()
    save(a1_grad, outputs.txt("a1-grad"))
    output_grad_phase3 = show_output_grad(phase=3)
    output_grad_phase3.save(outputs.png("output-grad-phase3"))
    output_grad_phase4 = show_output_grad(phase=4)
    output_grad_phase4.save(outputs.png("output-grad-phase4"))
