"""Story sequence for backpropagating through the hidden layer."""

from ai_explainer.iris.asset.binary import Outputs, save

from ._show_slope import show_hidden_a1_grad, show_hidden_grad, show_z11_grad


def produce(outputs: Outputs) -> None:
    """Produce the outputs for backpropagating through the hidden layer."""
    outputs = outputs.sequence("slope3")
    hidden_a1_grad_phase0 = show_hidden_a1_grad(phase=0)
    hidden_a1_grad_phase0.save(outputs.png("hidden-a1-grad-phase0"))
    z11_grad = show_z11_grad()
    save(z11_grad, outputs.txt("z11-grad"))
    hidden_a1_grad_phase1 = show_hidden_a1_grad(phase=1)
    hidden_a1_grad_phase1.save(outputs.png("hidden-a1-grad-phase1"))
    hidden_a1_grad_phase2 = show_hidden_a1_grad(phase=2)
    hidden_a1_grad_phase2.save(outputs.png("hidden-a1-grad-phase2"))
    hidden_grad = show_hidden_grad()
    hidden_grad.save(outputs.png("hidden-grad"))
