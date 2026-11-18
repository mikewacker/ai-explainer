"""Story sequence for computing loss for one sample."""

from ai_explainer.iris.asset.binary import Outputs, save

from ._show_loss import show_bce, show_loss_sample, show_loss_sample_neurons


def produce(outputs: Outputs) -> None:
    """Produce the outputs for computing loss for one sample."""
    outputs = outputs.sequence("loss1")
    loss_sample_neurons = show_loss_sample_neurons()
    loss_sample_neurons.save(outputs.png("loss-sample-neurons"))
    loss_sample_phase0 = show_loss_sample(phase=0)
    loss_sample_phase0.save(outputs.png("loss-sample-phase0"))
    loss_sample_phase1 = show_loss_sample(phase=1)
    loss_sample_phase1.save(outputs.png("loss-sample-phase1"))
    loss_sample_phase2 = show_loss_sample(phase=2)
    loss_sample_phase2.save(outputs.png("loss-sample-phase2"))
    loss_sample_phase3 = show_loss_sample(phase=3)
    loss_sample_phase3.save(outputs.png("loss-sample-phase3"))
    bce = show_bce()
    save(bce, outputs.png("bce"))
    loss_sample_phase4 = show_loss_sample(phase=4)
    loss_sample_phase4.save(outputs.png("loss-sample-phase4"))
