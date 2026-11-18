"""Story sequence for computing loss for one sample."""

from ai_explainer.iris.asset.multiclass import Outputs, save

from ._show_loss import show_ce, show_loss_sample


def produce(outputs: Outputs) -> None:
    """Produce the outputs for computing loss for one sample."""
    outputs = outputs.sequence("loss1")
    loss_sample_phase0 = show_loss_sample(phase=0)
    loss_sample_phase0.save(outputs.png("loss-sample-phase0"))
    loss_sample_phase1 = show_loss_sample(phase=1)
    loss_sample_phase1.save(outputs.png("loss-sample-phase1"))
    loss_sample_phase2 = show_loss_sample(phase=2)
    loss_sample_phase2.save(outputs.png("loss-sample-phase2"))
    loss_sample_phase3 = show_loss_sample(phase=3)
    loss_sample_phase3.save(outputs.png("loss-sample-phase3"))
    ce = show_ce()
    save(ce, outputs.png("ce"))
    loss_sample_phase4 = show_loss_sample(phase=4)
    loss_sample_phase4.save(outputs.png("loss-sample-phase4"))
