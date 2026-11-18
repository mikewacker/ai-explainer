"""Story sequence for computing loss for one sample."""

from ai_explainer.wine.asset import Outputs, save

from ._show_loss import show_loss_sample, show_sample, show_se
from ._show_model import show_model, show_prediction_no_params


def produce(outputs: Outputs) -> None:
    """Produce the outputs for computing loss for one sample."""
    outputs = outputs.sequence("loss1")
    sample = show_sample()
    sample.save(outputs.png("sample"))
    prediction_no_params = show_prediction_no_params()
    save(prediction_no_params, outputs.txt("prediction-no-params"))
    model_phase1 = show_model(phase=1)
    model_phase1.save(outputs.png("model-phase1"))
    model_phase2 = show_model(phase=2)
    model_phase2.save(outputs.png("model-phase2"))
    se = show_se()
    save(se, outputs.txt("se"))
    loss_sample = show_loss_sample()
    loss_sample.save(outputs.png("loss-sample"))
