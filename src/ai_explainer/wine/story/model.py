"""Story sequence for building the model."""

from ai_explainer.wine.asset import Outputs, save

from ._show_model import show_model, show_model_blackbox, show_model_hand, show_prediction_hand


def produce(outputs: Outputs) -> None:
    """Produce the outputs for building the model."""
    outputs = outputs.sequence("model")
    model_blackbox_wine = show_model_blackbox(show_names=True)
    model_blackbox_wine.save(outputs.png("model-blackbox-wine"))
    model_blackbox = show_model_blackbox()
    model_blackbox.save(outputs.png("model-blackbox"))
    model_hand = show_model_hand()
    model_hand.save(outputs.png("model-hand"))
    prediction_hand = show_prediction_hand()
    save(prediction_hand, outputs.txt("prediction-hand"))
    model_phase0 = show_model(phase=0)
    model_phase0.save(outputs.png("model-phase0"))
