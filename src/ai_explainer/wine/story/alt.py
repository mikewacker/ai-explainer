"""Story sequence for changing the training data."""

from ai_explainer.wine.asset import Outputs, save

from ._show_learn import show_params, show_predictions_alt
from ._show_loss import show_hill_alt, show_loss_io, show_training_data


def produce(outputs: Outputs) -> None:
    """Produce the outputs for changing the training data."""
    outputs = outputs.sequence("alt")
    training_data_alt = show_training_data(alt=True)
    training_data_alt.save(outputs.png("training-data-alt"))
    loss_io_alt = show_loss_io(alt=True)
    loss_io_alt.save(outputs.png("loss-io-alt"))
    hill_alt = show_hill_alt()
    save(hill_alt, outputs.png("hill-alt"))
    params_alt = show_params(alt=True)
    save(params_alt, outputs.png("params-alt"))
    predictions_alt = show_predictions_alt()
    predictions_alt.save(outputs.png("predictions-alt"))
