"""Story sequence for training the model."""

from ai_explainer.iris.asset.multiclass import Outputs, save

from ._show_learn import show_params, show_predictions, show_sgd


def produce(outputs: Outputs, *, video: bool = True) -> None:
    """Produce the outputs for training the model."""
    outputs = outputs.sequence("learn")
    if video:
        sgd = show_sgd()
        sgd.save(outputs.mp4("sgd"))
    else:
        outputs.skip()
    params = show_params()
    save(params, outputs.png("params"))
    predictions = show_predictions()
    save(predictions, outputs.txt("predictions"))
