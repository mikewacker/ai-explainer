"""Story sequence for training the model."""

from ai_explainer.number.asset import Outputs, save

from ._show_learn import (
    show_sgd,
    show_sgd_steps,
    show_step_downhill,
    show_step_large,
    show_step_uphill,
)


def produce(outputs: Outputs, *, video: bool = True) -> None:
    """Produce the outputs for training the model."""
    outputs = outputs.sequence("learn")
    step_large = show_step_large()
    save(step_large, outputs.png("step-large"))
    step_uphill = show_step_uphill()
    save(step_uphill, outputs.png("step-uphill"))
    step_downhill = show_step_downhill()
    save(step_downhill, outputs.png("step-downhill"))
    sgd_steps = show_sgd_steps()
    save(sgd_steps, outputs.txt("sgd-steps"))
    if video:
        sgd = show_sgd()
        sgd.save(outputs.mp4("sgd"))
    else:
        outputs.skip()
