"""Story sequence for computing loss."""

from ai_explainer.number.asset import Outputs, save

from ._show_loss import show_dark, show_hill, show_loss, show_se


def produce(outputs: Outputs) -> None:
    """Produce the outputs for computing loss."""
    outputs = outputs.sequence("loss")
    se = show_se()
    save(se, outputs.txt("se"))
    loss = show_loss()
    save(loss, outputs.png("loss"))
    hill = show_hill()
    save(hill, outputs.png("hill"))
    dark = show_dark()
    save(dark, outputs.png("dark"))
