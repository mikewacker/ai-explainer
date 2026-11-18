"""Story sequence for the introduction."""

from ai_explainer.number.asset import Outputs, save

from ._show_intro import show_function


def produce(outputs: Outputs) -> None:
    """Produce the outputs for the introduction."""
    outputs = outputs.sequence("intro")
    outputs.skip()  # external asset
    function = show_function()
    save(function, outputs.png("function"))
