"""Produce outputs for the wine example."""

import sys
from pathlib import Path

from ai_explainer.wine import story
from ai_explainer.wine.asset import Outputs


def main(names: list[str]) -> None:
    """Produce outputs for the wine example."""
    outputs = Outputs(Path("out")).act("wine")
    outputs.mkdir()
    for name in names:
        match name:
            case "model":
                story.model.produce(outputs)
            case "loss-sample":
                story.loss_sample.produce(outputs)
            case "loss-data":
                story.loss_data.produce(outputs)
            case "slope":
                story.slope.produce(outputs)
            case "learn":
                story.learn.produce(outputs)
            case "alt":
                story.alt.produce(outputs)
            case _:
                print(f"unknown name: {name}")


main(sys.argv[1:])
