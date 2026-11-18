"""Produce outputs for the number example."""

import sys
from pathlib import Path

from ai_explainer.number import story
from ai_explainer.number.asset import Outputs


def main(names: list[str]) -> None:
    """Produce outputs for the number example."""
    outputs = Outputs(Path("out")).act("number")
    outputs.mkdir()
    for name in names:
        match name:
            case "intro":
                story.intro.produce(outputs)
            case "loss":
                story.loss.produce(outputs)
            case "slope":
                story.slope.produce(outputs)
            case "learn":
                story.learn.produce(outputs)
            case _:
                print(f"unknown name: {name}")


main(sys.argv[1:])
