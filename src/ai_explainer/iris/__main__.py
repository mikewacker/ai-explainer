"""Produce outputs for the iris example."""

import sys
from pathlib import Path

from ai_explainer.iris import story
from ai_explainer.iris.asset.binary import Outputs


def main(args: list[str]) -> None:
    """Produce outputs for the iris example."""
    if not args:
        print("missing task")
        return

    task = args[0]
    if task == "binary":
        if len(args) < 2:
            print("missing act")
            return

        act = args[1]
        match act:
            case "model":
                _main_binary_model(args[2:])
            case "train":
                _main_binary_train(args[2:])
            case _:
                print(f"unknown act: {act}")
        return

    if task == "multiclass":
        _main_multiclass(args[1:])
        return

    print(f"unknown task: {task}")


def _main_binary_model(names: list[str]) -> None:
    outputs = Outputs(Path("out")).act("iris-binary-model")
    outputs.mkdir()
    for name in names:
        match name:
            case "nn":
                story.binary.model.model_nn.produce(outputs)
            case "input":
                story.binary.model.model_input.produce(outputs)
            case "hidden1":
                story.binary.model.model_hidden1.produce(outputs)
            case "hidden":
                story.binary.model.model_hidden.produce(outputs)
            case "output":
                story.binary.model.model_output.produce(outputs)
            case _:
                print(f"unknown name: {name}")


def _main_binary_train(names: list[str]) -> None:
    outputs = Outputs(Path("out")).act("iris-binary-train")
    outputs.mkdir()
    for name in names:
        match name:
            case "loss-sample":
                story.binary.train.loss_sample.produce(outputs)
            case "loss-data":
                story.binary.train.loss_data.produce(outputs)
            case "slope-logit":
                story.binary.train.slope_logit.produce(outputs)
            case "slope-output":
                story.binary.train.slope_output.produce(outputs)
            case "slope-hidden":
                story.binary.train.slope_hidden.produce(outputs)
            case "learn":
                story.binary.train.learn.produce(outputs)
            case _:
                print(f"unknown name: {name}")


def _main_multiclass(names: list[str]) -> None:
    outputs = Outputs(Path("out")).act("iris-multiclass")
    outputs.mkdir()
    for name in names:
        match name:
            case "model-nn":
                story.multiclass.model_nn.produce(outputs)
            case "model-output":
                story.multiclass.model_output.produce(outputs)
            case "loss-sample":
                story.multiclass.loss_sample.produce(outputs)
            case "loss-data":
                story.multiclass.loss_data.produce(outputs)
            case "slope-logit":
                story.multiclass.slope_logit.produce(outputs)
            case "slope-output":
                story.multiclass.slope_output.produce(outputs)
            case "learn":
                story.multiclass.learn.produce(outputs)
            case _:
                print(f"unknown name: {name}")


main(sys.argv[1:])
