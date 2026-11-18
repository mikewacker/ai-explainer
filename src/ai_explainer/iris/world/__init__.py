"""World of the iris example."""

from ._data import IrisData, IrisSpecies, load_data
from ._model import (
    IrisHyperparams,
    IrisNn,
    IrisParams,
    IrisSnapshot,
    IrisTask,
    get_loss_criterion,
    test,
    train,
)
from ._trace import IrisTrace, trace

__all__ = [
    "IrisData",
    "IrisHyperparams",
    "IrisNn",
    "IrisParams",
    "IrisSnapshot",
    "IrisSpecies",
    "IrisTask",
    "IrisTrace",
    "get_loss_criterion",
    "load_data",
    "test",
    "trace",
    "train",
]
