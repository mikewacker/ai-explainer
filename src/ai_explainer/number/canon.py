"""Canon for the number example."""

from functools import cache
from typing import override

import torch
from torch import Tensor

import ai_explainer.number as aix
from ai_explainer.common.canon import Canon
from ai_explainer.number.world import NumberData, NumberHyperparams, NumberModel, NumberSnapshot


class NumberCanon(Canon[NumberModel, Tensor, NumberData, NumberHyperparams]):
    """Canon for the number example."""

    def __init__(
        self, *, data: NumberData, sample: tuple[Tensor, Tensor], hparams: NumberHyperparams
    ) -> None:
        """Create the canon."""
        super().__init__(data, sample, hparams)

    @override
    def create_model(self) -> NumberModel:
        return NumberModel(w=self.hparams.w_init)

    @override
    def _train(self, model: NumberModel) -> list[NumberSnapshot]:
        return aix.world.train(model, self.hparams, self.data.y)


@cache
def get() -> NumberCanon:
    """Get the canon for the number example."""
    data = aix.world.load_data()
    return NumberCanon(
        data=data,
        sample=(torch.empty(0), data.y),
        hparams=NumberHyperparams(w_init=torch.tensor(5.0), lr=0.01, steps=100),
    )
