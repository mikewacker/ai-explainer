"""Canon for the wine example."""

from functools import cache
from typing import override

import torch
from torch import Tensor

import ai_explainer.wine as aix
from ai_explainer.common.canon import Canon
from ai_explainer.wine.world import WineData, WineHyperparams, WineModel, WineSnapshot


class WineCanon(Canon[WineModel, Tensor, WineData, WineHyperparams]):
    """Canon for the wine example."""

    def __init__(
        self, *, data: WineData, sample: tuple[Tensor, Tensor], hparams: WineHyperparams
    ) -> None:
        """Create the canon."""
        super().__init__(data, sample, hparams)

    @override
    def create_model(self) -> WineModel:
        return WineModel(w=self.hparams.w_init)

    @override
    def _train(self, model: WineModel) -> list[WineSnapshot]:
        return aix.world.train(model, self.hparams, self.data.x, self.data.y)


@cache
def get(alt: bool = False) -> WineCanon:
    """Get the canon for the wine example."""
    data = aix.world.load_data()
    sample_idxs = [44, 64, 174]  # similar parameters as the full dataset
    data = WineData(x=data.x[sample_idxs], y=data.y[sample_idxs])
    if alt:
        data.x[2, 0] -= 1.0
    return WineCanon(
        data=data,
        sample=(data.x[0], data.y[0]),
        hparams=WineHyperparams(w_init=torch.tensor([90.0, 340.0]), lr=0.01, steps=500),
    )
