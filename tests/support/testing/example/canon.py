"""Canon for the test example."""

from functools import cache
from typing import override

from torch import Tensor

import testing.example as aix
from ai_explainer.common.canon import Canon
from testing.example.world import ExampleData, ExampleHyperparams, ExampleModel, ExampleSnapshot


class ExampleCanon(Canon[ExampleModel, Tensor, ExampleData, ExampleHyperparams]):
    """Canon for the test example."""

    def __init__(
        self, *, data: ExampleData, sample: tuple[Tensor, Tensor], hparams: ExampleHyperparams
    ) -> None:
        """Create the canon."""
        super().__init__(data, sample, hparams)

    @override
    def create_model(self) -> ExampleModel:
        return ExampleModel()

    @override
    def _train(self, model: ExampleModel) -> list[ExampleSnapshot]:
        return aix.world.train(model, self.hparams, self.data.x, self.data.y)


@cache
def get() -> ExampleCanon:
    """Get the canon for the test example."""
    data = aix.world.load_data()
    return ExampleCanon(
        data=data,
        sample=(data.x[0], data.y[0]),
        hparams=ExampleHyperparams(lr=0.01, steps=3),
    )
