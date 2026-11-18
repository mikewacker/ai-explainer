"""Canon for the iris example."""

from functools import cache, cached_property
from typing import override

from torch import Tensor

import ai_explainer.iris as aix
from ai_explainer.common.canon import Canon
from ai_explainer.iris.world import (
    IrisData,
    IrisHyperparams,
    IrisNn,
    IrisParams,
    IrisSnapshot,
    IrisSpecies,
    IrisTask,
)


class IrisCanon(Canon[IrisNn, IrisParams, IrisData, IrisHyperparams]):
    """Canon for the iris example."""

    _task: IrisTask

    def __init__(
        self,
        *,
        task: IrisTask,
        data: IrisData,
        sample: tuple[Tensor, Tensor],
        hparams: IrisHyperparams,
    ) -> None:
        """Create the canon."""
        super().__init__(data, sample, hparams)
        self._task = task

    @property
    def task(self) -> IrisTask:
        """Binary or multiclass classification task."""
        return self._task

    @cached_property
    def accuracy_train(self) -> Tensor:
        """Accuracy of the trained canonical model against the training data."""
        return aix.world.test(self.model_trained, self.data.x_train, self.data.y_train)

    @cached_property
    def accuracy_test(self) -> Tensor:
        """Accuracy of the trained canonical model against the test data."""
        return aix.world.test(self.model_trained, self.data.x_test, self.data.y_test)

    @override
    def create_model(self) -> IrisNn:
        return IrisNn(task=self.task, hidden_size=self.hparams.hidden_size, seed=self.hparams.seed)

    @override
    def _train(self, model: IrisNn) -> list[IrisSnapshot]:
        return aix.world.train(model, self.hparams, self.data.x_train, self.data.y_train)


@cache
def get(*, task: IrisTask = "binary") -> IrisCanon:
    """Get the canon for the iris example."""
    variant = IrisSpecies.VIRGINICA if task == "binary" else "multiclass"
    data = aix.world.load_data(variant, seed=42)
    return IrisCanon(
        task=task,
        data=data,
        sample=(data.x_train[1], data.y_train[1]),
        hparams=IrisHyperparams(hidden_size=5, seed=42, lr=0.01, steps=3000),
    )
