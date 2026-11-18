"""Canon for an example."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

from torch import Tensor

from ai_explainer.common.world import Snapshot


class Canon[M, P, D, H](ABC):
    """Canon for an example."""

    _data: D
    _sample: tuple[Tensor, Tensor]
    _hparams: H

    def __init__(self, data: D, sample: tuple[Tensor, Tensor], hparams: H) -> None:
        """Create the canon."""
        self._data = data
        self._sample = sample
        self._hparams = hparams

    @cached_property
    def model_trained(self) -> M:
        """Canonical model in a trained form."""
        return self._trained.model

    @cached_property
    def history(self) -> list[Snapshot[P]]:
        """Training history for the canonical model."""
        return self._trained.history

    @cached_property
    def model_untrained(self) -> M:
        """Canonical model in an untrained form."""
        return self.create_model()

    @property
    def data(self) -> D:
        """Training data and test data."""
        return self._data

    @property
    def sample(self) -> tuple[Tensor, Tensor]:
        """Canonical sample."""
        return self._sample

    @property
    def hparams(self) -> H:
        """Hyperparameters for the canonical model."""
        return self._hparams

    @abstractmethod
    def create_model(self) -> M:
        """Create the canonical model in an untrained form."""

    @abstractmethod
    def _train(self, model: M) -> list[Snapshot[P]]:
        """Train the model."""

    @cached_property
    def _trained(self) -> _Trained[M, P]:
        model = self.create_model()
        history = self._train(model)
        return _Trained(model, history)


@dataclass(frozen=True)
class _Trained[M, P]:
    model: M
    history: list[Snapshot[P]]
