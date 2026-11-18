import re

import pytest
import torch

import ai_explainer.number as aix
import testing
from ai_explainer.number.world import NumberData, NumberHyperparams, NumberModel


class TestNumberModel:
    def test_params(self, hparams: NumberHyperparams) -> None:
        model = NumberModel(w=hparams.w_init)

        testing.assert_tensors_equal(model.params(), hparams.w_init)

    def test_params__copy(self, hparams: NumberHyperparams) -> None:
        model = NumberModel(w=hparams.w_init)
        hparams.w_init[...] += 1
        params1 = model.params()
        params2 = model.params()
        params2[...] += 1

        testing.assert_tensors_not_equal(params1, hparams.w_init)
        testing.assert_tensors_not_equal(params1, params2)
        assert not params1.requires_grad

    def test_forward(self, hparams: NumberHyperparams) -> None:
        model = NumberModel(w=hparams.w_init)
        yhat = model(torch.empty(0))

        testing.assert_tensors_equal(yhat, hparams.w_init)

    def test_error__invalid_w_shape(self) -> None:
        with pytest.raises(ValueError, match=re.escape("w.shape must be (): (1,)")):
            NumberModel(w=torch.tensor([5.0]))


def test_train(data: NumberData, hparams: NumberHyperparams) -> None:
    model = NumberModel(w=hparams.w_init)
    history = aix.world.train(model, hparams, data.y)

    assert history[-1].loss < history[0].loss
    assert len(history) == 4
