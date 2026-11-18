import re

import pytest
import torch

import ai_explainer.wine as aix
import testing
from ai_explainer.wine.world import WineData, WineHyperparams, WineModel


class TestWineModel:
    def test_params(self, hparams: WineHyperparams) -> None:
        model = WineModel(w=hparams.w_init)

        testing.assert_tensors_equal(model.params(), hparams.w_init)
        assert model.linear.bias is None

    def test_params__copy(self, hparams: WineHyperparams) -> None:
        model = WineModel(w=hparams.w_init)
        hparams.w_init[...] += 1
        params1 = model.params()
        params2 = model.params()
        params2[...] += 1

        testing.assert_tensors_not_equal(params1, hparams.w_init)
        testing.assert_tensors_not_equal(params1, params2)
        assert not params1.requires_grad

    def test_forward(self, data: WineData, hparams: WineHyperparams) -> None:
        model = WineModel(w=hparams.w_init)
        y_hat = model(data.x)

        assert y_hat.shape == (3,)

    def test_forward__sample(self, data: WineData, hparams: WineHyperparams) -> None:
        model = WineModel(w=hparams.w_init)
        y_hat = model(data.x[0])

        assert y_hat.shape == ()

    def test_error__invalid_w_shape(self) -> None:
        with pytest.raises(ValueError, match=re.escape("w.shape must be (2,): ()")):
            WineModel(w=torch.tensor(1.0))

    @pytest.mark.parametrize("shape", [(8, 4), (4,), (1, 8, 2), ()])
    def test_error__invalid_x_shape(self, hparams: WineHyperparams, shape: tuple[int, ...]) -> None:
        model = WineModel(w=hparams.w_init)
        x = torch.rand(shape)

        with pytest.raises(ValueError, match=re.escape(f"x.shape must be (m, 2) or (2,): {shape}")):
            model(x)


def test_train(data: WineData, hparams: WineHyperparams) -> None:
    model = WineModel(w=hparams.w_init)
    history = aix.world.train(model, hparams, data.x, data.y)

    assert history[-1].loss < history[0].loss
    assert len(history) == 4
