import re

import pytest
import torch

import ai_explainer.iris as aix
import testing
from ai_explainer.iris.world import IrisData, IrisHyperparams, IrisNn


class TestIrisNn:
    def test_shape(self, hparams: IrisHyperparams) -> None:
        model = IrisNn(hidden_size=hparams.hidden_size)

        assert model.task == "binary"
        assert model.in_size == 4
        assert model.hidden_size == 5
        assert model.out_size == 1

    def test_params(self, hparams: IrisHyperparams) -> None:
        model = IrisNn(hidden_size=hparams.hidden_size)
        params = model.params()

        assert params.w1.shape == (5, 4)
        assert params.b1.shape == (5,)
        assert params.w2.shape == (5,)
        assert params.b2.shape == ()

    def test_params__copy(self, hparams: IrisHyperparams) -> None:
        model = IrisNn(hidden_size=hparams.hidden_size)
        params1 = model.params()
        params2 = model.params()
        for w in [params2.w1, params2.b1, params2.w2, params2.b2]:
            w += 1

        testing.assert_tensors_not_equal(params1.w1, params2.w1)
        testing.assert_tensors_not_equal(params1.b1, params2.b1)
        testing.assert_tensors_not_equal(params1.w2, params2.w2)
        testing.assert_tensors_not_equal(params1.b2, params2.b2)
        assert not params1.w1.requires_grad
        assert not params1.b1.requires_grad
        assert not params1.w2.requires_grad
        assert not params1.b2.requires_grad

    def test_params__seed(self, hparams: IrisHyperparams) -> None:
        model1 = IrisNn(hidden_size=hparams.hidden_size, seed=hparams.seed)
        model2 = IrisNn(hidden_size=hparams.hidden_size, seed=hparams.seed)

        testing.assert_tensors_equal(model1.params().w1, model2.params().w1)

    def test_forward(self, data: IrisData, hparams: IrisHyperparams) -> None:
        model = IrisNn(hidden_size=hparams.hidden_size)
        logits = model(data.x_train)

        assert logits.shape == (8,)

    def test_forward__sample(self, data: IrisData, hparams: IrisHyperparams) -> None:
        model = IrisNn(hidden_size=hparams.hidden_size)
        logits = model(data.x_train[0])

        assert logits.shape == ()

    def test_shape__multiclass(self, hparams: IrisHyperparams) -> None:
        model = IrisNn(task="multiclass", hidden_size=hparams.hidden_size)

        assert model.task == "multiclass"
        assert model.in_size == 4
        assert model.hidden_size == 5
        assert model.out_size == 3

    def test_params__multiclass(self, hparams: IrisHyperparams) -> None:
        model = IrisNn(task="multiclass", hidden_size=hparams.hidden_size)
        params = model.params()

        assert params.w1.shape == (5, 4)
        assert params.b1.shape == (5,)
        assert params.w2.shape == (3, 5)
        assert params.b2.shape == (3,)

    def test_forward__multiclass(self, data_multiclass: IrisData, hparams: IrisHyperparams) -> None:
        model = IrisNn(task="multiclass", hidden_size=hparams.hidden_size)
        logits = model(data_multiclass.x_train)

        assert logits.shape == (8, 3)

    def test_forward__multiclass_sample(
        self, data_multiclass: IrisData, hparams: IrisHyperparams
    ) -> None:
        model = IrisNn(task="multiclass", hidden_size=hparams.hidden_size)
        logits = model(data_multiclass.x_train[0])

        assert logits.shape == (3,)

    @pytest.mark.parametrize("shape", [(8, 2), (2,), (1, 8, 4), ()])
    def test_error__invalid_x_shape(self, hparams: IrisHyperparams, shape: tuple[int, ...]) -> None:
        model = IrisNn(hidden_size=hparams.hidden_size)
        x = torch.rand(shape)

        with pytest.raises(ValueError, match=re.escape(f"x.shape must be (m, 4) or (4,): {shape}")):
            model(x)


def test_train(data: IrisData, hparams: IrisHyperparams) -> None:
    model = IrisNn(hidden_size=hparams.hidden_size, seed=hparams.seed)
    history = aix.world.train(model, hparams, data.x_train, data.y_train)

    assert history[-1].loss < history[0].loss
    assert len(history) == 4


def test_test(data: IrisData, hparams: IrisHyperparams) -> None:
    model = IrisNn(hidden_size=hparams.hidden_size, seed=hparams.seed)
    accuracy = aix.world.test(model, data.x_test, data.y_test)

    assert 0.0 <= accuracy <= 1.0


def test_train__multiclass(data_multiclass: IrisData, hparams: IrisHyperparams) -> None:
    model = IrisNn(task="multiclass", hidden_size=hparams.hidden_size, seed=hparams.seed)
    history = aix.world.train(model, hparams, data_multiclass.x_train, data_multiclass.y_train)

    assert history[-1].loss < history[0].loss
    assert len(history) == 4


def test_test__multiclass(data_multiclass: IrisData, hparams: IrisHyperparams) -> None:
    model = IrisNn(task="multiclass", hidden_size=hparams.hidden_size, seed=hparams.seed)
    accuracy = aix.world.test(model, data_multiclass.x_test, data_multiclass.y_test)

    assert 0.0 <= accuracy <= 1.0
