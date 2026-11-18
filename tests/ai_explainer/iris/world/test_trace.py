import torch
from torch import nn

import ai_explainer.iris as aix
import testing
from ai_explainer.iris.world import IrisData, IrisHyperparams, IrisNn


def test_trace(data: IrisData, hparams: IrisHyperparams) -> None:
    model = IrisNn(hidden_size=hparams.hidden_size)
    trace = aix.world.trace(model, data.x_train, data.y_train)

    assert trace.w1.shape == (5, 4)
    assert trace.b1.shape == (5,)
    assert trace.w2.shape == (5,)
    assert trace.b2.shape == ()
    assert trace.x.shape == (8, 4)
    assert trace.z1.shape == (8, 5)
    assert trace.a.shape == (8, 5)
    assert trace.z2.shape == (8,)
    assert trace.y_hat.shape == (8,)
    assert trace.y.shape == (8,)
    assert trace.loss.shape == ()
    assert trace.z2_grad.shape == (8,)
    assert trace.w2_grad.shape == (5,)
    assert trace.b2_grad.shape == ()
    assert trace.a_grad.shape == (8, 5)
    assert trace.z1_grad.shape == (8, 5)
    assert trace.w1_grad.shape == (5, 4)
    assert trace.b1_grad.shape == (5,)
    assert trace.x is not data.x_train
    assert trace.y is not data.y_train
    assert not trace.y_hat.requires_grad


def test_trace__sample(data: IrisData, hparams: IrisHyperparams) -> None:
    model = IrisNn(hidden_size=hparams.hidden_size)
    trace = aix.world.trace(model, data.x_train[0], data.y_train[0])

    assert trace.x.shape == (4,)
    assert trace.z1.shape == (5,)
    assert trace.a.shape == (5,)
    assert trace.z2.shape == ()
    assert trace.y_hat.shape == ()
    assert trace.y.shape == ()
    assert trace.loss.shape == ()
    assert trace.z2_grad.shape == ()
    assert trace.w2_grad.shape == (5,)
    assert trace.b2_grad.shape == ()
    assert trace.a_grad.shape == (5,)
    assert trace.z1_grad.shape == (5,)
    assert trace.w1_grad.shape == (5, 4)
    assert trace.b1_grad.shape == (5,)


def test_trace__steps(data: IrisData, hparams: IrisHyperparams) -> None:
    model = IrisNn(hidden_size=hparams.hidden_size, seed=hparams.seed)
    m = data.x_train.shape[0]
    trace = aix.world.trace(model, data.x_train, data.y_train)

    testing.assert_tensors_close(trace.z1, torch.inner(trace.x, trace.w1) + trace.b1)
    testing.assert_tensors_close(trace.a, torch.relu(trace.z1))
    testing.assert_tensors_close(trace.z2, torch.inner(trace.a, trace.w2) + trace.b2)
    testing.assert_tensors_close(trace.y_hat, torch.sigmoid(trace.z2))
    testing.assert_tensors_close(
        trace.loss, nn.functional.binary_cross_entropy_with_logits(trace.z2, trace.y)
    )
    testing.assert_tensors_close(trace.z2_grad, (trace.y_hat - data.y_train) / m)
    testing.assert_tensors_close(trace.w2_grad, torch.inner(trace.z2_grad, trace.a.T))
    testing.assert_tensors_close(trace.b2_grad, torch.inner(trace.z2_grad, torch.ones(m)))
    testing.assert_tensors_close(trace.a_grad, torch.outer(trace.z2_grad, trace.w2))
    testing.assert_tensors_close(trace.z1_grad, trace.a_grad * (trace.z1 > 0))
    testing.assert_tensors_close(trace.w1_grad, torch.inner(trace.z1_grad.T, trace.x.T))
    testing.assert_tensors_close(trace.b1_grad, torch.inner(trace.z1_grad.T, torch.ones(m)))


def test_trace__multiclass(data_multiclass: IrisData, hparams: IrisHyperparams) -> None:
    model = IrisNn(task="multiclass", hidden_size=hparams.hidden_size)
    trace = aix.world.trace(model, data_multiclass.x_train, data_multiclass.y_train)

    assert trace.w1.shape == (5, 4)
    assert trace.b1.shape == (5,)
    assert trace.w2.shape == (3, 5)
    assert trace.b2.shape == (3,)
    assert trace.x.shape == (8, 4)
    assert trace.z1.shape == (8, 5)
    assert trace.a.shape == (8, 5)
    assert trace.z2.shape == (8, 3)
    assert trace.y_hat.shape == (8, 3)
    assert trace.y.shape == (8, 3)
    assert trace.loss.shape == ()
    assert trace.z2_grad.shape == (8, 3)
    assert trace.w2_grad.shape == (3, 5)
    assert trace.b2_grad.shape == (3,)
    assert trace.a_grad.shape == (8, 5)
    assert trace.z1_grad.shape == (8, 5)
    assert trace.w1_grad.shape == (5, 4)
    assert trace.b1_grad.shape == (5,)
    assert trace.y.dtype == torch.float32


def test_trace__multiclass_sample(data_multiclass: IrisData, hparams: IrisHyperparams) -> None:
    model = IrisNn(task="multiclass", hidden_size=hparams.hidden_size)
    trace = aix.world.trace(model, data_multiclass.x_train[0], data_multiclass.y_train[0])

    assert trace.x.shape == (4,)
    assert trace.z1.shape == (5,)
    assert trace.a.shape == (5,)
    assert trace.z2.shape == (3,)
    assert trace.y_hat.shape == (3,)
    assert trace.y.shape == (3,)
    assert trace.loss.shape == ()
    assert trace.z2_grad.shape == (3,)
    assert trace.w2_grad.shape == (3, 5)
    assert trace.b2_grad.shape == (3,)
    assert trace.a_grad.shape == (5,)
    assert trace.z1_grad.shape == (5,)
    assert trace.w1_grad.shape == (5, 4)
    assert trace.b1_grad.shape == (5,)


def test_trace__multiclass_steps(data_multiclass: IrisData, hparams: IrisHyperparams) -> None:
    model = IrisNn(task="multiclass", hidden_size=hparams.hidden_size, seed=hparams.seed)
    m = data_multiclass.x_train.shape[0]
    trace = aix.world.trace(model, data_multiclass.x_train, data_multiclass.y_train)

    testing.assert_tensors_close(trace.z2, torch.inner(trace.a, trace.w2) + trace.b2)
    testing.assert_tensors_close(trace.y_hat, torch.softmax(trace.z2, dim=-1))
    testing.assert_tensors_close(trace.loss, nn.functional.cross_entropy(trace.z2, trace.y))
    testing.assert_tensors_close(trace.z2_grad, (trace.y_hat - trace.y) / m)
    testing.assert_tensors_close(trace.w2_grad, torch.inner(trace.z2_grad.T, trace.a.T))
    testing.assert_tensors_close(trace.b2_grad, torch.inner(trace.z2_grad.T, torch.ones(m)))
    testing.assert_tensors_close(trace.a_grad, torch.inner(trace.z2_grad, trace.w2.T))
