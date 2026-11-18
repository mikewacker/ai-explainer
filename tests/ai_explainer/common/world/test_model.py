import testing
import testing.example as aix
from testing.example.world import ExampleData, ExampleHyperparams, ExampleModel


def test_train(data: ExampleData, hparams: ExampleHyperparams) -> None:
    model = ExampleModel()
    model.eval()  # train() should call model.train()
    history = aix.world.train(model, hparams, data.x, data.y)

    assert history[-1].loss < history[0].loss
    assert not history[0].loss.requires_grad
    assert not history[-1].loss.requires_grad
    testing.assert_tensors_not_equal(history[-1].params, history[0].params)
    assert len(history) == 4
    assert model.training


def test_test(data: ExampleData) -> None:
    model = ExampleModel()
    model.train()  # test() should call model.eval()
    metric = aix.world.test(model, data.x, data.y)

    assert metric.shape == ()
    assert not metric.requires_grad
    assert not model.training
