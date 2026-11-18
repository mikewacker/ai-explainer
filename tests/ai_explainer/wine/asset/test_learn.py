import pytest
import torch
from torch import Tensor

import ai_explainer.wine as aix
import testing
from ai_explainer.wine.asset import Outputs, WineVisualSnapshot
from ai_explainer.wine.world import WineData, WineModel, WineSnapshot, WineTrace


@pytest.fixture
def snapshot1() -> WineSnapshot:
    return WineSnapshot(loss=torch.tensor(100.1), params=torch.tensor([2.0, 5.0]))


@pytest.fixture
def snapshot2() -> WineSnapshot:
    return WineSnapshot(loss=torch.tensor(90.1), params=torch.tensor([1.5, 4.5]))


@pytest.fixture
def model_trained() -> WineModel:
    return WineModel(w=torch.tensor([1.0, 3.0]))


class TestWineVisualSnapshot:
    def test_plot(
        self, loss_mesh: Tensor, snapshot1: WineSnapshot, outputs: Outputs, goldens: Outputs
    ) -> None:
        fig, ax = aix.asset.figure()
        WineVisualSnapshot(ax, loss_mesh, snapshot1)

        testing.assert_matches_golden(outputs, goldens, fig, "learn_plot_visual_snapshot")

    def test_plot__show_values(
        self, loss_mesh: Tensor, snapshot1: WineSnapshot, outputs: Outputs, goldens: Outputs
    ) -> None:
        fig, ax = aix.asset.figure()
        WineVisualSnapshot(ax, loss_mesh, snapshot1, show_values=True)

        testing.assert_matches_golden(
            outputs, goldens, fig, "learn_plot_visual_snapshot-show_values"
        )

    def test_animate(
        self,
        loss_mesh: Tensor,
        snapshot1: WineSnapshot,
        snapshot2: WineSnapshot,
        outputs: Outputs,
        goldens: Outputs,
    ) -> None:
        fig, ax = aix.asset.figure()
        vsnapshot = WineVisualSnapshot(ax, loss_mesh, snapshot1)
        artists = vsnapshot.animate(snapshot2)

        testing.assert_matches_golden(outputs, goldens, fig, "learn_animate_visual_snapshot")
        assert len(artists) == 2


def test_table_predictions(
    data: WineData, model: WineModel, model_trained: WineModel, outputs: Outputs, goldens: Outputs
) -> None:
    predictions = aix.asset.table_predictions(data, model, model_trained)

    testing.assert_matches_golden(outputs, goldens, predictions, "learn_table_predictions")


def test_latex_sgd_step(trace_data: WineTrace, outputs: Outputs, goldens: Outputs) -> None:
    grad_steps, delta_steps, update_steps = aix.asset.latex_sgd_step(trace_data, 0.01)
    grad = aix.asset.equation(grad_steps, approx_steps=1, multiline=True)
    delta = aix.asset.equation(delta_steps, approx_steps=2)
    update = aix.asset.equation(update_steps, approx_steps=2)

    testing.assert_matches_golden(outputs, goldens, [grad, delta, update], "learn_latex_sgd_step")
