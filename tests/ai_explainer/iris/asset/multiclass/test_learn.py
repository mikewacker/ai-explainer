import pytest
import torch

import ai_explainer.iris as aix
import testing
from ai_explainer.iris.asset.multiclass import IrisVisualSnapshot, Outputs
from ai_explainer.iris.world import IrisNn, IrisSnapshot


@pytest.fixture
def snapshot1(model: IrisNn) -> IrisSnapshot:
    return IrisSnapshot(torch.tensor(1.234), model.params())


@pytest.fixture
def snapshot2(model: IrisNn) -> IrisSnapshot:
    snapshot = IrisSnapshot(torch.tensor(1.134), model.params())
    for param in [snapshot.params.w1, snapshot.params.b1, snapshot.params.w2, snapshot.params.b2]:
        param.add_(0.25)
    return snapshot


class TestIrisVisualSnapshot:
    def test_plot(self, snapshot1: IrisSnapshot, outputs: Outputs, goldens: Outputs) -> None:
        fig = aix.asset.multiclass.figure_empty()
        IrisVisualSnapshot(fig, (-1, 1), snapshot1)

        testing.assert_matches_golden(outputs, goldens, fig, "learn_plot_visual_snapshot")

    def test_plot__show_values(
        self, snapshot1: IrisSnapshot, outputs: Outputs, goldens: Outputs
    ) -> None:
        fig = aix.asset.multiclass.figure_empty()
        IrisVisualSnapshot(fig, (-1, 1), snapshot1, show_values=True)

        testing.assert_matches_golden(
            outputs, goldens, fig, "learn_plot_visual_snapshot-show_values"
        )

    def test_animate(
        self, snapshot1: IrisSnapshot, snapshot2: IrisSnapshot, outputs: Outputs, goldens: Outputs
    ) -> None:
        fig = aix.asset.multiclass.figure_empty()
        vsnapshot = IrisVisualSnapshot(fig, (-1, 1), snapshot1)
        artists = vsnapshot.animate(snapshot2)

        testing.assert_matches_golden(outputs, goldens, fig, "learn_animate_visual_snapshot")
        assert len(artists) == 44
