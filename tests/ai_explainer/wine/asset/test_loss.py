import pytest
from torch import Tensor

import ai_explainer.wine as aix
import testing
from ai_explainer.wine.asset import Diagram, LossCrossFn, Outputs, drawio
from ai_explainer.wine.world import WineData, WineTrace


@pytest.mark.parametrize(
    "name",
    [
        "loss_sample",
        "loss_data",
        "loss_alt",
    ],
)
def test_drawio(name: str, outputs: Outputs, goldens: Outputs) -> None:
    diag = Diagram(drawio(name))

    testing.assert_matches_golden(outputs, goldens, diag, f"loss_drawio_{name}")


def test_plot_loss_fn(loss_mesh: Tensor, outputs: Outputs, goldens: Outputs) -> None:
    fig, ax = aix.asset.figure(is_3d=True)
    aix.asset.plot_loss_fn(ax, loss_mesh)

    testing.assert_matches_golden(outputs, goldens, fig, "loss_plot_loss_fn")


def test_plot_loss_contours(loss_mesh: Tensor, outputs: Outputs, goldens: Outputs) -> None:
    fig, ax = aix.asset.figure()
    aix.asset.plot_loss_contours(ax, loss_mesh)

    testing.assert_matches_golden(outputs, goldens, fig, "loss_plot_loss_contours")


def test_plot_loss_fn_w1(
    loss_mesh: Tensor, loss_fn_w1: LossCrossFn[Tensor], outputs: Outputs, goldens: Outputs
) -> None:
    fig, ax = aix.asset.figure(is_3d=True)
    aix.asset.plot_loss_fn(ax, loss_mesh, show_labels=["w1", "L"])
    aix.asset.plot_loss_fn_w1(ax, loss_fn_w1)

    testing.assert_matches_golden(outputs, goldens, fig, "loss_plot_loss_fn_w1")


def test_plot_loss_fn_w2(
    loss_mesh: Tensor, loss_fn_w2: LossCrossFn[Tensor], outputs: Outputs, goldens: Outputs
) -> None:
    fig, ax = aix.asset.figure(is_3d=True)
    aix.asset.plot_loss_fn(ax, loss_mesh, show_labels=["w2", "L"])
    aix.asset.plot_loss_fn_w2(ax, loss_fn_w2)

    testing.assert_matches_golden(outputs, goldens, fig, "loss_plot_loss_fn_w2")


def test_table_training_data(data: WineData, outputs: Outputs, goldens: Outputs) -> None:
    training_data = aix.asset.table_training_data(data.x, data.y)

    testing.assert_matches_golden(outputs, goldens, training_data, "loss_table_training_data")


def test_latex_se(trace_sample: WineTrace, outputs: Outputs, goldens: Outputs) -> None:
    e_steps, se_steps = aix.asset.latex_se(trace_sample)
    e = aix.asset.equation(e_steps)
    se = aix.asset.equation(se_steps, approx_steps=1)

    testing.assert_matches_golden(outputs, goldens, [e, se], "loss_latex_se")
