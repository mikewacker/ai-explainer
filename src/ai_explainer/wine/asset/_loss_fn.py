from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor, nn

import ai_explainer.wine as aix
from ai_explainer.wine.asset import LossCrossFn
from ai_explainer.wine.world import WineModel


def loss_mesh(x: Tensor, y: Tensor, *, loss_delta: float) -> Tensor:
    """Sample points to plot the loss function using a contour mesh."""
    # Create the contour mesh without translation.
    ab_e, e = _loss_eigen(x)
    w_e, loss = _elliptical_mesh(ab_e, loss_delta)
    w = torch.inner(w_e, e)

    # Add translation.
    w_min, loss_min = _lstsq(x, y)
    w += w_min
    loss += loss_min
    return torch.cat([w, loss.unsqueeze(-1)], dim=-1)


def loss_cross_fn(
    x: Tensor, y: Tensor, model_factory: Callable[[], WineModel], *, j: Literal[1, 2]
) -> LossCrossFn[Tensor]:
    """Create a 2D cross-section of the loss function for one parameter."""
    criterion = aix.world.get_loss_criterion()

    def set_param(model: WineModel, w: Tensor) -> None:
        mask = torch.ones_like(model.linear.weight, dtype=torch.bool)
        mask[0, j - 1] = False
        model.linear.weight = nn.Parameter(torch.where(mask, model.linear.weight, w))

    return LossCrossFn(x, y, model_factory, criterion, param_setter=set_param)


def _loss_eigen(x: Tensor) -> tuple[Tensor, Tensor]:
    q = torch.inner(x.T, x.T)
    ab_e, e = torch.linalg.eigh(q)
    ab_e /= x.shape[0]
    return ab_e, e


def _elliptical_mesh(ab: Tensor, z_max: float) -> tuple[Tensor, Tensor]:
    """Create an elliptical mesh for the equation `z = a x^2 + b y^2`."""
    theta_space = torch.linspace(0, 2 * torch.pi, 145)
    z_space = torch.linspace(0, z_max, 101)
    theta, z = torch.meshgrid(theta_space, z_space, indexing="xy")
    semi_axes = _sqrt(z.unsqueeze(-1) / ab)
    unit_circles = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    xy = semi_axes * unit_circles
    return xy, z.clone()


def _lstsq(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:  # for reproducibility
    x, y = x.double(), y.double()
    w, res, _, _ = torch.linalg.lstsq(x, y, driver="gels")
    loss = res.squeeze() / x.shape[0]
    return w.float(), loss.float()


def _sqrt(t: Tensor) -> Tensor:  # for reproducibility
    t = t.double()
    result = torch.sqrt(t)
    return result.float()
