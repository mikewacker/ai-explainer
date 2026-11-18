from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn

from ai_explainer.common.world import Model


class LossCrossFn[P]:
    """2D cross-section of the loss function for one parameter."""

    _params: P
    _flat_fn: Callable[[Tensor], Tensor]

    def __init__[M: Model](  # M: Model[P]
        self,
        x: Tensor,
        y: Tensor,
        model_factory: Callable[[], Model[P]],
        criterion: nn.Module,
        *,
        param_setter: Callable[[M, Tensor], None],
    ) -> None:
        """Create a 2D cross-section of the loss function for one parameter.

        model_factory and param_setter must be compatible with torch.vmap().
        """
        self._params = model_factory().params()

        def call_scalar(w: Tensor) -> Tensor:
            model = cast(M, model_factory())
            param_setter(model, w)
            model.train()
            with torch.no_grad():
                y_hat = model(x)
                return criterion(y_hat, y)

        self._flat_fn = torch.vmap(call_scalar, randomness="same")

    @property
    def params(self) -> P:
        """Get the fixed values of all the parameters."""
        return self._params

    def __call__(self, w: Tensor) -> Tensor:
        """Evaluate this function against different values of the changeable parameter."""
        w_flat = w.reshape(-1)
        loss_flat = self._flat_fn(w_flat)
        return loss_flat.reshape(w.shape)
