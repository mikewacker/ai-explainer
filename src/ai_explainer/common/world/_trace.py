from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterator
from contextlib import ExitStack, contextmanager

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle


class Trace[M: nn.Module](ABC):
    """Trace through a model, forward and backward."""

    @contextmanager
    def capture(self, model: M, criterion: nn.Module) -> Generator[None, None, None]:
        """Return a context manager that captures tensors for the trace."""
        with ExitStack() as hook_stack:
            for hook in self._register_hooks(model, criterion):
                hook_stack.callback(hook.remove)
            yield

    @abstractmethod
    def _register_hooks(self, model: M, criterion: nn.Module) -> Iterator[RemovableHandle]:
        """Register hooks that populate this trace."""

    @staticmethod
    def _register_out_hook(module: nn.Module, hook: Callable[[Tensor], None]) -> RemovableHandle:
        """Register a hook to capture the output for the module."""

        def capture_out(_, __, out: Tensor) -> None:
            out = out.detach().clone()
            hook(out)

        return module.register_forward_hook(capture_out)

    @staticmethod
    def _register_in_grad_hook(
        module: nn.Module, hook: Callable[[Tensor], None]
    ) -> RemovableHandle:
        """Register a hook to capture the input gradient for the module."""

        def capture_in_grad(_, in_grads: tuple[Tensor, ...] | Tensor, __) -> None:
            in_grad = in_grads[0] if isinstance(in_grads, tuple) else in_grads
            in_grad = in_grad.detach().clone()
            hook(in_grad)

        return module.register_full_backward_hook(capture_in_grad)

    @staticmethod
    def _register_param_grad_hook(param: Tensor, hook: Callable[[Tensor], None]) -> RemovableHandle:
        """Register a hook to capture the parameter's gradient."""

        def capture_param_grad(grad: Tensor) -> None:
            grad = grad.detach().clone()
            hook(grad)

        return param.register_hook(capture_param_grad)


def trace[M: nn.Module, T: Trace](
    model: M,
    criterion: nn.Module,
    x: Tensor,
    y: Tensor,
    *,
    trace_factory: Callable[[M, Tensor, Tensor], T],
) -> T:
    """Trace a step through the model, forward and backward."""
    model.train()
    trace = trace_factory(model, x, y)
    with trace.capture(model, criterion):
        model.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
    return trace
