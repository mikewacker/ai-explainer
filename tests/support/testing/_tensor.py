import torch
from torch import Tensor


def assert_tensors_equal(actual: Tensor, expected: Tensor) -> None:
    __tracebackhide__ = True
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def assert_tensors_not_equal(actual: Tensor, expected: Tensor) -> None:
    __tracebackhide__ = True
    assert not torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def assert_tensors_close(actual: Tensor, expected: Tensor, *, decimals: int | None = None) -> None:
    __tracebackhide__ = True
    if decimals is None:
        torch.testing.assert_close(actual, expected)
        return

    atol = 0.5 * 10**-decimals
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0.0)
