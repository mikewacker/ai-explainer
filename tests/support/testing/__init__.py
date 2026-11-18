import pytest

pytest.register_assert_rewrite("testing._golden", "testing._tensor")

# ruff: noqa: E402  # register_assert_rewrite() must be called before the import
from ._ffmpeg import ffprobe_video_stream
from ._golden import assert_matches_golden, asset_parameters
from ._tensor import (
    assert_tensors_close,
    assert_tensors_equal,
    assert_tensors_not_equal,
)

__all__ = [
    "assert_matches_golden",
    "assert_tensors_close",
    "assert_tensors_equal",
    "assert_tensors_not_equal",
    "asset_parameters",
    "ffprobe_video_stream",
]
