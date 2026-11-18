"""Tools for organizing hand-crafted assets."""

from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path

from ._cmd import run


class Outputs:
    """Paths for output files."""

    _dir: Path
    _act: str
    _sequence: str
    _idx: int

    def __init__(self, root_dir: Path) -> None:
        """Create the outputs from a root directory."""
        self._dir = root_dir
        self._act = ""
        self._sequence = ""
        self._idx = 1

    @property
    def dir(self) -> Path:
        """Get the directory for the outputs."""
        return self._dir

    def __repr__(self) -> str:
        """Get the string representation."""
        return repr(self._dir)

    def act(self, act: str) -> Outputs:
        """Get the outputs for the act."""
        if self._act:
            raise ValueError("act is already set")

        outputs = Outputs(self._dir)
        act_parts = act.split("-")
        outputs._dir = Path.joinpath(self._dir, *act_parts)
        outputs._act = act
        return outputs

    def sequence(self, sequence: str) -> Outputs:
        """Get the outputs for the sequence."""
        if not self._act:
            raise ValueError("act is not set")

        if self._sequence:
            raise ValueError("sequence is already set")

        outputs = Outputs(self._dir)
        outputs._act = self._act
        outputs._sequence = sequence
        return outputs

    def mkdir(self) -> None:
        """Create the directory for the outputs, if it does not exist."""
        self.dir.mkdir(parents=True, exist_ok=True)

    def png(self, name: str) -> Path:
        """Get the path for a PNG file."""
        return self._path(name, ext="png")

    def mp4(self, name: str) -> Path:
        """Get the path for an MP4 file."""
        return self._path(name, ext="mp4")

    def txt(self, name: str) -> Path:
        """Get the path for a text file."""
        return self._path(name, ext="txt")

    def pth(self, name: str) -> Path:
        """Get the path for a PyTorch file."""
        return self._path(name, ext="pth")

    def skip(self) -> None:
        """Skip a file in the current sequence."""
        if not self._sequence:
            raise ValueError("sequence is not set")

        self._idx += 1

    def _path(self, name: str, *, ext: str) -> Path:
        if not self._act:
            raise ValueError("act is not set")

        filename = self._act
        if self._sequence:
            filename = f"{filename}_{self._sequence}{self._idx:02d}"
            self._idx += 1
        filename = f"{filename}_{name}.{ext}"
        return self._dir / filename


def _chown_out() -> None:
    try:
        out = Path("out")
        if not out.exists():
            return

        wd_stat = Path.cwd().stat()
        run("chown", ["-R", f"{wd_stat.st_uid}:{wd_stat.st_gid}", str(out)])
    except Exception as e:
        logging.warning("[atexit] chown out failed: %s", e, exc_info=True)


if os.geteuid() == 0:
    atexit.register(_chown_out)
