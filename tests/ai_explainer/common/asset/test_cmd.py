from pathlib import Path
from subprocess import CalledProcessError

import pytest

from ai_explainer.common.asset._cmd import run


def test_run(tmp_path: Path) -> None:
    path = tmp_path / "test.txt"
    run("touch", [str(path)])

    assert path.exists()


def test_run__nonzero_exit_code() -> None:
    with pytest.raises(CalledProcessError) as e:
        run("sh", ["-c", "echo out; echo err >&2; exit 42"])
    assert e.value.returncode == 42
    assert e.value.stdout.decode() == "out\n"
    assert e.value.stderr.decode() == "err\n"
    assert e.value.__notes__ == ["stdout:\nout\n", "stderr:\nerr\n"]


def test_run__cmd_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        run("does-not-exist", [])
