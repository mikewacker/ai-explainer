import shutil
import subprocess
from functools import cache


def run(cmd: str, args: list[str]) -> None:
    cmd = _which(cmd)
    try:
        subprocess.run([cmd, *args], check=True, capture_output=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        if e.stdout:
            stdout = e.stdout.decode(errors="replace")
            e.add_note(f"stdout:\n{stdout}")
        if e.stderr:
            stderr = e.stderr.decode(errors="replace")
            e.add_note(f"stderr:\n{stderr}")
        raise


@cache
def _which(cmd: str) -> str:
    result = shutil.which(cmd)
    if result is None:
        raise FileNotFoundError(cmd)
    return result
