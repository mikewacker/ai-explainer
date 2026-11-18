import atexit
from collections.abc import Callable
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_explainer.iris.asset.binary._drawio import slope

_DRAWIO_DIR = Path(__file__).parent / "drawio"


@cache
def drawio(name: str) -> Path:
    """Get the path for a draw.io diagram."""
    match name:
        case "slope_output":
            return _edit_drawio("model_output", name, slope)
    return _DRAWIO_DIR / f"{name}.drawio"


def _edit_drawio(in_name: str, out_name: str, edit_fn: Callable[[str], str]) -> Path:
    in_path = drawio(in_name)
    out_path = _tmp_dir() / out_name
    in_text = in_path.read_text()
    out_text = edit_fn(in_text)
    out_path.write_text(out_text)
    return out_path


@cache
def _tmp_dir() -> Path:
    tmp_dir = TemporaryDirectory(prefix="ai_explainer-iris-multiclass-drawio-")
    atexit.register(tmp_dir.cleanup)
    return Path(tmp_dir.name)
