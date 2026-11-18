import atexit
import re
from collections.abc import Callable
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

_DRAWIO_DIR = Path(__file__).parent / "drawio"


@cache
def drawio(name: str) -> Path:
    """Get the path for a draw.io diagram."""
    match name:
        case "model_blackbox":
            return _edit_drawio("model_hand", name, _model_blackbox)
        case "loss_alt":
            return _edit_drawio("loss_data", name, _loss_alt)
    return _DRAWIO_DIR / f"{name}.drawio"


def _model_blackbox(text: str) -> str:
    return text.replace("fillColor=#f5f5f5", "fillColor=black")


def _loss_alt(text: str) -> str:
    pattern = re.compile(r'(id="x31"[^>]+?\[\[)data_color')
    return pattern.sub(r"\1alt_color", text)


def _edit_drawio(in_name: str, out_name: str, edit_fn: Callable[[str], str]) -> Path:
    in_path = drawio(in_name)
    out_path = _tmp_dir() / out_name
    in_text = in_path.read_text()
    out_text = edit_fn(in_text)
    out_path.write_text(out_text)
    return out_path


@cache
def _tmp_dir() -> Path:
    tmp_dir = TemporaryDirectory(prefix="ai_explainer-wine-drawio-")
    atexit.register(tmp_dir.cleanup)
    return Path(tmp_dir.name)
