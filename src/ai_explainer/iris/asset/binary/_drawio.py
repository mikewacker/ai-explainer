import atexit
import re
from collections.abc import Callable
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_explainer.iris.asset.binary import Diagram, pd

_DRAWIO_DIR = Path(__file__).parent / "drawio"


@cache
def drawio(name: str) -> Path:
    """Get the path for a draw.io diagram."""
    match name:
        case "model_hidden-no-layer":
            return _edit_drawio("model_hidden", name, _model_hidden_no_layer)
        case "model_hidden1":
            return _edit_drawio("model_hidden-a1", name, _model_hidden1)
        case "loss_sample-neurons":
            return _edit_drawio("loss_sample", name, _loss_sample_neurons)
        case "loss_sample-matrix":
            return _edit_drawio("loss_data", name, _loss_sample_matrix)
        case "slope_output":
            return _edit_drawio("model_output", name, slope)
        case "slope_hidden-a1":
            return _edit_drawio("model_hidden-a1", name, slope)
        case "slope_hidden":
            return _edit_drawio("model_hidden", name, slope)
    return _DRAWIO_DIR / f"{name}.drawio"


def slope(text: str) -> str:
    def to_pd(match: re.Match) -> str:
        name = match.group(1)
        value = match.group(2)
        if "color" in name:
            return match.group(0)

        value = pd("L", value, t=True)
        repl = f"[[{name}:{value}]]"
        if name[0] in ["w", "b"]:
            repl = rf"\textcolor{{firebrick}}{{{repl}}}"
        return repl

    return Diagram.PLACEHOLDER_PATTERN.sub(to_pd, text)


def _model_hidden_no_layer(text: str) -> str:
    return text.replace("^{(1)}", "")


def _model_hidden1(text: str) -> str:
    text = text.replace("^{(1)}_1", "")  # b and z
    text = text.replace("a_1", "a")  # a
    return re.sub(r"\^\{\(1\)}_\{1(\d)}", r"_\1", text)  # w's


def _loss_sample_neurons(text: str) -> str:
    text = re.sub(
        r'(id="params1"[^>]+?\\end\{bmatrix\*})[^$]+',
        r"\1 \\quad \\begin{matrix} [[b15:b^{(1)}_5]] \\end{matrix}",
        text,
    )
    return re.sub(
        r"(w\^\{\(1\)}_\{([1-4])4}]])\s+\\\\",
        r"\1 \\end{bmatrix*} \\quad "
        r"\\begin{matrix} [[b1\2:b^{(1)}_\2]] \\end{matrix} \\\\ \\begin{bmatrix*}[r]",
        text,
    )


def _loss_sample_matrix(text: str) -> str:
    text = text.replace(r"\mathbf{X}", r"\mathbf{x}")
    text = text.replace(r"\hat{\mathbf{y}}", r"\hat{y}")
    return text.replace(r"\mathbf{y}", "y")


def _edit_drawio(in_name: str, out_name: str, edit_fn: Callable[[str], str]) -> Path:
    in_path = drawio(in_name)
    out_path = _tmp_dir() / out_name
    in_text = in_path.read_text()
    out_text = edit_fn(in_text)
    out_path.write_text(out_text)
    return out_path


@cache
def _tmp_dir() -> Path:
    tmp_dir = TemporaryDirectory(prefix="ai_explainer-iris-binary-drawio-")
    atexit.register(tmp_dir.cleanup)
    return Path(tmp_dir.name)
