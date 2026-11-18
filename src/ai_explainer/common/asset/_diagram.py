import html
import re
import tempfile
from pathlib import Path

from torch import Tensor

from ._cmd import run
from ._png import pdf_to_png, quantize_png


class Diagram:
    """draw.io diagram with placeholders to fill."""

    PLACEHOLDER_PATTERN = re.compile(r"\[\[(\w+?):(.+?)]]", flags=re.ASCII)

    _path: Path
    _values: dict[str, str]

    def __init__(self, path: Path) -> None:
        """Create a diagram from the .drawio file."""
        if not path.exists():
            raise FileNotFoundError(path)

        self._path = path
        self._values = {}

    @property
    def path(self) -> Path:
        """Get the path of the .drawio file."""
        return self._path

    def values(self) -> dict[str, str]:
        """Get a copy of the values for placeholders."""
        return self._values.copy()

    def fill_tensor(self, name: str, t: Tensor, *, decimals: int | None = None) -> None:
        """Fill the placeholders for a tensor with values."""
        spec = "g" if decimals is None else f".{decimals}f"
        match t.shape:
            case ():
                self._values |= {name: f"{t:{spec}}"}
            case (m,):
                self._values |= {f"{name}{i + 1}": f"{t[i]:{spec}}" for i in range(m)}
            case (m, n):
                self._values |= {
                    f"{name}{i + 1}{j + 1}": f"{t[i, j]:{spec}}" for i in range(m) for j in range(n)
                }
            case _:
                raise ValueError("too many dimensions")

    def highlight_io(self) -> None:
        """Highlight the inputs and outputs."""
        self._values["in_color"] = "#dae8fc"
        self._values["out_color"] = "#ffe6cc"

    def highlight(self, *, fill: str = "", stroke: str = "") -> None:
        """Highlight specific elements."""
        if fill:
            self._values[fill] = "#fff2cc"
        if stroke:
            self._values[stroke] = "#d6b656"

    def save(self, path: Path) -> None:
        """Save the diagram as a PNG file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            filled_path = Path(tmp_dir) / "tmp.drawio"
            pdf_path = Path(tmp_dir) / "tmp.pdf"
            self._fill_placeholders(self.path, filled_path)
            self._save_pdf(filled_path, pdf_path)
            pdf_to_png(pdf_path, path)
            quantize_png(path)

    def _fill_placeholders(self, in_path: Path, out_path: Path) -> None:
        in_text = in_path.read_text()
        out_text = self.PLACEHOLDER_PATTERN.sub(self._fill_placeholder, in_text)
        out_path.write_text(out_text)

    def _fill_placeholder(self, match: re.Match[str]) -> str:
        name = match.group(1)
        default_value = match.group(2)
        value = self._values.get(name, default_value)
        return html.escape(value, quote=True)

    @staticmethod
    def _save_pdf(in_path: Path, out_path: Path) -> None:
        run(
            "drawio",
            [
                "--no-sandbox",
                "--export",
                *["--format", "pdf"],
                *["--output", str(out_path)],
                str(in_path),
            ],
        )
