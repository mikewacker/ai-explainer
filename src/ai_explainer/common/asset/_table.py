import tempfile
from pathlib import Path

import torch
from torch import Tensor

from ._cmd import run
from ._png import quantize_png


class Table:
    """Table with multiple column groups. Each column group is separated with a border."""

    _columns: list[str]
    _data: Tensor
    _vlines: list[int]
    _highlights: list[list[int]]

    def __init__(self, columns: list[str] | str, data: Tensor) -> None:
        """Create a table from an initial group of columns."""
        columns, data = self._normalize(columns, data)
        m = data.shape[0]
        self._columns = []
        self._data = torch.empty(m, 0, dtype=torch.float32)
        self._vlines = []
        self._highlights = [[] for _ in range(m)]
        self.add_group(columns, data)

    def add_group(self, columns: list[str] | str, data: Tensor) -> None:
        """Add a group of columns to the table."""
        columns, data = self._normalize(columns, data)
        self._check_data(columns, data)
        n = self._data.shape[1]
        if n:
            self._vlines.append(n)
        self._columns.extend(columns)
        self._data = torch.hstack((self._data, data))

    def highlight(self, row: int, col: int) -> None:
        """Highlight a cell."""
        row, col = row - 1, col - 1
        _ = self._data[row, col]  # check that the cell exists
        self._highlights[row].append(col)

    def save(self, path: Path) -> None:
        """Save the table as a PNG file."""
        markup = self._create_markup()
        with tempfile.TemporaryDirectory() as tmp_dir:
            typ_path = Path(tmp_dir) / "tmp.typ"
            typ_path.write_text(markup)
            self._compile_markup(typ_path, path)
            quantize_png(path)

    def _create_markup(self) -> str:
        lines = [
            "#set page(width: auto, height: auto, margin: 0pt)",
            '#set text(font: "STIX", size: 16pt)',
            "",
            "#table(",
            f"  columns: {self._data.shape[1]},",
            "",
            *self._create_style_lines(),
            "",
            *self._create_content_lines(),
            ")",
        ]
        return "\n".join(lines)

    def _create_style_lines(self) -> list[str]:
        return [
            "  align: (col, row) => if row == 0 { center } else { right },",
            "  inset: 8pt,",
            '  fill: (col, row) => if calc.odd(row) { rgb("#f5f5f5") },',
            "  stroke: none,",
            "  table.hline(y: 1),",
            *[f"  table.vline(x: {vline})," for vline in self._vlines],
        ]

    def _create_content_lines(self) -> list[str]:
        header_cells = [f"[*{column}*]," for column in self._columns]
        header_line = "  " + " ".join(header_cells)
        data_lines = []
        for row, row_highlights in zip(self._data, self._highlights, strict=True):
            data_cells = [f"[{value:g}]," for value in row]
            for col in row_highlights:
                data_cells[col] = f'table.cell(fill: rgb("#fff2cc")){data_cells[col]}'
            data_line = "  " + " ".join(data_cells)
            data_lines.append(data_line)
        return [header_line, *data_lines]

    @staticmethod
    def _compile_markup(in_path: Path, out_path: Path) -> None:
        run("typst", ["compile", "--ppi", "300", str(in_path), str(out_path)])

    @staticmethod
    def _normalize(columns: list[str] | str, data: Tensor) -> tuple[list[str], Tensor]:
        if isinstance(columns, str):
            columns = [columns]
            data = data.unsqueeze(-1)
        data = data.to(torch.float32)
        data = torch.atleast_2d(data)
        return columns, data

    def _check_data(self, columns: list[str], data: Tensor) -> None:
        m, n = self._data.shape[0], len(columns)
        if data.ndim != 2 or data.shape[0] != m or data.shape[1] != n:
            raise ValueError(f"data.shape must be ({m}, {n}): {tuple(data.shape)}")
