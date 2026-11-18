from pathlib import Path

import pytest
from PIL import Image

from ai_explainer.common.asset._png import pdf_to_png, quantize_png
from testing.example.asset import Outputs


@pytest.fixture
def pdf_path() -> Path:
    data_dir = Path(__file__).parent / "data"
    return data_dir / "test.pdf"


def test_pdf_to_png(pdf_path: Path, outputs: Outputs) -> None:
    pdf_to_png(pdf_path, outputs.png("image"))

    with Image.open(outputs.png("image")) as img:
        assert img.format == "PNG"
        assert img.size == (900, 422)


def test_quantize_png(tmp_path: Path) -> None:
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    path = tmp_path / "test.png"
    img.save(path, format="PNG")
    quantize_png(path)

    with Image.open(path) as img:
        assert img.format == "PNG"
        assert img.mode == "P"
        assert "transparency" not in img.info
