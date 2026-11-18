from pathlib import Path

import pytest
import torch
from PIL import Image

import testing
from testing.example.asset import Diagram, Outputs


@pytest.fixture
def drawio_path() -> Path:
    data_dir = Path(__file__).parent / "data"
    return data_dir / "diagram.drawio"


class TestDiagram:
    def test_diagram(self, drawio_path: Path, outputs: Outputs, goldens: Outputs) -> None:
        diag = Diagram(drawio_path)

        testing.assert_matches_golden(outputs, goldens, diag, "diagram")
        assert diag.path == drawio_path

    def test_fill_tensor(self, drawio_path: Path, outputs: Outputs, goldens: Outputs) -> None:
        diag = Diagram(drawio_path)
        diag.fill_tensor("yhat", torch.tensor(1.0))
        diag.fill_tensor("y", torch.tensor(3.0))
        diag.fill_tensor("L", torch.tensor(4.0))

        testing.assert_matches_golden(outputs, goldens, diag, "diagram-fill_tensor")

    def test_fill_tensor__0d(self, drawio_path: Path) -> None:
        diag = Diagram(drawio_path)
        diag.fill_tensor("b", torch.tensor(1.2))

        assert diag.values() == {"b": "1.2"}

    def test_fill_tensor__1d(self, drawio_path: Path) -> None:
        diag = Diagram(drawio_path)
        diag.fill_tensor("w", torch.tensor([1.2, 3.4]))

        assert diag.values() == {"w1": "1.2", "w2": "3.4"}

    def test_fill_tensor__2d(self, drawio_path: Path) -> None:
        diag = Diagram(drawio_path)
        diag.fill_tensor("x", torch.tensor([[1.2, 3.4], [5.6, 7.8], [9.1, 2.3]]))

        assert diag.values() == {
            **{"x11": "1.2", "x12": "3.4"},
            **{"x21": "5.6", "x22": "7.8"},
            **{"x31": "9.1", "x32": "2.3"},
        }

    def test_fill_tensor__decimals(self, drawio_path: Path) -> None:
        diag = Diagram(drawio_path)
        diag.fill_tensor("b", torch.tensor(4.567), decimals=1)

        assert diag.values() == {"b": "4.6"}

    def test_highlight_io(self, drawio_path: Path, outputs: Outputs, goldens: Outputs) -> None:
        diag = Diagram(drawio_path)
        diag.highlight_io()

        testing.assert_matches_golden(outputs, goldens, diag, "diagram-highlight_io")

    def test_highlight(self, drawio_path: Path, outputs: Outputs, goldens: Outputs) -> None:
        diag = Diagram(drawio_path)
        diag.highlight(fill="in_color", stroke="special_stroke")
        diag.highlight(fill="se_fill", stroke="se_stroke")

        testing.assert_matches_golden(outputs, goldens, diag, "diagram-highlight")

    def test_save(self, drawio_path: Path, outputs: Outputs) -> None:
        diag = Diagram(drawio_path)
        diag.save(outputs.png("diagram"))

        with Image.open(outputs.png("diagram")) as img:
            assert img.format == "PNG"
            assert img.size == (900, 422)
            assert img.mode == "P"

    def test_values__copy(self, drawio_path: Path) -> None:
        diag = Diagram(drawio_path)
        values1 = diag.values()
        values2 = diag.values()
        values2["yhat"] = "1"

        assert values1 != values2

    def test_error__file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            Diagram(Path("does-not-exist"))

    def test_error__invalid_tensor_shape(self, drawio_path: Path) -> None:
        diag = Diagram(drawio_path)

        with pytest.raises(ValueError, match="too many dimensions"):
            diag.fill_tensor("t", torch.tensor([[[1.0]]]))
