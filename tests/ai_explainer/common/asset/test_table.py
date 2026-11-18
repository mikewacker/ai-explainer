import re

import pytest
import torch
from PIL import Image

import testing
from testing.example.asset import Outputs, Table
from testing.example.world import ExampleData


@pytest.fixture
def features() -> list[str]:
    return ["Feature1", "Feature2"]


@pytest.fixture
def target() -> str:
    return "Target"


class TestTable:
    def test_table(
        self,
        data: ExampleData,
        features: list[str],
        target: str,
        outputs: Outputs,
        goldens: Outputs,
    ) -> None:
        table = Table(features, data.x)
        table.add_group(target, data.y)

        testing.assert_matches_golden(outputs, goldens, table, "table")

    def test_table__sample(
        self,
        data: ExampleData,
        features: list[str],
        target: str,
        outputs: Outputs,
        goldens: Outputs,
    ) -> None:
        table = Table(features, data.x[0])
        table.add_group(target, data.y[0])

        testing.assert_matches_golden(outputs, goldens, table, "table-sample")

    def test_table__highlight(
        self,
        data: ExampleData,
        features: list[str],
        target: str,
        outputs: Outputs,
        goldens: Outputs,
    ) -> None:
        table = Table(features, data.x)
        table.add_group(target, data.y)
        table.highlight(3, 2)

        testing.assert_matches_golden(outputs, goldens, table, "table-highlight")

    def test_table__dtype(self, data: ExampleData, features: list[str], target: str) -> None:
        table = Table(features, data.x)
        table.add_group(target, data.y.to(torch.long))

    def test_save(
        self, data: ExampleData, features: list[str], target: str, outputs: Outputs
    ) -> None:
        table = Table(features, data.x)
        table.add_group(target, data.y)
        table.save(outputs.png("table"))

        with Image.open(outputs.png("table")) as img:
            assert img.format == "PNG"
            assert img.size == (891, 444)
            assert img.mode == "P"

    @pytest.mark.parametrize("shape", [(2, 2), (3, 1), (3, 2, 1)])
    def test_error__invalid_data_shape(
        self, data: ExampleData, features: list[str], shape: tuple[int, ...]
    ) -> None:
        table = Table(features, data.x)
        t = torch.zeros(shape)

        with pytest.raises(ValueError, match=re.escape(f"data.shape must be (3, 2): {shape}")):
            table.add_group(["Column1", "Column2"], t)

    def test_error__highlight_invalid_index(self, data: ExampleData, features: list[str]) -> None:
        table = Table(features, data.x)

        with pytest.raises(IndexError):
            table.highlight(4, 1)
