from pathlib import Path

import pytest

import testing
from ai_explainer.iris import story
from ai_explainer.iris.asset.binary import Outputs


@pytest.fixture(scope="module", autouse=True)
def produce(outputs: Outputs) -> None:
    story.binary.train.slope_hidden.produce(outputs)


@pytest.mark.parametrize(*testing.asset_parameters("iris-binary-train", "slope3"))
def test_produce(outputs: Outputs, golden: Path) -> None:
    testing.assert_matches_golden(outputs.dir / golden.name, golden)
