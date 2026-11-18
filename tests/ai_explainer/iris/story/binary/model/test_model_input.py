from pathlib import Path

import pytest

import testing
from ai_explainer.iris import story
from ai_explainer.iris.asset.binary import Outputs


@pytest.fixture(scope="module", autouse=True)
def produce(outputs: Outputs) -> None:
    story.binary.model.model_input.produce(outputs)


@pytest.mark.parametrize(*testing.asset_parameters("iris-binary-model", "model2"))
def test_produce(outputs: Outputs, golden: Path) -> None:
    testing.assert_matches_golden(outputs.dir / golden.name, golden)
