from pathlib import Path

import pytest

import testing
from ai_explainer.iris import story
from ai_explainer.iris.asset.multiclass import Outputs


@pytest.fixture(scope="module", autouse=True)
def produce(outputs: Outputs) -> None:
    story.multiclass.loss_data.produce(outputs)


@pytest.mark.parametrize(*testing.asset_parameters("iris-multiclass", "loss2"))
def test_produce(outputs: Outputs, golden: Path) -> None:
    testing.assert_matches_golden(outputs.dir / golden.name, golden)
