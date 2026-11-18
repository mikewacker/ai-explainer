from pathlib import Path

import pytest

import testing
from ai_explainer.wine import story
from ai_explainer.wine.asset import Outputs


@pytest.fixture(scope="module", autouse=True)
def produce(outputs: Outputs, video: bool) -> None:
    story.learn.produce(outputs, video=video)


@pytest.mark.parametrize(*testing.asset_parameters("wine", "learn"))
def test_produce(outputs: Outputs, golden: Path) -> None:
    testing.assert_matches_golden(outputs.dir / golden.name, golden)
