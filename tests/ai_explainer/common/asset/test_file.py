from pathlib import Path

import pytest

from testing.example.asset import Outputs


class TestOutputContext:
    def test_paths(self) -> None:
        outputs = Outputs(Path("out")).act("act").sequence("sequence")
        path1 = outputs.png("name1")
        path2 = outputs.mp4("name2")
        path3 = outputs.txt("name3")
        path4 = outputs.pth("name4")

        assert outputs.dir == Path("out") / "act"
        assert path1 == Path("out") / "act" / "act_sequence01_name1.png"
        assert path2 == Path("out") / "act" / "act_sequence02_name2.mp4"
        assert path3 == Path("out") / "act" / "act_sequence03_name3.txt"
        assert path4 == Path("out") / "act" / "act_sequence04_name4.pth"

    def test_paths__skip(self) -> None:
        outputs = Outputs(Path("out")).act("act").sequence("sequence")
        outputs.skip()
        path = outputs.png("name")

        assert path == Path("out") / "act" / "act_sequence02_name.png"

    def test_paths__multi_level_act(self) -> None:
        outputs = Outputs(Path("out")).act("arc-act").sequence("sequence")
        path = outputs.png("name")

        assert outputs.dir == Path("out") / "arc" / "act"
        assert path == Path("out") / "arc" / "act" / "arc-act_sequence01_name.png"

    def test_paths__no_sequence(self) -> None:
        outputs = Outputs(Path("out")).act("act")
        path = outputs.png("name")

        assert path == Path("out") / "act" / "act_name.png"

    def test_mkdir(self, tmp_path: Path) -> None:
        outputs = Outputs(tmp_path).act("act")
        outputs.mkdir()

        assert outputs.dir.exists()

    def test_mkdir__exists(self, tmp_path: Path) -> None:
        outputs = Outputs(tmp_path).act("act")
        outputs.dir.mkdir(parents=True)
        outputs.mkdir()

    def test_repr(self) -> None:
        outputs = Outputs(Path("out")).act("act")
        text = repr(outputs)

        assert text == repr(Path("out") / "act")

    def test_error__act(self) -> None:
        outputs = Outputs(Path("out"))

        with pytest.raises(ValueError, match="act is not set"):
            outputs.sequence("sequence")
        with pytest.raises(ValueError, match="act is not set"):
            outputs.png("name")

        outputs = outputs.act("act1")

        with pytest.raises(ValueError, match="act is already set"):
            outputs.act("act2")

    def test_error__sequence(self) -> None:
        outputs = Outputs(Path("out")).act("act")

        with pytest.raises(ValueError, match="sequence is not set"):
            outputs.skip()

        outputs = outputs.sequence("sequence1")

        with pytest.raises(ValueError, match="sequence is already set"):
            outputs.sequence("sequence2")
