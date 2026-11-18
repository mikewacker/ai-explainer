import re
import shutil
from pathlib import Path
from typing import overload

import pytest
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import nn

from ai_explainer.common.asset import Animation, Diagram, Outputs, Table, save

from ._ffmpeg import ffprobe_video_stream
from ._tensor import assert_tensors_equal


@overload
def assert_matches_golden(output: Path, golden: Path) -> None: ...


@overload
def assert_matches_golden(
    output: Outputs,
    golden: Outputs,
    asset: Figure | Animation | Diagram | Table | list[str] | nn.Module,
    name: str,
) -> None: ...


def assert_matches_golden(
    output: Path | Outputs,
    golden: Path | Outputs,
    asset: Figure | Animation | Diagram | Table | list[str] | nn.Module | None = None,
    name: str = "",
) -> None:
    __tracebackhide__ = True
    if isinstance(output, Outputs) and isinstance(golden, Outputs) and asset is not None:
        output, golden = _save_and_get_paths(output, golden, asset, name)
    if isinstance(output, Outputs) or isinstance(golden, Outputs):
        raise TypeError()

    assert output.exists(), f"output file not found: {output}"
    try:
        assert golden.exists(), f"golden file not found: {golden}"
        match golden.suffix:
            case ".png":
                _assert_png_metadata_match(output, golden)
                _assert_binary_files_equal(output, golden)
            case ".mp4":
                _assert_mp4_metadata_match(output, golden)
                _assert_binary_files_equal(output, golden)
            case ".txt":
                _assert_txt_files_equal(output, golden)
            case ".pth":
                _assert_pth_files_match(output, golden)
            case _:
                raise ValueError(f"unsupported extension: {golden.suffix}")
    except AssertionError as failure:
        _keep_output(output, failure)
        raise


def asset_parameters(
    act: str, sequence: str, *, skip: list[int] | None = None
) -> tuple[str, list[object]]:
    skip = skip or []
    goldens = Outputs(Path("golden", "asset")).act(act)
    pattern = re.compile(rf"^{re.escape(act)}_{re.escape(sequence)}(\d\d)_")
    asset_paths = [path for path in goldens.dir.iterdir() if _is_tested_asset(pattern, skip, path)]
    if not asset_paths:
        raise ValueError(f'no assets found: act "{act}", sequence "{sequence}"')

    asset_paths.sort(key=lambda path: path.name)
    params = [_param(path) for path in asset_paths]
    return "golden", params


def _assert_png_metadata_match(actual_path: Path, expected_path: Path) -> None:
    __tracebackhide__ = True
    with Image.open(actual_path) as actual, Image.open(expected_path) as expected:
        assert actual.format == expected.format, "formats are not equal"
        assert actual.size == expected.size, "sizes are not equal"
        assert actual.mode == expected.mode, "modes are not equal"
        if actual.info.get("transparency") != expected.info.get("transparency"):
            raise AssertionError("transparency is not equal")


def _assert_mp4_metadata_match(actual_path: Path, expected_path: Path) -> None:
    __tracebackhide__ = True
    actual = ffprobe_video_stream(actual_path)
    expected = ffprobe_video_stream(expected_path)
    assert actual["width"] == expected["width"], "widths are not equal"
    assert actual["height"] == expected["height"], "heights are not equal"
    assert actual["nb_frames"] == expected["nb_frames"], "frames are not equal"
    assert actual["avg_frame_rate"] == expected["avg_frame_rate"], "frame rates are not equal"
    assert actual["codec_name"] == expected["codec_name"], "codecs are not equal"
    assert actual["pix_fmt"] == expected["pix_fmt"], "pixel formats are not equal"


def _assert_txt_files_equal(actual_path: Path, expected_path: Path) -> None:
    __tracebackhide__ = True
    actual = actual_path.read_text().splitlines()
    expected = expected_path.read_text().splitlines()
    assert actual == expected


def _assert_pth_files_match(actual_path: Path, expected_path: Path) -> None:
    __tracebackhide__ = True
    actual = torch.load(actual_path, weights_only=True)
    expected = torch.load(expected_path, weights_only=True)
    assert actual.keys() == expected.keys(), "parameter structure is not equal"
    for name, actual_param in actual.items():
        expected_param = expected[name]
        try:
            assert_tensors_equal(actual_param, expected_param)
        except AssertionError as failure:
            failure.add_note(f"Parameter: {name}")
            raise


def _assert_binary_files_equal(actual_path: Path, expected_path: Path) -> None:
    __tracebackhide__ = True
    actual = actual_path.read_bytes()
    expected = expected_path.read_bytes()
    if actual != expected:
        raise AssertionError("binary files are not equal")


def _save_and_get_paths(
    outputs: Outputs,
    goldens: Outputs,
    asset: Figure | Animation | Diagram | Table | list[str] | nn.Module,
    name: str,
) -> tuple[Path, Path]:
    match asset:
        case Figure():
            output = outputs.png(name)
            save(asset, output)
            golden = goldens.png(name)
        case Animation():
            output = outputs.mp4(name)
            asset.save(output)
            golden = goldens.mp4(name)
        case Diagram():
            output = outputs.png(name)
            asset.save(output)
            golden = goldens.png(name)
        case Table():
            output = outputs.png(name)
            asset.save(output)
            golden = goldens.png(name)
        case nn.Module():
            output = outputs.pth(name)
            torch.save(asset.state_dict(), output)
            golden = goldens.pth(name)
        case list():
            output = outputs.txt(name)
            save(asset, output)
            golden = goldens.txt(name)
    return output, golden


def _keep_output(output: Path, failure: AssertionError) -> None:
    try:
        failures_dir = Path("out", "failures")
        try:
            failures_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            failure.add_note(f"failed to create failures dir: {e}")
            return

        try:
            shutil.copy2(output, failures_dir)
        except OSError as e:
            failure.add_note(f"failed to copy {output.name}: {e}")
    except Exception as e:
        failure.add_note(f"unexpected error copying {output.name}: {e}")


def _is_tested_asset(pattern: re.Pattern, skip: list[int], path: Path) -> bool:
    if not path.is_file():
        return False

    match = pattern.match(path.name)
    if match is None:
        return False

    idx = int(match.group(1))
    return idx not in skip


def _param(path: Path) -> object:
    marks = [pytest.mark.asset]
    if path.suffix == ".mp4":
        marks.append(pytest.mark.video)
    return pytest.param(path, id=path.name, marks=marks)
