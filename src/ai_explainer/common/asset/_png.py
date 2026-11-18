from pathlib import Path

from PIL import Image

from ._cmd import run


def pdf_to_png(in_path: Path, out_path: Path) -> None:
    run(
        "pdftocairo",
        [
            "-png",
            *["-r", "300"],
            *["-antialias", "best"],
            "-singlefile",
            str(in_path),
            str(out_path.with_suffix("")),
        ],
    )


def quantize_png(path: Path) -> None:
    """Quantize the color palette into 8-bit colors for a PNG file."""
    with Image.open(path) as img:
        img.convert("RGB").quantize(colors=256).save(path)
