import math
from pathlib import Path
from typing import cast


def equation(
    steps: list[str] | list[list[str]],
    *,
    approx_steps: int = 0,
    multiline: bool = False,
    align_terms: bool = False,
) -> str:
    """Create an equation with multiple steps."""
    steps = _format_steps(steps)
    if not steps:
        return ""
    if multiline and len(steps) > 2:
        return _equation_multiline(steps, approx_steps=approx_steps, align_terms=align_terms)
    return _equation_oneline(steps, approx_steps=approx_steps)


def pd(o: str, i: str, *, t: bool = False) -> str:
    """Create the symbol for a partial derivative."""
    frac = "tfrac" if t else "frac"
    return rf"\{frac}{{\partial {o}}}{{\partial {i}}}"


def save_equations(equations: list[str], path: Path) -> None:
    """Save the equations as a text file."""
    text = "".join(f"{equation}\n\n" for equation in equations)
    path.write_text(text[:-1])


def _equation_oneline(steps: list[str], *, approx_steps: int) -> str:
    split = _get_split(steps, approx_steps)
    steps = [
        steps[0],
        *[f"= {step}" for step in steps[1:split]],
        *[rf"\approx {step}" for step in steps[split:]],
    ]
    delimiter = "\n" if steps[0].startswith(r"\begin{bmatrix}") else " "
    return delimiter.join(steps)


def _equation_multiline(steps: list[str], *, approx_steps: int, align_terms: bool) -> str:
    split = _get_split(steps, approx_steps)
    steps = [
        steps[0],
        *[rf"&= {step} \\" for step in steps[1:split]],
        *[rf"&\approx {step} \\" for step in steps[split:]],
    ]
    steps[-1] = steps[-1].removesuffix(r" \\")
    if align_terms:
        steps = [step.replace("+", "&&+") for step in steps]
        cols = max(step.count("&") for step in steps) + 1
        n = math.ceil(cols / 2)
        begin_env = rf"\begin{{alignat*}}{{{n}}}"
        end_env = r"\end{alignat*}"
    else:
        begin_env = r"\begin{align*}"
        end_env = r"\end{align*}"
    return "\n".join([begin_env, *steps, end_env])


def _format_steps(steps: list[str] | list[list[str]]) -> list[str]:
    if not steps or isinstance(steps[0], str):
        return cast(list[str], steps)
    return [
        "\n".join([r"\begin{bmatrix}", " \\\\\n".join(step), r"\end{bmatrix}"])
        for step in zip(*steps, strict=True)
    ]


def _get_split(steps: list[str], approx_steps: int) -> int:
    split = len(steps) - approx_steps
    return max(split, 1)
