import testing
import testing.example as aix
from testing.example.asset import Outputs, pd


def test_equation() -> None:
    steps = [r"\hat{y}", "w", "5"]
    equation = aix.asset.equation(steps)

    assert equation == r"\hat{y} = w = 5"


def test_equation__approx_steps() -> None:
    steps = [r"\hat{y}", "w x", r"0.11 \cdot 0.22", "0.02"]
    equation = aix.asset.equation(steps, approx_steps=2)

    assert equation == r"\hat{y} = w x \approx 0.11 \cdot 0.22 \approx 0.02"


def test_equation__vector() -> None:
    steps = [[r"\hat{y}_1", "w_1 x"], [r"\hat{y}_2", "w_2 x"]]
    equation = aix.asset.equation(steps)

    assert equation.splitlines() == [
        r"\begin{bmatrix}",
        r"\hat{y}_1 \\",
        r"\hat{y}_2",
        r"\end{bmatrix}",
        r"= \begin{bmatrix}",
        r"w_1 x \\",
        r"w_2 x",
        r"\end{bmatrix}",
    ]


def test_equation__multiline() -> None:
    steps = [r"\hat{y}", "w", "5"]
    equation = aix.asset.equation(steps, multiline=True)

    assert equation.splitlines() == [
        r"\begin{align*}",
        r"\hat{y}",
        r"&= w \\",
        r"&= 5",
        r"\end{align*}",
    ]


def test_equation__multiline__approx_steps() -> None:
    steps = [r"\hat{y}", "w x", r"0.11 \cdot 0.22", "0.02"]
    equation = aix.asset.equation(steps, multiline=True, approx_steps=2)

    assert equation.splitlines() == [
        r"\begin{align*}",
        r"\hat{y}",
        r"&= w x \\",
        r"&\approx 0.11 \cdot 0.22 \\",
        r"&\approx 0.02",
        r"\end{align*}",
    ]


def test_equation__multiline_align_terms() -> None:
    steps = [r"\hat{y}", "w_1 x_1 + w_2 x_2", r"2 \cdot 3 + 4 \cdot 5", "6 + 20", "26"]
    equation = aix.asset.equation(steps, multiline=True, align_terms=True)

    assert equation.splitlines() == [
        r"\begin{alignat*}{2}",
        r"\hat{y}",
        r"&= w_1 x_1 &&+ w_2 x_2 \\",
        r"&= 2 \cdot 3 &&+ 4 \cdot 5 \\",
        r"&= 6 &&+ 20 \\",
        r"&= 26",
        r"\end{alignat*}",
    ]


def test_equation__multiline_2_steps() -> None:
    steps = ["w", "5"]
    equation = aix.asset.equation(steps, multiline=True)

    assert equation == "w = 5"


def test_equation__1_step() -> None:
    steps = ["w"]
    equation = aix.asset.equation(steps)

    assert equation == "w"


def test_equation__0_steps() -> None:
    equation = aix.asset.equation([])

    assert equation == ""


def test_equation__too_many_approx_steps() -> None:
    steps = ["w", "0.11"]
    equation = aix.asset.equation(steps, approx_steps=10)

    assert equation == r"w \approx 0.11"


def test_pd() -> None:
    step = pd("L", "w_1")

    assert step == r"\frac{\partial L}{\partial w_1}"


def test_pd__t() -> None:
    step = pd("L", "w_1", t=True)

    assert step == r"\tfrac{\partial L}{\partial w_1}"


def test_save(outputs: Outputs, goldens: Outputs) -> None:
    equations = [r"\hat{y} = w", r"\hat{y} = w_1 x_1 + w_2 x_2"]

    testing.assert_matches_golden(outputs, goldens, equations, "equation")
