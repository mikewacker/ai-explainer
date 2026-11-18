import pytest


@pytest.fixture(scope="session")
def act() -> str:
    return "iris-multiclass"
