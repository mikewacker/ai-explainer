MAKEFLAGS += -k
.PHONY: check test fix lint lint-fix type format format-fix

check: lint type format

test:
	uv run pytest

fix: lint-fix format-fix

lint:
	uv run ruff check

lint-fix:
	uv run ruff check --fix

type:
	uv run ty check

format:
	uv run ruff format --check

format-fix:
	uv run ruff format
