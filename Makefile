.PHONY: install lint test cov all

# Source trees that must stay lint-clean (examples included, so a broken
# example import can never slip through again).
LINT_PATHS = nanoma tests examples

install:
	pip install -e ".[dev]"

lint:
	python -m pyflakes $(LINT_PATHS)

test:
	python -m pytest tests/ -q

cov:
	python -m pytest tests/ -q --cov=nanoma --cov-report=term-missing

# Run before committing.
all: lint test
