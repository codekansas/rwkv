# Makefile

py-files := $(filter-out ml/api.py, $(shell git ls-files '*.py'))

format:
	@black $(py-files)
	@ruff --fix $(py-files)
.PHONY: format

static-checks:
	@black --diff --check $(py-files)
	@ruff $(py-files)
	@mypy --install-types --non-interactive $(py-files)
.PHONY: lint

test:
	python -m pytest
.PHONY: test
