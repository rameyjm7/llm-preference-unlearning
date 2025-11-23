PYTHON := python3

.PHONY: install dev format lint test clean ft unlearn build

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e .[dev]

format:
	black src tests
	isort src tests

lint:
	ruff check src

test:
	pytest -q

ft:
	bash experiments/run_ft.sh

unlearn:
	bash experiments/run_unlearn.sh

build:
	$(PYTHON) -m build

clean:
	rm -rf build dist *.egg-info
	find . -name "__pycache__" -exec rm -rf {} +
