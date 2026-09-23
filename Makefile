.PHONY: setup test lint fmt

setup:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	@echo "Activate with: source .venv/bin/activate"

# Diarization and framing are heavy; install them when you reach those stages.
setup-full:
	.venv/bin/pip install -e ".[dev,diarize,framing]"

test:
	pytest -q

lint:
	ruff check .

fmt:
	ruff format .
