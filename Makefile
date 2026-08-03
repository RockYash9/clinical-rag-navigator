.PHONY: install install-dev test lint format run-api ingest build-index clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest --cov=src/clinical_rag --cov-report=term-missing

lint:
	ruff check src tests
	mypy src

format:
	black src tests scripts

run-api:
	uvicorn clinical_rag.api.main:app --reload --host 0.0.0.0 --port 8000

cli:
	python scripts/cli.py

ingest:
	python scripts/ingest.py

build-index:
	python scripts/build_index.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
