.PHONY: install install-docs data train baseline test lint docs docs-build docker clean

install:
	pip install -e ".[dev]"

install-docs:
	pip install -e ".[docs]"

data:
	python scripts/generate_data.py --config configs/default.yaml

train:
	python scripts/train.py --config configs/default.yaml --auto-generate

baseline:
	python scripts/train.py --config configs/baseline_no_physics.yaml --auto-generate

test:
	pytest -q

lint:
	ruff check src scripts tests

docs:
	mkdocs serve

docs-build:
	mkdocs build --strict

docker:
	docker build -f k8s/Dockerfile -t marsworld:latest .

clean:
	rm -rf site build dist *.egg-info data checkpoints
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
