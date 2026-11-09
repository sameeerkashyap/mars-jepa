.PHONY: install install-docs data train baseline test lint docs docs-build docker clean

install:
	pip install -e ".[dev]"

install-docs:
	pip install -e ".[docs]"

data:
	python -m scripts.generate_data --config configs/pi_jepa/default.yaml

train:
	python -m scripts.train --config configs/pi_jepa/default.yaml --auto-generate

baseline:
	python -m scripts.train --config benchmarks/pi_jepa/no_physics.yaml --auto-generate

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
