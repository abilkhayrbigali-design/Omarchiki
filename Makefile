.PHONY: sync test test-fast check lint fmt docker-build docker-check

sync:
	uv sync

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow"

check:
	uv run python -m auditgym check

lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests

docker-build:
	docker build -t auditgym:dev .

docker-check: docker-build
	docker run --rm --network none --cpus 2 --memory 4g auditgym:dev
