.PHONY: install lint test pipeline api docker

install:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

test:
	uv run pytest --cov --cov-report=term-missing

# Download BTS data for the given months, validate, build features, train.
pipeline:
	uv run flightdelay run --year 2024 --months 1 2 3 4

api:
	uv run uvicorn flightdelay.api:app --reload

docker:
	docker build -t flightdelay-api .
