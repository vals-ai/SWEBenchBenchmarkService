.PHONY: start-fastapi install task-setup quick-start test-unit test-integration deploy-ecs force-deploy-ecs

PYTHON_VERSION := 3.12

help:
	@echo "Available commands:"
	@echo "  install - Install the dependencies"
	@echo "  start-fastapi - Start the FastAPI server"

install:
	uv venv --python $(PYTHON_VERSION)
	uv sync --directory . --group dev
	@echo "🎉 Done! Run 'source .venv/bin/activate' to activate the environment locally."

task-setup:
	uv run python -m src.setup

start-fastapi:
	uv run fastapi dev main.py

quick-start:
	docker build -t swe-bench:latest -f Dockerfile .
	docker run -d --privileged swe-bench:latest

test-unit:
	uv run pytest tests/unit -vv

test-integration:
	uv run pytest tests/integration -vv

deploy-ecs:
	uv run cdk deploy --verbose

force-deploy-ecs:
	uv run cdk deploy --hotswap
