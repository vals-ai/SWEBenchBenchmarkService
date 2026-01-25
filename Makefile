.PHONY: setup benchmark-service test-unit test-integration deploy-ecs force-deploy-ecs

PYTHON_VERSION := 3.12
IMAGE_NAME := swebench.benchmark.service
IMAGE_TAG := latest

help:
	@echo "Available commands:"
	@echo "  install - Initialize the .venv and download dependencies"
	@echo "  setup - Download dataset"
	@echo "  benchmark-service - Build and run the benchmark service"
	@echo "  test-unit - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  deploy-ecs - Deploy the benchmark service to AWS ECS"
	@echo "  force-deploy-ecs - Force deploy the benchmark service to AWS ECS"

install:
	uv venv --python $(PYTHON_VERSION)
	uv sync --directory . --group dev
	
setup:
	uv run python -m src.setup

benchmark-service:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile .
	docker run -d --name $(IMAGE_NAME) -p 8000:8000 --privileged $(IMAGE_NAME):$(IMAGE_TAG)

start-fastapi:
	uv run fastapi run main.py --host 0.0.0.0 --port 8000

test-unit:
	uv run pytest tests/unit -vv

test-integration:
	uv run pytest tests/integration -vv

deploy-ecs:
	uv run cdk deploy --verbose

force-deploy-ecs:
	uv run cdk deploy --hotswap
