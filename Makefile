.PHONY: setup benchmark-service test-unit test-integration deploy-ecs force-deploy-ecs

PYTHON_VERSION := 3.12
IMAGE_NAME := swebench.benchmark.service
IMAGE_TAG := latest

help:
	@echo "Available commands:"
	@echo "  setup - Initialize the .venv and download dataset"
	@echo "  benchmark-service - Build and run the benchmark service"
	@echo "  test-unit - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  deploy-ecs - Deploy the benchmark service to AWS ECS"
	@echo "  force-deploy-ecs - Force deploy the benchmark service to AWS ECS"

setup:
	uv venv --python $(PYTHON_VERSION)
	uv sync --directory . --group dev
	uv run python -m src.setup

benchmark-service:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile .
	docker run -d --name $(IMAGE_NAME) --privileged $(IMAGE_NAME):$(IMAGE_TAG)

test-unit:
	uv run pytest tests/unit -vv

test-integration:
	uv run pytest tests/integration -vv

deploy-ecs:
	uv run cdk deploy --verbose

force-deploy-ecs:
	uv run cdk deploy --hotswap
