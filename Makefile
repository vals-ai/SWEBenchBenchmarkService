.PHONY: install setup benchmark-service benchmark-service-local test-unit test-integration deploy-ecs force-deploy-ecs

PYTHON_VERSION := 3.12
IMAGE_NAME := swebench.benchmark.service
IMAGE_TAG := latest

help:
	@echo "Available commands:"
	@echo "  install - Initialize the .venv and download dependencies"
	@echo "  setup - Download dataset"
	@echo "  benchmark-service - Build and run the benchmark service"
	@echo "  benchmark-service-local - Build and run on tracker-network (port 8001)"
	@echo "  test-unit - Run unit tests"
	@echo "  test-integration - Run integration tests"

install:
	uv venv --python $(PYTHON_VERSION)
	uv sync --directory . --group dev
	
setup:
	uv run python -m src.setup

benchmark-service:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile .
	docker run -d --name $(IMAGE_NAME) -p 8000:8000 --privileged $(IMAGE_NAME):$(IMAGE_TAG)

benchmark-service-local:
	docker compose down --volumes
	docker compose up --build

test-unit:
	uv run pytest tests/unit -vv

test-integration:
	uv run pytest tests/integration -vv
