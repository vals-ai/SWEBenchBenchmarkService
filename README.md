# Benchmark Service Template

A template repository for creating benchmark services with FastAPI. Fork this repository and implement the `BenchmarkService` class to create your own service.

## Quick Start

```bash
# Install dependencies
make install

# Run the example service locally
make dev

# Run the example service in Docker
make docker-run
```

The API will be available at `http://localhost:8000`.

## What's Included

This skeleton provides:

- **Complete FastAPI implementation** - All endpoints are fully implemented in `src/app.py`
- **BenchmarkService base class** - Abstract class with methods to implement (`src/benchmark_service.py`)
- **Pydantic schemas** - Request/response validation (`src/schemas.py`)
- **Example implementation** - Working example in `main.py`
- **Dockerfile & Makefile** - Ready for containerization and development

## Architecture

```
benchmark-service-skeleton/
├── main.py                      # Entry point (update TODO here!)
├── benchmark_service/           # Framework package
│   ├── __init__.py              # Package exports
│   ├── app.py                   # FastAPI factory (fully implemented)
│   ├── base.py                  # BenchmarkService ABC
│   └── schemas.py               # Pydantic models
├── pyproject.toml               # Python dependencies
├── Dockerfile                   # Container configuration
├── Makefile                     # Development commands
└── README.md                    # This file
```

## How to Create Your Benchmark

### 1. Create your benchmark class

Implement the `BenchmarkService` abstract class. 

```python
from benchmark_service import BenchmarkService, create_app, Resources, RetrieveTaskResponse

class MyBenchmark(BenchmarkService):
    def __init__(self):
        # Load your benchmark dataset
        self.tasks = load_your_dataset()

    def filter_tasks(self, task_filter: TaskFilter) -> list[str]:
        # Filter tasks by IDs or slice
        return list(self.tasks.keys())

    def validate_task_ids(self, task_ids: list[str]) -> list[str]:
        # Validate task IDs exist
        for tid in task_ids:
            if tid not in self.tasks:
                raise ValueError(f"Invalid task: {tid}")
        return task_ids

    def retrieve_task(self, task_id: str, skip_validation: bool = False):
        # Return task metadata
        task = self.tasks[task_id]
        return RetrieveTaskResponse(
            docker_image=task["docker_image"],
            problem_statement=task["problem"],
            request_setup=False,
            cwd="/workspace",
            resources=Resources(vcpu=2, memory=4, disk=10)
        )

    def evaluate_response(self, request: EvaluateResponseRequest):
        # Evaluate a text response
        # Return EvaluationResult with your scoring
        return EvaluationResult()

    # Implement other methods as needed...
```

### 2. Update `main.py`

Pass your benchmark to `create_app()`:

```python
from benchmark_service import create_app

# Your benchmark class implementation

app = create_app(MyBenchmark())
```

### 3. Run your service

```bash
make dev
```

## API Endpoints

All endpoints are fully implemented. You just implement the `BenchmarkService` methods.

### `GET /health`
Health check.

```bash
curl http://localhost:8000/health
```

### `GET /verify-task-ids`
Verify which task IDs exist.

```bash
curl "http://localhost:8000/verify-task-ids?task_ids=task1&task_ids=task2"
```

### `GET /retrieve-task/`
Get task metadata.

```bash
curl "http://localhost:8000/retrieve-task/?task_id=example-task-1"
```

### `POST /evaluate-response/`
Evaluate a text response.

```bash
curl -X POST http://localhost:8000/evaluate-response/ \
  -H "Content-Type: application/json" \
  -d '{"task_id": "example-task-1", "response": "Hello, World!"}'
```

### `WebSocket /ws/setup-task`
Setup a task in a sandbox (for execution-based benchmarks).

### `WebSocket /ws/evaluate-instance`
Evaluate in a sandbox (for execution-based benchmarks).

### `POST /final-score/`
Calculate aggregate score.

```bash
curl -X POST http://localhost:8000/final-score/ \
  -H "Content-Type: application/json" \
  -d '{"evaluation_results": {"task1": {...}, "task2": {...}}}'
```

## BenchmarkService Methods

Implement these methods in your class:

| Method | Description | Required |
|--------|-------------|----------|
| `filter_tasks()` | Filter tasks by IDs or slice | Yes |
| `validate_task_ids()` | Validate task IDs exist | Yes |
| `retrieve_task()` | Get task metadata | Yes |
| `evaluate_response()` | Evaluate text response | For text-based benchmarks |
| `setup_task()` | Setup task in sandbox | For execution benchmarks |
| `evaluate_instance()` | Evaluate in sandbox | For execution benchmarks |
| `calculate_final_score()` | Aggregate results | Yes |

## Docker

Build and run:

```bash
make docker-build
make docker-run
```

## Development Commands

```bash
make help            # Show this help message
make install         # Install dependencies
make dev             # Start the development server
make lint            # Check style with ruff
make format          # Format code with ruff
make typecheck       # Type check code with basedpyright
make test            # Run tests
make docker-build    # Build Docker image
make docker-run      # Run Docker container

```

## Next Steps

1. Fork this repository
2. Modify `main.py` with your `BenchmarkService` implementation
3. Test with `make run` and visit `/docs`
4. Deploy with Docker
