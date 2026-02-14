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

- **Complete FastAPI implementation** - All endpoints are fully implemented in `src/benchmark_service/app.py`
- **BenchmarkService base class** - Abstract class with common implementations (`src/benchmark_service/base.py`)
- **Pydantic schemas** - Request/response validation (`src/benchmark_service/schemas.py`)
- **Example implementation** - Working example in `main.py`
- **Dockerfile & Makefile** - Ready for containerization and development

## Architecture

```
benchmark-service-skeleton/
├── main.py                      # Entry point - implement your benchmark here!
├── src/
│   └── benchmark_service/       # Framework package
│       ├── __init__.py          # Package exports
│       ├── app.py               # FastAPI factory (fully implemented)
│       ├── base.py              # BenchmarkService ABC with common methods
│       └── schemas.py           # Pydantic models
├── pyproject.toml               # Python dependencies
├── Dockerfile                   # Container configuration
├── Makefile                     # Development commands
└── README.md                    # This file
```

## How to Create Your Benchmark

### 1. Create your benchmark class

Implement the `BenchmarkService` abstract class

```python
from typing import Any
from benchmark_service import BenchmarkService, create_app, Resources, RetrieveTaskResponse

class MyBenchmark(BenchmarkService):
    def load_dataset(self) -> dict[str, Any]:
        """Load and return your benchmark dataset as a dict mapping task_id to task data."""
        return {
            "task-1": {
                "docker_image": "python:3.12-slim",
                "problem": "Write a function that returns 'Hello, World!'",
                "answer": "Hello, World!",
            },
            "task-2": {
                "docker_image": "python:3.12-slim",
                "problem": "What is 2 + 2?",
                "answer": "4",
            },
        }

    def retrieve_task(self, task_id: str, skip_validation: bool = False):
        """Return task metadata."""
        if not skip_validation:
            self.validate_task_ids([task_id])

        task = self.tasks[task_id]
        return RetrieveTaskResponse(
            docker_image=task["docker_image"],
            problem_statement=task["problem"],
            request_setup=False,
            cwd="/workspace",
            resources=Resources(vcpu=2, memory=4, disk=10)
        )

    def evaluate_response(self, request: EvaluateResponseRequest):
        """Evaluate a text response."""
        task = self.tasks[request.task_id]
        is_correct = request.response.strip() == task["answer"]

        return {
            "task_id": request.task_id,
            "resolved": is_correct,
            "score": 1.0 if is_correct else 0.0,
        }

    async def setup_task(self, task_id: str, sandbox: AsyncSandbox):
        """Setup task in sandbox environment."""
        from benchmark_service import StreamMessageChunk, StreamResultChunk

        # The sandbox is already connected - just use it!
        # Upload files, execute commands, etc.

        yield StreamMessageChunk(type="message", data="Starting setup...")

        # Example: upload a setup script
        # await sandbox.fs.upload_file(script.encode(), "/setup.sh")

        # Example: execute commands
        # result = await sandbox.process.exec("bash /setup.sh")
        # yield StreamMessageChunk(type="message", data=result.result)

        yield StreamResultChunk(type="result", data={"status": "ok"})

    async def evaluate_instance(self, task_id: str, sandbox: AsyncSandbox):
        """Evaluate solution in sandbox environment."""
        from benchmark_service import StreamMessageChunk, StreamResultChunk

        # Run tests in the sandbox
        yield StreamMessageChunk(type="message", data="Running tests...")

        # Example: execute tests
        # result = await sandbox.process.exec("pytest tests/")
        # yield StreamMessageChunk(type="message", data=result.result)

        # Yield final evaluation result
        yield StreamResultChunk(type="result", data={"resolved": True, "score": 1.0})

    def calculate_final_score(self, evaluation_results: dict[str, Any]) -> FinalScoreResult:
        """Calculate aggregate score from all evaluations."""
        from benchmark_service.schemas import FinalScoreResult

        total = len(evaluation_results)
        resolved = sum(1 for r in evaluation_results.values() if r.get("resolved", False))
        score = (resolved / total * 100) if total > 0 else 0.0

        return FinalScoreResult(score=score, metadata={"total": total, "resolved": resolved})
```

**Streaming Chunk Types:**

When yielding from `setup_task` or `evaluate_instance`, use these Pydantic models:
- `StreamMessageChunk(type="message", data="...")` - Log messages and progress updates
- `StreamResultChunk(type="result", data={...})` - Final result (any structure)
- `StreamErrorChunk(type="error", data="...")` - Error messages

These are unified as the `StreamChunk` union type for type safety.
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

### Methods to Implement

| Method | Description | Required |
|--------|-------------|----------|
| `load_dataset()` | Load and return task dataset as dict | Yes |
| `retrieve_task()` | Get task metadata | Yes |
| `evaluate_response()` | Evaluate text response | For text-based benchmarks |
| `setup_task()` | Setup task in sandbox | For execution benchmarks |
| `evaluate_instance()` | Evaluate in sandbox | For execution benchmarks |
| `calculate_final_score()` | Aggregate results | Yes |

### Methods Provided by Base Class

These are already implemented - you don't need to override them:

| Method | Description |
|--------|-------------|
| `__init__()` | Constructor that calls `load_dataset()` and stores tasks |
| `filter_tasks()` | Filter tasks by IDs or slice |
| `validate_task_ids()` | Validate task IDs exist |

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
