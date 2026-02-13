"""
Benchmark service entry point.

Create your own benchmark service by implementing the BenchmarkService abstract
class and passing your implementation to the create_app() function to create
the FastAPI server.
"""

from typing import Any

from fastapi import WebSocket

from benchmark_service import (
    BenchmarkService,
    EvaluateInstanceRequest,
    EvaluateResponseRequest,
    Resources,
    RetrieveTaskResponse,
    SetupTaskRequest,
    TaskFilter,
    create_app,
)


class MyBenchmark(BenchmarkService):
    """
    TODO: Replace this example with your benchmark implementation.

    This example shows a simple text-based Q&A benchmark.
    Modify it to load your own dataset and implement your evaluation logic.
    """

    def __init__(self):
        # Load your benchmark dataset here
        self.tasks = {
            "example-task-1": {
                "problem": "Write a function that returns 'Hello, World!'",
                "answer": "Hello, World!",
            },
            "example-task-2": {
                "problem": "What is 2 + 2?",
                "answer": "4",
            },
        }

    def filter_tasks(self, task_filter: TaskFilter) -> list[str]:
        """Filter tasks based on criteria."""
        all_task_ids = list(self.tasks.keys())

        if task_filter.task_ids:
            return [tid for tid in task_filter.task_ids if tid in all_task_ids]

        if task_filter.slice_str:
            slice_obj = task_filter.parse_slice()
            return all_task_ids[slice_obj]

        return all_task_ids

    def validate_task_ids(self, task_ids: list[str]) -> list[str]:
        """Validate that task IDs exist."""
        for task_id in task_ids:
            if task_id not in self.tasks:
                raise ValueError(f"Task ID not found: {task_id}")
        return task_ids

    def retrieve_task(self, task_id: str, skip_validation: bool = False) -> RetrieveTaskResponse:
        """Retrieve task metadata."""
        if not skip_validation:
            self.validate_task_ids([task_id])

        task = self.tasks[task_id]

        return RetrieveTaskResponse(
            docker_image="python:3.12-slim",
            problem_statement=task["problem"],
            request_setup=False,
            cwd="/workspace",
            resources=Resources(vcpu=2, memory=4, disk=10),
        )

    async def setup_task(self, request: SetupTaskRequest, websocket: WebSocket) -> None:
        """Setup task in sandbox (not needed for this example)."""
        await websocket.send_json({"type": "message", "data": "No setup required for example benchmark"})
        await websocket.send_json({"type": "result", "data": {"status": "ok"}})
        await websocket.close()

    def evaluate_response(self, request: EvaluateResponseRequest) -> Any:
        """Evaluate a text response."""
        task = self.tasks[request.task_id]

        # Simple string comparison
        is_correct = request.response.strip() == task["answer"]

        # Return evaluation result as a dict (you can use any structure)
        return {
            "task_id": request.task_id,
            "resolved": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "expected": task["answer"],
            "received": request.response.strip(),
        }

    async def evaluate_instance(self, request: EvaluateInstanceRequest, websocket: WebSocket) -> None:
        """Evaluate in sandbox (not implemented for this example)."""
        await websocket.send_json(
            {
                "type": "error",
                "data": "Sandbox evaluation not implemented. Use /evaluate-response/ endpoint instead.",
            }
        )
        await websocket.close()

    def calculate_final_score(self, evaluation_results: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        """Calculate final score across all evaluations."""
        total = len(evaluation_results)

        # Count resolved tasks based on result structure
        resolved = sum(1 for r in evaluation_results.values() if r and r.get("resolved", False))

        score = (resolved / total * 100) if total > 0 else 0.0

        metadata = {
            "total_tasks": total,
            "resolved_tasks": resolved,
            "unresolved_tasks": total - resolved,
        }

        return score, metadata


# Create the FastAPI app with your benchmark implementation
app = create_app(MyBenchmark())
