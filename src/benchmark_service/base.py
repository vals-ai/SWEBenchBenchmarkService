"""
Base class for benchmark service implementations.

To create your own benchmark service, subclass BenchmarkService and implement
all the abstract methods. Then use the factory function in benchmark_service.app to create
a FastAPI app with your implementation.
"""

from abc import ABC, abstractmethod
from typing import Any

from fastapi import WebSocket

from benchmark_service.schemas import (
    EvaluateInstanceRequest,
    EvaluateResponseRequest,
    RetrieveTaskResponse,
    SetupTaskRequest,
    TaskFilter,
)


class BenchmarkService(ABC):
    """
    Abstract base class for benchmark implementations.

    Implement all abstract methods to create a working benchmark service.
    The FastAPI endpoints are already implemented and will call these methods.
    """

    @abstractmethod
    def filter_tasks(self, task_filter: TaskFilter) -> list[str]:
        """
        Filter tasks based on provided criteria.

        Implement task filtering logic:
        - Load your benchmark dataset
        - Filter by task_ids if provided
        - Apply slice if provided
        - Return list of valid task IDs

        Args:
            task_filter: Filter criteria for selecting tasks

        Returns:
            List of task IDs that match the filter criteria
        """
        ...

    @abstractmethod
    def validate_task_ids(self, task_ids: list[str]) -> list[str]:
        """
        Validate that task IDs exist in your benchmark dataset.

        Implement validation logic:
        - Check if task_ids exist in your dataset
        - Raise ValueError for invalid task IDs
        - Return the validated task IDs

        Args:
            task_ids: List of task IDs to validate

        Returns:
            The same list of task IDs if all are valid

        Raises:
            ValueError: If any task ID is invalid
        """
        ...

    @abstractmethod
    def retrieve_task(self, task_id: str, skip_validation: bool = False) -> RetrieveTaskResponse:
        """
        Retrieve task metadata including environment specification and problem statement.

        Implement metadata retrieval:
        - Validate task_id exists (unless skip_validation=True)
        - Load task information from your dataset
        - Return docker image, problem statement, and resource requirements

        Args:
            task_id: The task to retrieve metadata for
            skip_validation: Skip validation of task existence

        Returns:
            RetrieveTaskResponse with task metadata
        """
        ...

    @abstractmethod
    async def setup_task(self, request: SetupTaskRequest, websocket: WebSocket) -> None:
        """
        Setup a task in a sandbox environment.

        This method should handle the complete setup process and communicate
        progress via the WebSocket connection.

        Implement setup logic:
        1. Connect to sandbox service using headers from websocket
           (x-api-key, x-api-url, x-target)
        2. Upload any setup scripts or data
        3. Execute setup commands
        4. Stream output: await websocket.send_json({"type": "message", "data": "log line"})
        5. Send final result: await websocket.send_json({"type": "result", "data": {...}})

        Args:
            request: Setup request containing task_id and instance_id
            websocket: WebSocket connection for streaming updates
        """
        ...

    @abstractmethod
    def evaluate_response(self, request: EvaluateResponseRequest) -> Any:
        """
        Evaluate a text response directly (without sandbox execution).

        Use this for benchmarks where you can evaluate responses without
        executing code or running tests.

        Implement evaluation logic:
        1. Load the expected answer/criteria for the task
        2. Compare the response against expected output
        3. Calculate scores or metrics
        4. Return your benchmark-specific evaluation result

        Args:
            request: Evaluation request with task_id and response text

        Returns:
            Your benchmark-specific evaluation result (dict, Pydantic model, etc.)
        """
        ...

    @abstractmethod
    async def evaluate_instance(self, request: EvaluateInstanceRequest, websocket: WebSocket) -> None:
        """
        Evaluate a solution in a sandbox environment.

        This method should execute tests/evaluation and communicate results
        via the WebSocket connection.

        Implement evaluation logic:
        1. Connect to sandbox service using headers from websocket
           (x-api-key, x-api-url, x-target)
        2. Execute tests or evaluation scripts
        3. Parse test output and grade results
        4. Stream logs: await websocket.send_json({"type": "message", "data": "log line"})
        5. Send final result: await websocket.send_json({"type": "result", "data": evaluation_result})

        Args:
            request: Evaluation request containing task_id and instance_id
            websocket: WebSocket connection for streaming updates
        """
        ...

    @abstractmethod
    def calculate_final_score(self, evaluation_results: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        """
        Calculate final aggregate score from all evaluation results.

        Implement scoring logic:
        - Define how to aggregate individual task results
        - Common approach: percentage of resolved tasks
        - You may want weighted scoring or other metrics
        - Return any metadata you want to include

        Args:
            evaluation_results: Dictionary mapping task_id to your evaluation result objects

        Returns:
            Tuple of (final_score, metadata_dict)
        """
        ...
