# Starlette 1.2+ type-checks TestClient against the optional `httpx2` package, which is
# not installed here, so members of TestClient resolve to Unknown under basedpyright.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
import json
import logging
import os
from collections.abc import AsyncGenerator, Mapping
from typing import Any

from benchmark_service import ExecResult, Sandbox
from fastapi.testclient import TestClient
from httpx import Response
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from main import app

logger = logging.getLogger(__name__)


class BenchmarkServiceTestClient:
    """
    Contains utilities for testing the benchmark service
    """

    _client: TestClient

    def __init__(self) -> None:
        self._client = TestClient(app, raise_server_exceptions=False)
        self._client.__enter__()

    def close(self) -> None:
        self._client.__exit__(None, None, None)

    def _receive_websocket_message(self, websocket: WebSocketTestSession) -> str | dict[str, Any]:
        message: dict[str, Any] = websocket.receive_json()
        if message["type"] == "message":
            return message["data"]

        if message["type"] == "result":
            data = message["data"]
            # Handle both dict and string formats
            if isinstance(data, str):
                return json.loads(data)
            return data

        raise ValueError(f"Unknown websocket message type: {message['type']}")

    async def request_health_check(self) -> Response:
        """
        Requests health check from benchmark service
        """
        response = self._client.get("/health")
        logger.info(f"Health check response: {response.text}")
        return response

    async def request_verify_task_ids(
        self,
        task_ids: list[str] | None = None,
        slice_str: str | None = None,
        dataset: str | None = None,
    ) -> Response:
        """
        Requests verify task ids from benchmark service
        """
        params: dict[str, Any] = {}
        if task_ids is not None:
            params["task_ids"] = task_ids
        if slice_str is not None:
            params["slice"] = slice_str
        if dataset is not None:
            params["dataset"] = dataset

        response = self._client.get("/verify-task-ids", params=params)
        logger.info(f"Verify task ids response: {response.text}")
        return response

    async def request_retrieve_task(
        self, task_id: str, skip_validation: bool = False, dataset: str | None = None
    ) -> Response:
        """
        Requests retrieve task from benchmark service
        """
        params: dict[str, Any] = {"task_id": task_id, "skip_validation": str(skip_validation), "dataset": dataset}
        response = self._client.get("/retrieve-task/", params=params)
        logger.info(f"Retrieve task response: {response.text}")
        return response

    async def request_setup_task(
        self, task_id: str, instance_id: str, dataset: str | None = None
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        """
        Requests setup task from benchmark service via WebSocket
        """
        api_key = os.getenv("DAYTONA_API_KEY")
        api_url = os.getenv("DAYTONA_API_URL")
        target = os.getenv("DAYTONA_TARGET")

        if not api_key or not api_url or not target:
            raise ValueError("API key, API URL, and target are required")

        json_data: dict[str, Any] = {
            "task_id": task_id,
            "instance_id": instance_id,
            "dataset": dataset,
        }

        headers = {
            "x-api-key": api_key,
            "x-api-url": api_url,
            "x-target": target,
        }

        with self._client.websocket_connect("/ws/setup-task", headers=headers) as websocket:
            websocket.send_json(json_data)

            while True:
                try:
                    message = self._receive_websocket_message(websocket)
                    yield message
                except WebSocketDisconnect:
                    logger.info("WebSocket closed for setup task")
                    break

    async def request_evaluate_instance(
        self, task_id: str, instance_id: str, dataset: str | None = None
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        """
        Requests evaluate instance from benchmark service via WebSocket
        """
        api_key = os.getenv("DAYTONA_API_KEY")
        api_url = os.getenv("DAYTONA_API_URL")
        target = os.getenv("DAYTONA_TARGET")

        if not api_key or not api_url or not target:
            raise Exception("API key, API URL, and target are required")

        json_data: dict[str, Any] = {
            "task_id": task_id,
            "instance_id": instance_id,
            "dataset": dataset,
        }

        headers = {
            "x-api-key": api_key,
            "x-api-url": api_url,
            "x-target": target,
        }

        with self._client.websocket_connect("/ws/evaluate-instance", headers=headers) as websocket:
            websocket.send_json(json_data)

            while True:
                try:
                    message = self._receive_websocket_message(websocket)
                    yield message
                except WebSocketDisconnect:
                    logger.info("WebSocket closed for evaluate instance")
                    break

    async def request_final_score(
        self, evaluation_results: Mapping[str, dict[str, Any] | None], dataset: str | None = None
    ) -> Response:
        """
        Requests final score from benchmark service
        """
        json_data: dict[str, Any] = {"evaluation_results": evaluation_results, "dataset": dataset}
        headers = {"Content-Type": "application/json"}

        response = self._client.post("/final-score", json=json_data, headers=headers)
        logger.info(f"Final score response: {response.text}")
        return response


async def apply_patch(sandbox: Sandbox, patch_path: str) -> str:
    GIT_APPLY_CMDS = [
        "git apply --verbose",
        "git apply --verbose --reject",
        "patch --batch --fuzz=5 -p1 -i",
    ]

    for git_apply_cmd in GIT_APPLY_CMDS:
        result: ExecResult = await sandbox.exec(
            f"{git_apply_cmd} {patch_path}",
            cwd="/testbed",
        )

        if result.exit_code == 0:
            return result.stdout
        else:
            logger.warning(f"Failed to apply patch command `{git_apply_cmd}`:{result.stdout}")

    raise ValueError(f"Failed to apply patch `{patch_path}`")
