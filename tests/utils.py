import asyncio
import json
import logging
import os
from asyncio import Task
from collections.abc import AsyncGenerator, Mapping
from typing import Any

from daytona import AsyncSandbox, ExecuteResponse
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
        self, task_ids: list[str] | None = None, slice_str: str | None = None
    ) -> Response:
        """
        Requests verify task ids from benchmark service
        """
        params: dict[str, Any] = {}
        if task_ids is not None:
            params["task_ids"] = task_ids
        if slice_str is not None:
            params["slice"] = slice_str

        response = self._client.get("/verify-task-ids", params=params)
        logger.info(f"Verify task ids response: {response.text}")
        return response

    async def request_retrieve_task(self, task_id: str, skip_validation: bool = False) -> Response:
        """
        Requests retrieve task from benchmark service
        """
        params = {"task_id": task_id, "skip_validation": str(skip_validation)}
        response = self._client.get("/retrieve-task/", params=params)
        logger.info(f"Retrieve task response: {response.text}")
        return response

    async def request_setup_task(self, task_id: str, instance_id: str) -> AsyncGenerator[str | dict[str, Any], None]:
        """
        Requests setup task from benchmark service via WebSocket
        """
        api_key = os.getenv("DAYTONA_API_KEY")
        api_url = os.getenv("DAYTONA_API_URL")
        target = os.getenv("DAYTONA_TARGET")

        if not api_key or not api_url or not target:
            raise ValueError("API key, API URL, and target are required")

        json_data = {
            "task_id": task_id,
            "instance_id": instance_id,
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
        self, task_id: str, instance_id: str
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        """
        Requests evaluate instance from benchmark service via WebSocket
        """
        api_key = os.getenv("DAYTONA_API_KEY")
        api_url = os.getenv("DAYTONA_API_URL")
        target = os.getenv("DAYTONA_TARGET")

        if not api_key or not api_url or not target:
            raise Exception("API key, API URL, and target are required")

        json_data = {
            "task_id": task_id,
            "instance_id": instance_id,
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

    async def request_final_score(self, evaluation_results: Mapping[str, dict[str, Any] | None]) -> Response:
        """
        Requests final score from benchmark service
        """
        json_data = {"evaluation_results": evaluation_results}
        headers = {"Content-Type": "application/json"}

        response = self._client.post("/final-score", json=json_data, headers=headers)
        logger.info(f"Final score response: {response.text}")
        return response


async def get_session_logger(sandbox: AsyncSandbox, session_id: str, cmd_id: str, logger: logging.Logger) -> Task[None]:
    """Creates a new task that will log the stdout and stderr of the command to the logger"""

    def log_stdout(stdout: str) -> None:
        if stdout.strip():
            logger.debug(f"[STDOUT]: {stdout.rstrip()}")

    def log_stderr(stderr: str) -> None:
        if stderr.strip():
            logger.error(f"[STDERR]: {stderr.rstrip()}")

    log_task = asyncio.create_task(
        sandbox.process.get_session_command_logs_async(
            session_id,
            cmd_id,
            log_stdout,
            log_stderr,
        )
    )

    return log_task


async def apply_patch(sandbox: AsyncSandbox, patch_path: str) -> str:
    GIT_APPLY_CMDS = [
        "git apply --verbose",
        "git apply --verbose --reject",
        "patch --batch --fuzz=5 -p1 -i",
    ]

    for git_apply_cmd in GIT_APPLY_CMDS:
        result: ExecuteResponse = await sandbox.process.exec(
            command=f"{git_apply_cmd} {patch_path}",
            cwd="/testbed",
        )

        if result.exit_code == 0:
            return result.result
        else:
            logger.warning(f"Failed to apply patch command `{git_apply_cmd}`:{result.result}")

    raise ValueError(f"Failed to apply patch `{patch_path}`")
