import asyncio
import logging
import os
from asyncio import Task
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from daytona import AsyncDaytona, AsyncSandbox, CreateSandboxFromImageParams, Resources
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.logger import get_logger
from src.models import EvaluationResult

logger = get_logger(__name__)


class BenchmarkServiceTestClient:
    """
    Contains utilities for testing the benchmark service
    """

    _client: TestClient
    _BASE_URL: str = "http://localhost:8000"

    def __init__(self, app: FastAPI) -> None:
        self._client = TestClient(app)

    async def request_health_check(self) -> dict[str, str]:
        """
        Requests health check from benchmark service
        """
        async with AsyncClient(base_url=self._BASE_URL) as client:
            response = await client.get("/health")

        logger.info(f"Health check response: {response.text}")

        if response.status_code != 200:
            raise Exception(f"Health check failed with status code {response.status_code}, response: {response.text}")

        return response.json()

    async def request_verify_task_ids(self, task_ids: list[str]) -> dict[str, list[str]]:
        """
        Requests verify task ids from benchmark service
        """

        params = {"task_ids": task_ids}

        async with AsyncClient(base_url=self._BASE_URL) as client:
            response = await client.get("/verify-task-ids", params=params)

        logger.info(f"Verify task ids response: {response.text}")

        if response.status_code != 200:
            raise Exception(
                f"Verify task ids failed with status code {response.status_code}, response: {response.text}"
            )

        return response.json()

    async def request_retrieve_task(self, task_id: str, skip_validation: bool = False) -> dict[str, str]:
        """
        Requests retrieve task from benchmark service
        """

        params = {"task_id": task_id, "skip_validation": str(skip_validation)}

        async with AsyncClient(base_url=self._BASE_URL) as client:
            response = await client.get("/retrieve-task/", params=params)

        logger.info(f"Retrieve task response: {response.text}")

        if response.status_code != 200:
            raise Exception(f"Retrieve task failed with status code {response.status_code}, response: {response.text}")

        return response.json()

    async def request_setup_task(self, task_id: str, instance_id: str) -> dict[str, str]:
        """
        Requests setup task from benchmark service
        """

        api_key = os.getenv("DAYTONA_API_KEY")
        api_url = os.getenv("DAYTONA_API_URL")
        target = os.getenv("DAYTONA_TARGET")

        if not api_key or not api_url or not target:
            raise ValueError("API key, API URL, and target are required")

        json_data = {"task_id": task_id, "instance_id": instance_id}
        headers = {"X-Api-Key": api_key, "X-Api-Url": api_url, "X-Target": target}

        async with AsyncClient(base_url=self._BASE_URL) as client:
            response = await client.post("/setup-task", json=json_data, headers=headers)

        logger.info(f"Setup task response: {response.text}")

        return response.json()

    async def request_evaluate_instance(self, task_id: str, instance_id: str) -> dict[str, str]:
        """
        Requests evaluate instance from benchmark service
        """

        api_key = os.getenv("DAYTONA_API_KEY")
        api_url = os.getenv("DAYTONA_API_URL")
        target = os.getenv("DAYTONA_TARGET")

        if not api_key or not api_url or not target:
            raise Exception("API key, API URL, and target are required")

        json_data = {"task_id": task_id, "instance_id": instance_id}
        headers = {"X-Api-Key": api_key, "X-Api-Url": api_url, "X-Target": target}

        async with AsyncClient(base_url=self._BASE_URL) as client:
            response = await client.post("/evaluate-instance/", json=json_data, headers=headers)

        logger.info(f"Evaluate instance response: {response.text}")

        if response.status_code != 200:
            raise Exception(
                f"Evaluate instance failed with status code {response.status_code}, response: {response.text}"
            )

        return response.json()

    async def request_final_score(self, evaluation_results: dict[str, EvaluationResult]) -> dict[str, Any]:
        """
        Requests final score from benchmark service
        """

        json_data = {"evaluation_results": evaluation_results}
        headers = {"Content-Type": "application/json"}

        async with AsyncClient(base_url=self._BASE_URL) as client:
            response = await client.post("/final-score", json=json_data, headers=headers)

        logger.info(f"Final score response: {response.text}")

        if response.status_code != 200:
            raise Exception(f"Final score failed with status code {response.status_code}, response: {response.text}")

        return response.json()


@asynccontextmanager
async def build_task_environment(
    daytona: AsyncDaytona,
    task_id: str,
    docker_image: str,
) -> AsyncIterator[AsyncSandbox]:
    """
    Builds the task environment using the dockerfile path
    """

    sandbox = await daytona.create(
        CreateSandboxFromImageParams(
            env_vars={"TEST_DIR": "/tests"},
            image=docker_image,
            name=task_id,
            network_block_all=False,
            resources=Resources(cpu=4, memory=8, disk=10),
        ),
        timeout=360,
    )

    await sandbox.process.create_session(sandbox.id)

    try:
        yield sandbox
    finally:
        await daytona.delete(sandbox)
        pass


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
