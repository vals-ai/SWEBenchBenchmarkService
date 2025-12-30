import asyncio
import json
import logging
import os
from asyncio import Task
from contextlib import asynccontextmanager
from typing import AsyncIterator

from daytona import AsyncDaytona, AsyncSandbox, CreateSandboxFromImageParams, Image, Resources
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.logger import get_logger

logger = get_logger(__name__)


class BenchmarkServiceTestClient:
    """
    Contains utilities for testing the benchmark service
    """

    _daytona: AsyncDaytona
    _client: TestClient
    _sandbox: AsyncSandbox

    def __init__(self, app: FastAPI, daytona: AsyncDaytona, sandbox: AsyncSandbox) -> None:
        self._daytona = daytona
        self._client = TestClient(app)
        self._sandbox = sandbox

    async def request_health_check(self) -> dict[str, str]:
        """
        Requests health check from benchmark service
        """
        response = await self._sandbox.process.exec(
            command="curl -s -X GET http://localhost:8000/health",
        )

        logger.info(f"Health check response: {response.result}")

        if response.exit_code != 0:
            raise Exception(
                f"Health check failed with exit code {response.exit_code}, command error: {response.result}"
            )

        return json.loads(response.result or "{}")

    async def request_verify_task_ids(self, task_ids: list[str]) -> dict[str, list[str]]:
        """
        Requests verify task ids from benchmark service
        """

        query_params = "&".join([f"task_ids={task_id}" for task_id in task_ids])
        response = await self._sandbox.process.exec(
            f"curl -s -X GET 'http://localhost:8000/verify-task-ids?{query_params}'",
        )

        logger.info(f"Verify task ids response: {response.result}")

        if response.exit_code != 0:
            raise Exception(
                f"Verify task ids failed with exit code {response.exit_code}, command error: {response.result}"
            )

        return json.loads(response.result or "{}")

    async def request_retrieve_tasks(self, task_ids: list[str]) -> dict[str, dict[str, str]]:
        """
        Requests retrieve tasks from benchmark service
        """

        query_params = "&".join([f"task_ids={task_id}" for task_id in task_ids])
        response = await self._sandbox.process.exec(
            f"curl -s -X GET 'http://localhost:8000/retrieve-tasks?{query_params}'",
        )

        logger.info(f"Retrieve tasks response: {response.result}")

        if response.exit_code != 0:
            raise Exception(
                f"Retrieve tasks failed with exit code {response.exit_code}, command error: {response.result}"
            )

        return json.loads(response.result or "{}")

    async def request_setup_task(self, task_id: str, instance_id: str) -> dict[str, str]:
        """
        Requests setup task from benchmark service
        """

        api_key = os.getenv("DAYTONA_API_KEY")
        api_url = os.getenv("DAYTONA_API_URL")
        target = os.getenv("DAYTONA_TARGET")

        if not api_key or not api_url or not target:
            raise ValueError("API key, API URL, and target are required")

        response = await self._sandbox.process.exec(
            f"curl -s -X POST http://localhost:8000/setup-task -H 'Content-Type: application/json' -H 'X-Api-Key: {api_key}' -H 'X-Api-Url: {api_url}' -H 'X-Target: {target}' -d '{{\"task_id\": \"{task_id}\", \"instance_id\": \"{instance_id}\"}}'",
        )

        logger.info(f"Setup task response: {response.result}")

        if response.exit_code != 0:
            raise Exception(f"Setup task failed with exit code {response.exit_code}, command error: {response.result}")

        return json.loads(response.result or "{}")

    async def request_evaluate_instance(self, task_id: str, instance_id: str) -> dict[str, str]:
        """
        Requests evaluate instance from benchmark service
        """

        api_key = os.getenv("DAYTONA_API_KEY")
        api_url = os.getenv("DAYTONA_API_URL")
        target = os.getenv("DAYTONA_TARGET")

        if not api_key or not api_url or not target:
            raise Exception("API key, API URL, and target are required")

        response = await self._sandbox.process.exec(
            f"curl -s -X POST http://localhost:8000/evaluate-instance/ -H 'Content-Type: application/json' -H 'X-Api-Key: {api_key}' -H 'X-Api-Url: {api_url}' -H 'X-Target: {target}' -d '{{\"task_id\": \"{task_id}\", \"instance_id\": \"{instance_id}\"}}'",
        )

        logger.info(f"Evaluate instance response: {response.result}")

        if response.exit_code != 0:
            raise Exception(
                f"Evaluate instance failed with exit code {response.exit_code}, command error: {response.result}"
            )

        return json.loads(response.result or "{}")


@asynccontextmanager
async def build_task_environment(
    daytona: AsyncDaytona,
    task_id: str,
    dockerfile_path: str,
) -> AsyncIterator[AsyncSandbox]:
    """
    Builds the task environment using the dockerfile path
    """

    task_image = Image.base(dockerfile_path)
    sandbox = await daytona.create(
        CreateSandboxFromImageParams(
            env_vars={"TEST_DIR": "/tests"},
            image=task_image,
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
