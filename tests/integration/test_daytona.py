import asyncio
import json
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from daytona import (
    AsyncDaytona,
    AsyncSandbox,
    DaytonaConfig,
)
from dotenv import load_dotenv

from main import app
from src.logger import get_logger
from src.utils import TaskContext, create_sandbox, load_dataset_from_disk
from tests.utils import BenchmarkServiceTestClient, apply_patch

logger = get_logger(__name__)

load_dotenv()


@pytest_asyncio.fixture
async def daytona() -> AsyncGenerator[AsyncDaytona, None]:
    config = DaytonaConfig(
        api_key=os.getenv("DAYTONA_API_KEY"),
        api_url=os.getenv("DAYTONA_API_URL"),
        target=os.getenv("DAYTONA_TARGET"),
    )

    async with AsyncDaytona(config) as daytona:
        yield daytona


@pytest.fixture
def setup_script_path() -> Path:
    if not Path("setup.sh").exists():
        pytest.fail("Setup script `setup.sh` does not exist")

    return Path("setup.sh")


@pytest_asyncio.fixture
async def test_client() -> BenchmarkServiceTestClient:
    return BenchmarkServiceTestClient(app)


class TestDaytona:
    async def _insert_patch_and_evaluate(
        self,
        sandbox: AsyncSandbox,
        task_context: TaskContext,
        request_setup: bool,
        test_client: BenchmarkServiceTestClient,
    ) -> dict[str, Any]:
        # Use the additional_setup flag
        if request_setup:
            await test_client.request_setup_task(task_id=task_context.task_id, instance_id=sandbox.id)

        if not task_context.patch:
            raise Exception(f"Patch not found for task {task_context.task_id}")

        # Once that is done, we simulate an agent being copied into the sandbox and running by copying the solution patch inside
        await sandbox.fs.upload_file(
            task_context.patch.encode("utf-8"),
            "/tmp/patch.diff",
        )

        # Apply the solution patch to the testbed
        try:
            await apply_patch(sandbox, "/tmp/patch.diff")
        except Exception:
            raise Exception(f"Error applying solution patch for task {task_context.task_id}")

        # Request for evaluation
        evaluation_result = await test_client.request_evaluate_instance(
            task_id=task_context.task_id,
            instance_id=sandbox.id,
        )

        return evaluation_result

    @pytest.mark.experimental
    async def test_build_all_sandboxes(self, daytona: AsyncDaytona) -> None:
        dataset_map = load_dataset_from_disk()

        assert len(dataset_map) == 500, "Expected 500 tasks to be available"

        semaphore = asyncio.Semaphore(20)

        async def build_image(task_id: str) -> tuple[str, str, bool]:
            async with semaphore:
                task_context = TaskContext(task_id)

                try:
                    async with create_sandbox(daytona, task_id, task_context.docker_image) as _:
                        return task_id, "", True
                except Exception as e:
                    return task_id, str(e), False

        results: list[tuple[str, str, bool]] = await asyncio.gather(
            *[build_image(task_id=task_id) for task_id in dataset_map.keys()]
        )

        errors: list[str] = []
        for task_id, error, was_built in results:
            if not was_built:
                errors.append(f"Task `{task_id}`: {error}")

        assert len(errors) == 0, f"{' \n'.join(errors)}"

    async def test_evaluate_instance(
        self,
        daytona: AsyncDaytona,
        test_client: BenchmarkServiceTestClient,
    ) -> None:
        task_id = "astropy__astropy-12907"
        task_context = TaskContext(task_id)

        async with create_sandbox(daytona, task_id, task_context.docker_image) as sandbox:
            # Setup environment on base commit
            await test_client.request_setup_task(task_id=task_id, instance_id=sandbox.id)

            # Verify we're on the correct commit
            verify_result = await sandbox.process.exec(
                command="git rev-parse HEAD",
                cwd="/testbed",
            )

            if verify_result.exit_code != 0:
                pytest.fail(f"Error verifying commit: {verify_result.result}")

            actual_commit = verify_result.result.strip()
            assert actual_commit == task_context.base_commit, (
                f"Expected commit {task_context.base_commit} but got {actual_commit}"
            )

            # Insert the patch and evaluate the instance
            try:
                evaluation_result = await self._insert_patch_and_evaluate(sandbox, task_context, True, test_client)
            except Exception as e:
                pytest.fail(f"Error inserting patch and evaluating instance: {e}")

            # Verify the evaluation result
            assert evaluation_result, "Expected evaluation result"
            assert "resolved" in evaluation_result, "Expected 'resolved' field in evaluation result"
            assert evaluation_result["resolved"] is True, (
                f"Expected instance to be resolved. Result: {evaluation_result}"
            )
            assert "prediction" in evaluation_result, "Expected prediction result"

    async def test_end_to_end(
        self,
        daytona: AsyncDaytona,
        test_client: BenchmarkServiceTestClient,
    ) -> None:
        task_id = "astropy__astropy-12907"

        # Ensure service is running
        response = await test_client.request_health_check()
        assert response == {"status": "ok"}, "Expected health check to return ok"

        # Verify task ids are valid
        response = await test_client.request_verify_task_ids(task_ids=[task_id])
        assert response == {"task_ids": [task_id]}, "Expected task ids to be valid"

        # Retrieve task
        response = await test_client.request_retrieve_task(task_id=task_id)
        dataset_map = load_dataset_from_disk()

        problem_statement: str = dataset_map[task_id].get("problem_statement", "")
        assert response == {
            "docker_image": f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{task_id}:latest",
            "problem_statement": problem_statement,
            "request_setup": True,
            "cwd": "/testbed",
            "resources": {
                "vcpu": 2,
                "memory": 4,
                "disk": 10,
            },
        }, "Expected task to be retrieved"

        # Create sandbox from the provided docker image
        async with create_sandbox(daytona, task_id, response["docker_image"]) as sandbox:
            task_context: TaskContext = TaskContext(task_id)

            # Insert the patch and evaluate the instance
            try:
                evaluation_result = await self._insert_patch_and_evaluate(
                    sandbox, task_context, bool(response["request_setup"]), test_client
                )
            except Exception as e:
                pytest.fail(f"Error inserting patch and evaluating instance: {e}")

            # Verify the evaluation result
            assert evaluation_result, "Expected evaluation result"
            assert "resolved" in evaluation_result, "Expected 'resolved' field in evaluation result"
            assert evaluation_result["resolved"] is True, (
                f"Expected instance to be resolved. Result: {evaluation_result}"
            )

    @pytest.mark.experimental
    async def test_validate_all_images(
        self,
        daytona: AsyncDaytona,
        test_client: BenchmarkServiceTestClient,
    ) -> None:
        dataset_map = load_dataset_from_disk()
        task_ids: list[str] = list(dataset_map.keys())
        response = await test_client.request_health_check()
        assert response == {"status": "ok"}, "Expected health check to return ok"

        # Validate all tasks inside of dataset
        response = await test_client.request_verify_task_ids(task_ids=task_ids)
        assert response == {"task_ids": task_ids}, "Expected task ids to be valid"

        # cap amount of concurrent evaluations to 15
        semaphore = asyncio.Semaphore(15)

        async def start_and_evaluate_instance(task_id: str) -> dict[str, Any]:
            try:
                async with semaphore:
                    task_response = await test_client.request_retrieve_task(task_id=task_id, skip_validation=True)
                    async with create_sandbox(daytona, task_id, task_response["docker_image"]) as sandbox:
                        task_context = TaskContext(task_id)

                        return await self._insert_patch_and_evaluate(
                            sandbox, task_context, bool(task_response["request_setup"]), test_client
                        )
            except Exception as e:
                return {"task_id": task_id, "error": str(e)}

        results: list[dict[str, Any]] = await asyncio.gather(
            *[start_and_evaluate_instance(task_id) for task_id in task_ids]
        )

        # Large output so we store it to a file for easy viewing
        if not Path("tests/test_outputs").exists():
            Path("tests/test_outputs").mkdir(parents=True, exist_ok=True)

        with open("tests/test_outputs/results.json", "w") as f:
            json.dump(results, f, indent=4)

        assert len(results) == len(task_ids), "Expected all results to be returned"

        # No instances should have exit errors
        errors: list[str] = []
        for result in results:
            if "error" in result:
                errors.append(f"Task `{result['task_id']}`: {result['error']}")

        assert len(errors) == 0, f"{' \n'.join(errors)}"

        # All instances should be resolved
        not_resolved: list[str] = []
        for result in results:
            if "resolved" in result and not result["resolved"]:
                not_resolved.append(f"Task `{result['task_id']}`: {result['resolution_status']}")

        assert len(not_resolved) == 0, f"{' \n'.join(not_resolved)}"
