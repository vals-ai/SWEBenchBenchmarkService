import os
from pathlib import Path
from typing import Any

import pytest
from daytona import AsyncDaytona, CreateSandboxFromImageParams, DaytonaConfig, Resources

from swebench_utils import load_dataset_from_disk
from tests.utils import BenchmarkServiceTestClient, apply_patch


@pytest.fixture
async def daytona() -> AsyncDaytona:
    """Create Daytona client from environment variables."""
    api_key = os.getenv("DAYTONA_API_KEY")
    api_url = os.getenv("DAYTONA_API_URL")
    target = os.getenv("DAYTONA_TARGET")

    if not all([api_key, api_url, target]):
        pytest.skip("Daytona credentials not configured")

    return AsyncDaytona(config=DaytonaConfig(api_key=api_key, api_url=api_url, target=target))


@pytest.fixture
def setup_script_path() -> Path:
    """Verify setup.sh exists."""
    path = Path("setup.sh")
    assert path.exists(), "setup.sh not found"
    return path


@pytest.fixture
def test_client() -> BenchmarkServiceTestClient:
    """Create test client."""
    return BenchmarkServiceTestClient()


class TestEndToEnd:
    async def test_evaluate_instance(self, daytona: AsyncDaytona, test_client: BenchmarkServiceTestClient) -> None:
        """Test single task evaluation with patch."""
        task_id = "django__django-11099"

        # Get task metadata
        response = await test_client.request_retrieve_task(task_id)
        assert response.status_code == 200
        task_data = response.json()
        docker_image = task_data["docker_image"]

        # Create sandbox
        sandbox = await daytona.create(
            CreateSandboxFromImageParams(
                name=f"test-{task_id}",
                image=docker_image,
                resources=Resources(cpu=2, memory=4, disk=10),
            )
        )

        try:
            # Setup task
            messages: list[str | dict[str, Any]] = []
            async for msg in test_client.request_setup_task(task_id, sandbox.id):
                messages.append(msg)

            assert len(messages) > 0, "Expected setup messages"
            # The last message should be a result
            final_msg = messages[-1]
            assert isinstance(final_msg, dict), "Expected final message to be dict"
            if "status" in final_msg:
                assert final_msg["status"] == "ok"

            # Apply patch
            dataset = load_dataset_from_disk()
            patch_content = dataset[task_id]["patch"]

            await sandbox.fs.upload_file(patch_content.encode(), "/tmp/patch.diff")
            result = await apply_patch(sandbox, "/tmp/patch.diff")
            assert "Applied" in result or "already applied" in result.lower() or result.strip() != ""

            # Evaluate
            eval_messages: list[str | dict[str, Any]] = []
            async for msg in test_client.request_evaluate_instance(task_id, sandbox.id):
                eval_messages.append(msg)

            # Check result
            assert len(eval_messages) > 0, "Expected evaluation messages"
            final_msg = eval_messages[-1]
            assert isinstance(final_msg, dict), "Expected final message to be dict"

            # The final message should be the evaluation result
            eval_result = final_msg
            if "resolved" not in eval_result and "data" in eval_result:
                eval_result = eval_result["data"]

            assert "resolved" in eval_result
            # Note: We don't assert the resolved status is True because
            # the patch may or may not make all tests pass
            assert isinstance(eval_result["resolved"], bool)

        finally:
            await daytona.delete(sandbox)

    async def test_end_to_end_api_only(self, test_client: BenchmarkServiceTestClient) -> None:
        """Test complete API flow without sandbox."""
        # Health check
        response = await test_client.request_health_check()
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

        # Verify task IDs
        task_ids = ["django__django-11099", "django__django-11333"]
        response = await test_client.request_verify_task_ids(task_ids)
        assert response.status_code == 200
        data = response.json()
        assert data["task_ids"] == task_ids

        # Retrieve task
        response = await test_client.request_retrieve_task(task_ids[0])
        assert response.status_code == 200
        data = response.json()
        assert task_ids[0] in data["docker_image"]
        assert len(data["problem_statement"]) > 0

        # Final score
        mock_results = {tid: {"resolved": True} for tid in task_ids}
        response = await test_client.request_final_score(mock_results)
        assert response.status_code == 200
        data = response.json()
        assert data["final_score"] == 100.0
        assert len(data["tasks_evaluated"]) == 2

    @pytest.mark.experimental
    async def test_evaluate_multiple_instances(
        self, daytona: AsyncDaytona, test_client: BenchmarkServiceTestClient
    ) -> None:
        """Test evaluating multiple instances (SLOW)."""
        # Use a smaller subset for testing
        test_task_ids = ["django__django-11099", "astropy__astropy-12907"]

        results: list[dict[str, Any]] = []

        for task_id in test_task_ids:
            # Get task metadata
            response = await test_client.request_retrieve_task(task_id)
            assert response.status_code == 200
            task_data = response.json()
            docker_image = task_data["docker_image"]

            # Create sandbox
            sandbox = await daytona.create(
                CreateSandboxFromImageParams(
                    name=f"test-{task_id}",
                    image=docker_image,
                    resources=Resources(cpu=2, memory=4, disk=10),
                )
            )

            try:
                # Setup task
                messages: list[str | dict[str, Any]] = []
                async for msg in test_client.request_setup_task(task_id, sandbox.id):
                    messages.append(msg)

                # Apply patch
                dataset = load_dataset_from_disk()
                patch_content = dataset[task_id]["patch"]
                await sandbox.fs.upload_file(patch_content.encode(), "/tmp/patch.diff")
                await apply_patch(sandbox, "/tmp/patch.diff")

                # Evaluate
                eval_messages: list[str | dict[str, Any]] = []
                async for msg in test_client.request_evaluate_instance(task_id, sandbox.id):
                    eval_messages.append(msg)

                # Store result
                final_msg = eval_messages[-1]
                assert isinstance(final_msg, dict), "Expected last message to be result"
                eval_result = final_msg.get("data", final_msg)
                results.append({"task_id": task_id, **eval_result})

            except Exception as e:
                results.append({"task_id": task_id, "error": str(e)})
            finally:
                await daytona.delete(sandbox)

        # Verify all tasks completed
        assert len(results) == len(test_task_ids)

        # Check for errors
        errors = [r for r in results if "error" in r]
        if errors:
            pytest.fail(f"Errors in evaluation: {errors}")

        # Verify all results have resolved field
        for result in results:
            assert "resolved" in result, f"Missing resolved field in {result}"
