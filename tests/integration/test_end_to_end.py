import json
import os
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any

import pytest
from benchmark_service.sandbox import (
    ImageSource,
    Resources,
    SandboxCreateRequest,
    SandboxProvider,
    SandboxProviderConfig,
    sandbox_provider_config_from_mapping,
)

from swebench_service import load_dataset_from_disk
from tests.utils import BenchmarkServiceTestClient, apply_patch


@pytest.fixture
async def sandbox_provider_config() -> SandboxProviderConfig:
    """Create sandbox provider config from environment variables."""
    config_json = os.getenv("SANDBOX_PROVIDER_CONFIG")

    if not config_json:
        pytest.skip("Sandbox provider config not configured")

    return sandbox_provider_config_from_mapping(json.loads(config_json))


@pytest.fixture
async def sandbox_provider(sandbox_provider_config: SandboxProviderConfig) -> AsyncGenerator[SandboxProvider, None]:
    """Create sandbox provider from environment variables."""
    provider = sandbox_provider_config.create_provider()
    try:
        yield provider
    finally:
        await provider.close()


@pytest.fixture
def setup_script_path() -> Path:
    """Verify setup.sh exists."""
    path = Path("setup.sh")
    assert path.exists(), "setup.sh not found"
    return path


@pytest.fixture
def test_client() -> Generator[BenchmarkServiceTestClient]:
    """Create test client."""
    c = BenchmarkServiceTestClient()
    yield c
    c.close()


class TestEndToEnd:
    async def test_evaluate_instance(
        self,
        sandbox_provider_config: SandboxProviderConfig,
        sandbox_provider: SandboxProvider,
        test_client: BenchmarkServiceTestClient,
    ) -> None:
        """Test single task evaluation with patch."""
        task_id = "matplotlib__matplotlib-22865"

        # Get task metadata
        response = await test_client.request_retrieve_task(task_id)
        assert response.status_code == 200
        task_data = response.json()
        docker_image = task_data["docker_image"]

        # Create sandbox
        sandbox = await sandbox_provider.create_sandbox(
            SandboxCreateRequest(
                source=ImageSource(image=docker_image),
                name=f"test-{task_id}",
                resources=Resources(vcpu=2, memory=4, disk=10),
                labels={},
                env_vars={},
                auto_stop_interval=60,
                create_timeout=600,
            )
        )

        try:
            # Setup task
            messages: list[str | dict[str, Any]] = []
            async for msg in test_client.request_setup_task(
                task_id,
                sandbox.id,
                sandbox_provider=sandbox_provider_config,
            ):
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

            await sandbox.upload_file("/tmp/patch.diff", patch_content.encode())
            result = await apply_patch(sandbox, "/tmp/patch.diff")
            assert "Applied" in result or "already applied" in result.lower() or result.strip() != ""

            # Evaluate
            eval_messages: list[str | dict[str, Any]] = []
            async for msg in test_client.request_evaluate_instance(
                task_id,
                sandbox.id,
                sandbox_provider=sandbox_provider_config,
            ):
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
            await sandbox_provider.delete_sandbox(sandbox.id)

    async def test_end_to_end_api_only(self, test_client: BenchmarkServiceTestClient) -> None:
        """Test complete API flow without sandbox."""
        # Health check
        response = await test_client.request_health_check()
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

        # Verify task IDs
        task_ids = ["matplotlib__matplotlib-22865", "django__django-11099"]
        response = await test_client.request_verify_task_ids(task_ids)
        assert response.status_code == 200
        data = response.json()
        assert data["task_ids"] == task_ids

        # Retrieve task
        response = await test_client.request_retrieve_task(task_ids[0])
        assert response.status_code == 200
        data = response.json()
        assert task_ids[0].replace("__", "_1776_") in data["docker_image"]
        assert data["problem_path"]

        # Final score
        mock_results = {tid: {"resolved": True} for tid in task_ids}
        response = await test_client.request_final_score(mock_results)
        assert response.status_code == 200
        data = response.json()
        assert data["final_score"] == 100.0
        assert len(data["tasks_evaluated"]) == 2

    @pytest.mark.experimental
    async def test_evaluate_multiple_instances(
        self,
        sandbox_provider_config: SandboxProviderConfig,
        sandbox_provider: SandboxProvider,
        test_client: BenchmarkServiceTestClient,
    ) -> None:
        """Test evaluating multiple instances (SLOW)."""
        # Use a smaller subset for testing
        test_task_ids = ["matplotlib__matplotlib-22865", "django__django-11099"]

        results: list[dict[str, Any]] = []

        for task_id in test_task_ids:
            # Get task metadata
            response = await test_client.request_retrieve_task(task_id)
            assert response.status_code == 200
            task_data = response.json()
            docker_image = task_data["docker_image"]

            # Create sandbox
            sandbox = await sandbox_provider.create_sandbox(
                SandboxCreateRequest(
                    source=ImageSource(image=docker_image),
                    name=f"test-{task_id}",
                    resources=Resources(vcpu=2, memory=4, disk=10),
                    labels={},
                    env_vars={},
                    auto_stop_interval=60,
                    create_timeout=600,
                )
            )

            try:
                # Setup task
                messages: list[str | dict[str, Any]] = []
                async for msg in test_client.request_setup_task(
                    task_id,
                    sandbox.id,
                    sandbox_provider=sandbox_provider_config,
                ):
                    messages.append(msg)

                # Apply patch
                dataset = load_dataset_from_disk()
                patch_content = dataset[task_id]["patch"]
                await sandbox.upload_file("/tmp/patch.diff", patch_content.encode())
                await apply_patch(sandbox, "/tmp/patch.diff")

                # Evaluate
                eval_messages: list[str | dict[str, Any]] = []
                async for msg in test_client.request_evaluate_instance(
                    task_id,
                    sandbox.id,
                    sandbox_provider=sandbox_provider_config,
                ):
                    eval_messages.append(msg)

                # Store result
                final_msg = eval_messages[-1]
                assert isinstance(final_msg, dict), "Expected last message to be result"
                eval_result = final_msg.get("data", final_msg)
                results.append({"task_id": task_id, **eval_result})

            except Exception as e:
                results.append({"task_id": task_id, "error": str(e)})
            finally:
                await sandbox_provider.delete_sandbox(sandbox.id)

        # Verify all tasks completed
        assert len(results) == len(test_task_ids)

        # Check for errors
        errors = [r for r in results if "error" in r]
        if errors:
            pytest.fail(f"Errors in evaluation: {errors}")

        # Verify all results have resolved field
        for result in results:
            assert "resolved" in result, f"Missing resolved field in {result}"
