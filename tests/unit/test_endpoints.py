import pytest

from swebench_utils import load_dataset_from_disk
from tests.utils import BenchmarkServiceTestClient


class TestEndpoints:
    @pytest.fixture
    def client(self) -> BenchmarkServiceTestClient:
        return BenchmarkServiceTestClient()

    async def test_health_check(self, client: BenchmarkServiceTestClient) -> None:
        """Test /health endpoint."""
        response = await client.request_health_check()
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    async def test_verify_task_ids_valid(self, client: BenchmarkServiceTestClient) -> None:
        """Test /verify-task-ids with valid task IDs."""
        response = await client.request_verify_task_ids(["django__django-11099", "django__django-11333"])
        assert response.status_code == 200
        data = response.json()
        assert "task_ids" in data
        assert len(data["task_ids"]) == 2
        assert "django__django-11099" in data["task_ids"]
        assert "django__django-11333" in data["task_ids"]

    async def test_verify_task_ids_invalid(self, client: BenchmarkServiceTestClient) -> None:
        """Test /verify-task-ids with invalid task ID."""
        response = await client.request_verify_task_ids(["invalid-task-id"])
        assert response.status_code == 500  # Should fail validation

    async def test_verify_task_ids_all(self, client: BenchmarkServiceTestClient) -> None:
        """Test /verify-task-ids without task_ids parameter returns all tasks."""
        response = await client.request_verify_task_ids()
        assert response.status_code == 200
        data = response.json()
        assert "task_ids" in data

        dataset_map = load_dataset_from_disk()
        expected_task_ids = list(dataset_map.keys())
        assert data["task_ids"] == expected_task_ids

    async def test_verify_task_ids_slice_start_stop_step(self, client: BenchmarkServiceTestClient) -> None:
        """Test /verify-task-ids with slice notation (start:stop:step)."""
        response = await client.request_verify_task_ids(slice_str="3:10:1")
        assert response.status_code == 200
        data = response.json()
        assert "task_ids" in data
        assert len(data["task_ids"]) == 7  # 10 - 3 = 7

        dataset_map = load_dataset_from_disk()
        all_task_ids = list(dataset_map.keys())
        expected_task_ids = all_task_ids[3:10:1]
        assert data["task_ids"] == expected_task_ids

    async def test_verify_task_ids_slice_start_stop(self, client: BenchmarkServiceTestClient) -> None:
        """Test /verify-task-ids with slice notation (start:stop)."""
        response = await client.request_verify_task_ids(slice_str="3:10")
        assert response.status_code == 200
        data = response.json()
        assert "task_ids" in data
        assert len(data["task_ids"]) == 7

        dataset_map = load_dataset_from_disk()
        all_task_ids = list(dataset_map.keys())
        expected_task_ids = all_task_ids[3:10]
        assert data["task_ids"] == expected_task_ids

    async def test_verify_task_ids_slice_start(self, client: BenchmarkServiceTestClient) -> None:
        """Test /verify-task-ids with slice notation (start:)."""
        response = await client.request_verify_task_ids(slice_str="3:")
        assert response.status_code == 200
        data = response.json()
        assert "task_ids" in data

        dataset_map = load_dataset_from_disk()
        all_task_ids = list(dataset_map.keys())
        expected_task_ids = all_task_ids[3:]
        assert data["task_ids"] == expected_task_ids

    async def test_verify_task_ids_slice_stop(self, client: BenchmarkServiceTestClient) -> None:
        """Test /verify-task-ids with slice notation (:stop)."""
        response = await client.request_verify_task_ids(slice_str=":10")
        assert response.status_code == 200
        data = response.json()
        assert "task_ids" in data
        assert len(data["task_ids"]) == 10

        dataset_map = load_dataset_from_disk()
        all_task_ids = list(dataset_map.keys())
        expected_task_ids = all_task_ids[:10]
        assert data["task_ids"] == expected_task_ids

    async def test_retrieve_task(self, client: BenchmarkServiceTestClient) -> None:
        """Test /retrieve-task/ endpoint."""
        task_id = "django__django-11099"
        response = await client.request_retrieve_task(task_id)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "docker_image" in data
        assert task_id in data["docker_image"]
        assert data["docker_image"] == f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{task_id}:latest"

        assert "problem_statement" in data
        assert len(data["problem_statement"]) > 0

        assert "request_setup" in data
        assert data["request_setup"] is True

        assert "cwd" in data
        assert data["cwd"] == "/testbed"

        assert "resources" in data
        assert data["resources"]["vcpu"] >= 2
        assert data["resources"]["memory"] >= 4
        assert data["resources"]["disk"] >= 10

    async def test_retrieve_task_invalid(self, client: BenchmarkServiceTestClient) -> None:
        """Test /retrieve-task/ with invalid task ID."""
        response = await client.request_retrieve_task("invalid-task-id")
        assert response.status_code == 500

    async def test_retrieve_task_multiple(self, client: BenchmarkServiceTestClient) -> None:
        """Test /retrieve-task/ with multiple different tasks."""
        task_ids = ["django__django-11099", "astropy__astropy-12907"]

        for task_id in task_ids:
            response = await client.request_retrieve_task(task_id)
            assert response.status_code == 200

            data = response.json()
            assert task_id in data["docker_image"]
            assert len(data["problem_statement"]) > 0

    async def test_final_score(self, client: BenchmarkServiceTestClient) -> None:
        """Test /final-score endpoint."""
        # Use real task IDs from the dataset
        dataset = load_dataset_from_disk()
        task_ids = list(dataset.keys())[:3]

        evaluation_results = {
            task_ids[0]: {"resolved": True},
            task_ids[1]: {"resolved": False},
            task_ids[2]: {"resolved": True},
        }

        response = await client.request_final_score(evaluation_results)
        assert response.status_code == 200

        data = response.json()
        assert "final_score" in data
        assert data["final_score"] == pytest.approx(66.666666, rel=1e-5)  # type: ignore[reportUnknownMemberType]

        assert "tasks_evaluated" in data
        assert len(data["tasks_evaluated"]) == 3

        assert "metadata" in data
        assert "resolved_tasks" in data["metadata"]
        assert "unresolved_tasks" in data["metadata"]
        assert len(data["metadata"]["resolved_tasks"]) == 2
        assert len(data["metadata"]["unresolved_tasks"]) == 1

    async def test_final_score_all_resolved(self, client: BenchmarkServiceTestClient) -> None:
        """Test /final-score with all tasks resolved."""
        # Use real task IDs from the dataset
        dataset = load_dataset_from_disk()
        task_ids = list(dataset.keys())[:2]

        evaluation_results = {
            task_ids[0]: {"resolved": True},
            task_ids[1]: {"resolved": True},
        }

        response = await client.request_final_score(evaluation_results)
        assert response.status_code == 200

        data = response.json()
        assert data["final_score"] == 100.0
        assert len(data["metadata"]["resolved_tasks"]) == 2
        assert len(data["metadata"]["unresolved_tasks"]) == 0

    async def test_final_score_none_resolved(self, client: BenchmarkServiceTestClient) -> None:
        """Test /final-score with no tasks resolved."""
        # Use real task IDs from the dataset
        dataset = load_dataset_from_disk()
        task_ids = list(dataset.keys())[:2]

        evaluation_results = {
            task_ids[0]: {"resolved": False},
            task_ids[1]: None,
        }

        response = await client.request_final_score(evaluation_results)
        assert response.status_code == 200

        data = response.json()
        assert data["final_score"] == 0.0
        assert len(data["metadata"]["resolved_tasks"]) == 0
        assert len(data["metadata"]["unresolved_tasks"]) == 2

    async def test_final_score_empty(self, client: BenchmarkServiceTestClient) -> None:
        """Test /final-score with empty results."""
        evaluation_results: dict[str, dict[str, bool] | None] = {}

        response = await client.request_final_score(evaluation_results)
        assert response.status_code == 200

        data = response.json()
        assert data["final_score"] == 0.0
        assert len(data["tasks_evaluated"]) == 0
