from pathlib import Path

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from main import app
from src.utils import load_dataset_from_disk

client = TestClient(app)


class TestFastApiServer:
    async def test_health_check(self) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    async def test_verify_task_ids(self, setup_dataset: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.utils._DISK_PATH", setup_dataset)

        # Contains valid task ids
        response = client.get(
            "/verify-task-ids", params={"task_ids": ["astropy__astropy-12907", "django__django-12050"]}
        )

        assert response.status_code == 200
        assert response.json() == {"task_ids": ["astropy__astropy-12907", "django__django-12050"]}

        # Contains invalid task ids
        response = client.get("/verify-task-ids", params={"task_ids": ["astropy__astropy-12907", "invalid-task-id"]})

        assert response.status_code == 500

        # When no task ids are provided, all tasks are verified
        response = client.get("/verify-task-ids")

        assert response.status_code == 200
        assert response.json() == {"task_ids": [row["instance_id"] for row in load_dataset_from_disk()]}  # type: ignore

    async def test_retrieve_tasks(self, setup_dataset: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.utils._DISK_PATH", setup_dataset)

        registry_image_format = "ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"

        valid_task_ids: list[str] = ["astropy__astropy-12907", "django__django-12050"]

        # Contains valid task ids
        response = client.get("/retrieve-tasks", params={"task_ids": valid_task_ids})

        dataset = load_dataset_from_disk()
        problem_statements: dict[str, str] = {}
        for task_id in valid_task_ids:
            problem_statements[task_id] = dataset.filter(lambda x: x["instance_id"] == task_id)[0].get(  # type: ignore
                "problem_statement", ""
            )

        assert response.status_code == 200

        expected_response = {
            task_id: {
                "docker_image": registry_image_format.format(instance_id=task_id),
                "problem_statement": problem_statements[task_id],
                "request_setup": True,
            }
            for task_id in valid_task_ids
        }

        assert response.json() == expected_response

        # Contains invalid task ids
        response = client.get("/retrieve-tasks", params={"task_ids": ["astropy__astropy-12907", "invalid-task-id"]})

        assert response.status_code == 500

        # All tasks are valid
        task_ids: list[str] = [row["instance_id"] for row in dataset]  # type: ignore

        assert len(task_ids) == 500, "Expected 500 tasks to be available"

        response = client.get("/retrieve-tasks", params={"task_ids": task_ids})

        assert response.status_code == 200, f"Expected 200 OK {response.json()}"
