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

    async def test_retrieve_tasks(self, setup_dataset: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.utils._DISK_PATH", setup_dataset)

        registry_image_format = "ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"

        # Contains valid task ids
        response = client.get(
            "/retrieve-tasks", params={"task_ids": ["astropy__astropy-12907", "django__django-12050"]}
        )

        assert response.status_code == 200
        assert response.json() == {
            "astropy__astropy-12907": {
                "docker_image": registry_image_format.format(instance_id="astropy__astropy-12907"),
                "request_setup": True,
            },
            "django__django-12050": {
                "docker_image": registry_image_format.format(instance_id="django__django-12050"),
                "request_setup": True,
            },
        }

        # Contains invalid task ids
        response = client.get("/retrieve-tasks", params={"task_ids": ["astropy__astropy-12907", "invalid-task-id"]})

        assert response.status_code == 500

        # All tasks are valid
        task_ids: list[str] = [row["instance_id"] for row in load_dataset_from_disk()]  # type: ignore

        assert len(task_ids) == 500, "Expected 500 tasks to be available"

        response = client.get("/retrieve-tasks", params={"task_ids": task_ids})

        assert response.status_code == 200, f"Expected 200 OK {response.json()}"
