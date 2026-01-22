from pathlib import Path

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from main import app
from src.types import EvaluationResult
from src.utils import load_dataset_from_disk

client = TestClient(app, raise_server_exceptions=False)


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
        assert response.json() == {"task_ids": list(load_dataset_from_disk().keys())}  # type: ignore

        # Slice with start, stop, step
        response = client.get("/verify-task-ids", params={"slice": "3:10:1"})

        assert response.status_code == 200
        dataset_map = load_dataset_from_disk()
        all_task_ids = list(dataset_map.keys())
        expected_task_ids = all_task_ids[3:10:1]  # type: ignore
        assert response.json() == {"task_ids": expected_task_ids}

        # Slice with start, stop
        response = client.get("/verify-task-ids", params={"slice": "3:10"})

        assert response.status_code == 200
        expected_task_ids = all_task_ids[3:10]  # type: ignore
        assert response.json() == {"task_ids": expected_task_ids}

        # Slice with start
        response = client.get("/verify-task-ids", params={"slice": "3:"})

        assert response.status_code == 200
        expected_task_ids = all_task_ids[3:]  # type: ignore
        assert response.json() == {"task_ids": expected_task_ids}

        # Slice with stop
        response = client.get("/verify-task-ids", params={"slice": ":10"})

        assert response.status_code == 200
        expected_task_ids = all_task_ids[:10]  # type: ignore
        assert response.json() == {"task_ids": expected_task_ids}

    async def test_retrieve_task(self, setup_dataset: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.utils._DISK_PATH", setup_dataset)

        registry_image_format = "ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"

        valid_task_id: str = "astropy__astropy-12907"

        response = client.get("/retrieve-task/", params={"task_id": valid_task_id})

        dataset_map = load_dataset_from_disk()
        problem_statement: str = dataset_map[valid_task_id].get("problem_statement", "")

        assert response.status_code == 200

        expected_response: dict[str, str | bool] = {
            "docker_image": registry_image_format.format(instance_id=valid_task_id),
            "problem_statement": problem_statement,
            "request_setup": True,
            "cwd": "/testbed",
        }

        assert response.json() == expected_response

        response = client.get("/retrieve-task/", params={"task_id": "invalid-task-id"})

        assert response.status_code == 500

        response = client.get("/retrieve-task/", params={"task_id": "django__django-12050"})

        problem_statement_django: str = dataset_map["django__django-12050"].get("problem_statement", "")

        expected_response_django: dict[str, str | bool] = {
            "docker_image": registry_image_format.format(instance_id="django__django-12050"),
            "problem_statement": problem_statement_django,
            "request_setup": True,
            "cwd": "/testbed",
        }

        assert response.status_code == 200, f"Expected 200 OK {response.json()}"
        assert response.json() == expected_response_django

        response = client.get("/retrieve-task/")
        assert response.status_code == 422, f"Expected 422 Unprocessable Entity {response.json()}"

    async def test_final_score(self) -> None:
        first_evaluation_result = EvaluationResult(
            task_id="astropy__astropy-12907",
            instance_id="astropy__astropy-12907",
            patch_successfully_applied=True,
            resolved=True,
            resolution_status="FULL",
        )

        evaluation_results = {
            "astropy__astropy-12907": first_evaluation_result.model_dump(exclude_none=True),
            "django__django-12050": None,
        }

        response = client.post(
            "/final-score",
            json={
                "evaluation_results": evaluation_results,
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "tasks_evaluated": ["astropy__astropy-12907", "django__django-12050"],
            "final_score": round(50.0, 6),
            "metadata": {
                "resolved_tasks": ["astropy__astropy-12907"],
                "unresolved_tasks": ["django__django-12050"],
            },
        }
