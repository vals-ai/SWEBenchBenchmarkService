from pathlib import Path

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from main import app
from src.types import EvaluationResult
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

        # Validate task_id restriction (min length 1)
        response = client.get("/retrieve-tasks", params={"task_ids": []})
        assert response.status_code == 422, f"Expected 422 Unprocessable Entity {response.json()}"

    async def test_final_score(self) -> None:
        first_evaluation_result = EvaluationResult(
            task_id="astropy__astropy-12907",
            instance_id="astropy__astropy-12907",
            patch_successfully_applied=True,
            resolved=True,
            resolution_status="FULL",
        )

        second_evaluation_result = EvaluationResult(
            task_id="django__django-12050",
            instance_id="django__django-12050",
            patch_successfully_applied=True,
            resolved=False,
            resolution_status="NO",
        )

        evaluation_results = {
            "astropy__astropy-12907": first_evaluation_result.model_dump(),
            "django__django-12050": second_evaluation_result.model_dump(),
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
            "resolved_tasks": ["astropy__astropy-12907"],
            "unresolved_tasks": ["django__django-12050"],
            "evaluation_results": evaluation_results,
        }
