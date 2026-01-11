from typing import Any

from daytona import AsyncDaytona, DaytonaConfig
from daytona.common.process import ExecuteResponse
from fastapi import FastAPI, Header, HTTPException, Query

from src.evaluation import grade_test_output
from src.types import (
    EvaluateInstanceRequest,
    EvaluateResponseRequest,
    EvaluationResult,
    FinalScoreRequest,
    SetupTaskRequest,
    TaskFilter,
)
from src.utils import TaskContext, create_final_score, fetch_docker_image, filter_tasks, run_tests

app = FastAPI()


@app.get("/health")
def health_check():
    """
    Health check to ensure that we are able to connect to the server.

    Usage
    curl -X GET http://<endpoint>/health
    {
        "status": "ok"
    }

    Returns:
    - 200 OK if the server is running
    - 500 Internal Server Error if the server is not running

    """
    return {"status": "ok"}


@app.get("/verify-task-ids")
def verify_task_ids(task_ids: list[str] | None = Query(default=None, description="List of task ids to verify")):
    """
    Verify the task ids and return list of task ids that can be found inside of the SWE-bench benchmark service.
    Used later to request tasks from the SWE-bench benchmark service.

    Usage
    curl -X GET http://<endpoint>/verify-task-ids?task_ids=task_id_1&task_ids=task_id_2&task_ids=task_id_3
    {
        "task_ids": ["task_id_1", "task_id_2", "task_id_3"]  # Only the provided task ids are verified
    }

    curl -X GET http://<endpoint>/verify-task-ids
    {
        "task_ids": ["task_id_1", "task_id_2", "task_id_3"]  # All tasks are verified
    }

    Returns:
    - 200 OK if the task ids are verified successfully
    - 500 Internal Server Error if the task ids are not verified successfully
    """
    try:
        task_filter = TaskFilter()

        if task_ids:
            task_filter.task_ids = list(dict.fromkeys(task_ids))

        filtered_task_ids = filter_tasks(task_filter)

        return {"task_ids": filtered_task_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retrieve-task/")
async def retrieve_task(
    task_id: str = Query(..., description="Task id to retrieve"),
    skip_validation: bool = Query(False, description="Skip validation of the docker image"),
) -> dict[str, Any]:
    """
    Returns the docker image and metadata for a single task.

    Following this format:
    ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest

    Usage
    curl -X GET http://<endpoint>/retrieve-task/?task_id=task_id_1&skip_validation=true
    {
        "docker_image": "ghcr.io/e.../{instance_id}:latest",
        "problem_statement": "...",
        "request_setup": true
    }

    Returns:
    - 200 OK if the task is retrieved successfully
    - 500 Internal Server Error if the task is not retrieved successfully

    """
    try:
        docker_image, problem_statement, request_setup = await fetch_docker_image(task_id, skip_validation)

        return {
            "docker_image": docker_image,
            "problem_statement": problem_statement,
            "request_setup": request_setup,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/setup-task")
async def setup_task(
    request: SetupTaskRequest,
    x_api_key: str = Header(...),
    x_api_url: str = Header(...),
    x_target: str = Header(...),
) -> dict[str, str]:
    """
    Setup the task by running the setup script for the task.

    Usage
    curl -X POST http://<endpoint>/setup-task -H "x-api-key: <api_key>" -H "x-api-url: <api_url>" -H "x-target: <target>" -d '{"task_id": "task_id_1", "instance_id": "instance_id_1"}'
    {
        "status": "ok"
    }

    Returns:
    - 200 OK if the task is setup successfully
    - 500 Internal Server Error if the task is not setup successfully

    """
    try:
        daytona = AsyncDaytona(
            config=DaytonaConfig(
                api_key=x_api_key,
                api_url=x_api_url,
                target=x_target,
            )
        )

        task_context = TaskContext(request.task_id)

        sandbox = await daytona.get(request.instance_id)

        await sandbox.fs.upload_file(
            "setup.sh",
            "/setup.sh",
        )

        result: ExecuteResponse = await sandbox.process.exec(
            command=f"chmod +x /setup.sh && bash /setup.sh {task_context.base_commit}",
        )

        if result.exit_code != 0:
            raise HTTPException(status_code=500, detail=result.result)

        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate-response/")
def evaluate_response(_request: EvaluateResponseRequest):
    raise NotImplementedError(
        "SWE-bench evaluation is done inside of the container itself, use the `evaluate-instance` endpoint to evaluate the response."
    )


@app.post("/evaluate-instance/")
async def evaluate_instance(
    request: EvaluateInstanceRequest,
    x_api_key: str = Header(...),
    x_api_url: str = Header(...),
    x_target: str = Header(...),
) -> dict[str, Any]:
    """
    Executes tests and grades the results for an instance.

    Usage
    curl -X POST http://<endpoint>/evaluate-instance/ -H "Content-Type: application/json" -H "X-Api-Key: <api_key>" -H "X-Api-Url: <api_url>" -H "X-Target: <target>" -d '{
        "task_id": "task_id_1", "instance_id": "instance_id_1"}'
    {
        "task_id": "task_id_1",
        "instance_id": "instance_id_1",
        "patch_successfully_applied": true,
        "resolved": true,
        "resolution_status": "RESOLVED_FULL",
        "fail_to_pass": {"success": ["error_1", "error_2"], "failure": ["error_3"]},
        "pass_to_pass": {"success": ["pass_1", "pass_2"], "failure": ["pass_3"]},
        "f2p_score": 1.0,
        "p2p_score": 1.0,
        "status_map": {"test_1": "PASSED", "test_2": "SKIPPED", "test_3": "FAILED"}
    }

    Returns:
    - 200 OK if the instance is evaluated successfully
    - 500 Internal Server Error if the instance is not evaluated successfully
    """
    try:
        daytona = AsyncDaytona(
            config=DaytonaConfig(
                api_key=x_api_key,
                api_url=x_api_url,
                target=x_target,
            )
        )

        sandbox = await daytona.get(request.instance_id)

        test_output: str = await run_tests(sandbox, request.task_id, request.instance_id)

        final_result: EvaluationResult = grade_test_output(test_output, request.task_id, request.instance_id)

        return final_result.model_dump(exclude_none=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/final-score/")
async def final_score(request: FinalScoreRequest) -> dict[str, Any]:
    """
    Takes the evaluation results and produces a json containing the final score and evaluation metadata.

    Usage
    curl -X POST http://<endpoint>/final-score -H "Content-Type: application/json" -d '{"evaluation_results": {"task_id_1": {"resolved": true...}, "task_id_2": {"resolved": false...}}}'
    {
        "tasks_evaluated": ["task_id_1", "task_id_2"],
        "final_score": 50.0,
        "resolved_tasks": ["task_id_1"],
        "unresolved_tasks": ["task_id_2"],
        "evaluation_results": {"task_id_1": {"resolved": true...}, "task_id_2": {"resolved": false...}}
    }

    Returns:
    - 200 OK if the final score is calculated successfully
    - 500 Internal Server Error if the final score is not calculated successfully
    """
    try:
        tasks_evaluated = list(request.evaluation_results.keys())

        resolved_tasks: list[str] = []
        unresolved_tasks: list[str] = []
        for task_id, evaluation_result in request.evaluation_results.items():
            if evaluation_result and evaluation_result.resolved:
                resolved_tasks.append(task_id)
            else:
                unresolved_tasks.append(task_id)

        return {
            "tasks_evaluated": tasks_evaluated,
            "final_score": create_final_score(len(resolved_tasks), len(tasks_evaluated)),
            "resolved_tasks": resolved_tasks,
            "unresolved_tasks": unresolved_tasks,
            **request.model_dump(exclude_none=True),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
