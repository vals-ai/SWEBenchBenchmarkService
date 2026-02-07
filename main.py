import asyncio
import json
import traceback
from pathlib import Path

from daytona import AsyncDaytona, DaytonaConfig
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket

from src.evaluation import grade_test_output
from src.logger import get_logger
from src.models import (
    EvaluateInstanceRequest,
    EvaluateResponseRequest,
    EvaluationResult,
    FinalScoreRequest,
    FinalScoreResponse,
    HealthCheckResponse,
    Metadata,
    Resources,
    RetrieveTaskResponse,
    SetupTaskRequest,
    SetupTaskResponse,
    TaskFilter,
    VerifyTaskIdsResponse,
)
from src.utils import (
    TaskContext,
    create_final_score,
    fetch_docker_image,
    filter_tasks,
    log_output,
    run_tests,
    stream_command_output,
    validate_task_ids,
)

app = FastAPI()

logger = get_logger(__name__)


@app.exception_handler(Exception)
async def exception_handler(_request: Request, exc: Exception):
    logger.error(exc, exc_info=True)
    raise HTTPException(status_code=500, detail=f"{str(exc)}: {traceback.format_exc()}") from exc


@app.get("/health")
def health_check() -> HealthCheckResponse:
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
    logger.info("Health check endpoint request received")

    return HealthCheckResponse(status="ok")


@app.get("/verify-task-ids")
def verify_task_ids(
    task_ids: list[str] | None = Query(default=None, description="List of task ids to verify"),
    slice: str | None = Query(default=None, description="Slice of the dataset to verify (e.g. 3:10:1, 1:10:2)"),
) -> VerifyTaskIdsResponse:
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
    logger.info(
        f"Verify task ids endpoint request received: ({len(task_ids or [])}) task ids to verify: {', '.join(task_ids[:14]) + ('...' if len(task_ids) > 14 else '') if task_ids else 'all tasks'}."
    )

    task_filter = TaskFilter()

    if task_ids:
        task_filter.task_ids = list(dict.fromkeys(task_ids))

    if slice:
        task_filter.slice_str = slice

    filtered_task_ids = filter_tasks(task_filter)

    return VerifyTaskIdsResponse(task_ids=filtered_task_ids)


@app.get("/retrieve-task/")
async def retrieve_task(
    task_id: str = Query(..., description="Task id to retrieve"),
    skip_validation: bool = Query(False, description="Skip validation of the docker image"),
) -> RetrieveTaskResponse:
    """
    Returns the docker image and metadata for a single task.

    Following this format:
    ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest

    Usage
    curl -X GET http://<endpoint>/retrieve-task/?task_id=task_id_1&skip_validation=true
    {
        "docker_image": "ghcr.io/e.../{instance_id}:latest",
        "problem_statement": "...",
        "request_setup": true,
        "cwd": "/testbed"
    }

    Returns:
    - 200 OK if the task is retrieved successfully
    - 500 Internal Server Error if the task is not retrieved successfully

    """
    larger_tasks: list[str] = ["scikit-learn__scikit-learn-14710", "psf__requests-2317"]

    logger.info(f"Retrieve task endpoint request received: {task_id}, {skip_validation}")

    validated_task_id = validate_task_ids([task_id])[0]

    docker_image, problem_statement, request_setup = await fetch_docker_image(validated_task_id, skip_validation)

    # Default resources required to run a task
    resources = Resources(
        vcpu=2,
        memory=4,
        disk=10,
    )

    # For these tasks, we need a little more
    if validated_task_id in larger_tasks:
        resources.vcpu = 4
        resources.memory = 8

    return RetrieveTaskResponse(
        docker_image=docker_image,
        problem_statement=problem_statement,
        request_setup=request_setup,
        cwd="/testbed",
        resources=resources,
    )


@app.websocket("/ws/setup-task")
async def setup_task(
    websocket: WebSocket,
):
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

    await websocket.accept()

    api_key = websocket.headers.get("x-api-key")
    api_url = websocket.headers.get("x-api-url")
    target = websocket.headers.get("x-target")

    if not api_key or not api_url or not target:
        await websocket.close(code=1008, reason="Missing required headers: x-api-key, x-api-url, x-target")
        return

    data = await websocket.receive_json()

    request = SetupTaskRequest(**data)

    logger.info(f"Setup task endpoint request received: {request.model_dump_json(indent=4)}")

    daytona_config = DaytonaConfig(
        api_key=api_key,
        api_url=api_url,
        target=target,
    )

    async with AsyncDaytona(config=daytona_config) as daytona:
        task_context = TaskContext(request.task_id)

        sandbox = await daytona.get(request.instance_id)

        setup_script = Path("setup.sh").read_text()

        if task_context.pre_install_script:
            setup_script += "\n" + "\n".join(task_context.pre_install_script)

        await sandbox.fs.upload_file(
            setup_script.encode("utf-8"),
            "/setup.sh",
        )

        log_queue: asyncio.Queue[str] = asyncio.Queue()

        def on_output(text: str) -> None:
            if text.strip():
                log_queue.put_nowait(json.dumps({"type": "message", "data": text}))

        log_task = asyncio.create_task(log_output(log_queue, websocket))

        await stream_command_output(
            sandbox, f"chmod +x /setup.sh && bash /setup.sh {task_context.base_commit}", on_output, ignore_error=True
        )

        log_task.cancel()

        try:
            await log_task
        except asyncio.CancelledError:
            pass

        await websocket.send_json({"type": "result", "data": SetupTaskResponse(status="ok").model_dump()})

        await websocket.close()


@app.post("/evaluate-response/")
def evaluate_response(_request: EvaluateResponseRequest):
    raise NotImplementedError(
        "SWE-bench evaluation is done inside of the container itself, use the `evaluate-instance` endpoint to evaluate the response."
    )


@app.websocket("/ws/evaluate-instance")
async def evaluate_instance(websocket: WebSocket):
    """
    Executes tests and grades the results for an instance.

    Usage
    curl -X POST http://<endpoint>/evaluate-instance/ -H "Content-Type: application/json" -H "X-Api-Key: <api_key>" -H "X-Api-Url: <api_url>" -H "X-Target: <target>" -d '{
        "task_id": "task_id_1", "instance_id": "instance_id_1"}'
    {
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

    await websocket.accept()

    api_key = websocket.headers.get("x-api-key")
    api_url = websocket.headers.get("x-api-url")
    target = websocket.headers.get("x-target")

    if not api_key or not api_url or not target:
        await websocket.close(code=1008, reason="Missing required headers: x-api-key, x-api-url, x-target")
        return

    data = await websocket.receive_json()

    request = EvaluateInstanceRequest(**data)

    logger.info(f"Evaluate instance endpoint request received: {request.model_dump_json(indent=4)}")

    # Theres only one task id we need to validate since we are evaluating a single instance
    validated_task_id = validate_task_ids([request.task_id])[0]

    daytona_config = DaytonaConfig(
        api_key=api_key,
        api_url=api_url,
        target=target,
    )

    async with AsyncDaytona(config=daytona_config) as daytona:
        sandbox = await daytona.get(request.instance_id)

        test_output, prediction = await run_tests(sandbox, validated_task_id, websocket)

        final_result: EvaluationResult = grade_test_output(test_output, validated_task_id, prediction)

        await websocket.send_json({"type": "result", "data": final_result.model_dump(exclude_none=True)})

        await websocket.close()


@app.post("/final-score/")
async def final_score(request: FinalScoreRequest) -> FinalScoreResponse:
    """
    Takes the evaluation results and produces a FinalScoreResponse containing the final score and evaluation metadata.

    Usage
    curl -X POST http://<endpoint>/final-score -H "Content-Type: application/json" -d '{"evaluation_results": {"task_id_1": {"resolved": true...}, "task_id_2": {"resolved": false...}}}'
    {
        "tasks_evaluated": ["task_id_1", "task_id_2"],
        "final_score": 50.0,
        "metadata": {"resolved_tasks": ["task_id_1"], "unresolved_tasks": ["task_id_2"]}
    }

    Returns:
    - 200 OK if the final score is calculated successfully
    - 500 Internal Server Error if the final score is not calculated successfully
    """

    tasks_evaluated = list(request.evaluation_results.keys())

    logger.info(
        f"Final score endpoint request received: ({len(tasks_evaluated)}) tasks evaluated: {', '.join(tasks_evaluated[:14]) + ('...' if len(tasks_evaluated) > 14 else '') if tasks_evaluated else 'no tasks evaluated'}."
    )

    # Validate the task ids
    validated_task_ids = validate_task_ids(tasks_evaluated)

    resolved_tasks: list[str] = []
    unresolved_tasks: list[str] = []
    for task_id, evaluation_result in request.evaluation_results.items():
        if evaluation_result and evaluation_result.resolved:
            resolved_tasks.append(task_id)
        else:
            unresolved_tasks.append(task_id)

    metadata = Metadata(resolved_tasks=resolved_tasks, unresolved_tasks=unresolved_tasks)

    return FinalScoreResponse(
        tasks_evaluated=validated_task_ids,
        final_score=create_final_score(len(resolved_tasks), len(validated_task_ids)),
        metadata=metadata,
    )
