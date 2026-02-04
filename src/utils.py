import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncGenerator, cast

from datasets import load_from_disk  # type: ignore
from daytona import AsyncDaytona, AsyncSandbox, CreateSandboxFromImageParams, ExecuteResponse, Resources
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec

from src.logger import get_logger
from src.models import TaskFilter

logger = get_logger(__name__)

_DISK_PATH: Path = Path("/tmp/swe-bench-verified")


@lru_cache(maxsize=None)
def load_dataset_from_disk() -> dict[str, dict[str, Any]]:
    """Load the dataset from disk and return a mapping of instance_id to row data.

    Returns:
        dict[str, dict[str, Any]]: A dictionary mapping instance_id to the corresponding dataset row
    """
    dataset = load_from_disk(_DISK_PATH)  # type: ignore
    return {row["instance_id"]: dict(row) for row in dataset}  # type: ignore


def validate_task_ids(provided_task_ids: list[str]) -> list[str]:
    """Validate the task ids.

    Args:
        provided_task_ids: The list of task ids to validate

    Returns:
        list[str]: The list of task ids that are valid
    """
    task_ids: list[str] = list(load_dataset_from_disk().keys())
    missing_tasks = set(provided_task_ids) - set(task_ids)

    if missing_tasks:
        raise ValueError(f"{len(missing_tasks)} tasks are missing from the dataset: {', '.join(missing_tasks)}")

    return provided_task_ids


def filter_tasks(filter: TaskFilter) -> list[str]:
    """Filter tasks based on the filter type.

    Args:
        filter: The filter type

    Returns:
        list[str]: The list of task ids

    Raises:
        FileNotFoundError: If the SWE-bench tasks are not found
        ValueError: If the filter is invalid
    """

    if not _DISK_PATH.exists():
        raise FileNotFoundError(f"SWE-bench tasks not found at `{_DISK_PATH}`. Run `make setup` to download tasks.")

    dataset_map = load_dataset_from_disk()

    task_ids: list[str] = list(dataset_map.keys())

    print(f"Found `{len(task_ids)}` tasks in `{_DISK_PATH}`")

    if not task_ids:
        raise ValueError(f"No tasks found in `{_DISK_PATH}`. Run `make setup` to download tasks.")

    if not filter.task_ids and not filter.slice_str:
        return task_ids

    if filter.task_ids:
        task_ids = [task_id for task_id in filter.task_ids if task_id in dataset_map]

        if len(task_ids) != len(filter.task_ids):
            missing_task_ids = set(filter.task_ids) - set(task_ids)

            raise ValueError(
                f"Some task ids in the filter are not found in the tasks. Expected `{len(filter.task_ids)}` tasks, but found `{len(task_ids)}`. Missing task ids: `{missing_task_ids}`."
            )

    if filter.slice_str:
        slice = filter.parse_slice()
        task_ids = task_ids[slice]

    return task_ids


async def fetch_docker_image(task_id: str, skip_validation: bool = False) -> tuple[str, str, bool]:
    """Fetch the docker image for a given task id and returns a boolean indicating that after the container is created, there are additional setup steps to be performed.

    Args:
        task_id: The task id

    Returns:
        tuple[str, str, bool]: The docker image, problem statement, and a boolean indicating that after the container is created, there are additional setup steps to be performed

    """
    context = TaskContext(task_id)

    if skip_validation:
        return context.docker_image, context.problem_statement, True

    try:
        docker_image_exists = await context.validate_docker_image(context.docker_image)

        if not docker_image_exists:
            raise ValueError(f"Docker image `{context.docker_image}` was unable to be validated")

    except Exception as e:
        raise ValueError(f"Error validating task `{task_id}`: {e}")

    return context.docker_image, context.problem_statement, True


class TaskContext:
    _task_id: str

    """
    The directory that will be used to upload the files to the task environment.
    """

    def __init__(self, task_id: str):
        self._task_id = task_id

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def _row(self) -> dict[str, Any]:
        dataset_map = load_dataset_from_disk()
        if self._task_id not in dataset_map:
            raise ValueError(f"Task `{self._task_id}` not found in dataset")

        return dataset_map[self._task_id]

    @property
    def problem_statement(self) -> str:
        problem_statement = self._row.get("problem_statement", "")
        if not problem_statement:
            raise ValueError(f"Problem statement not found for task `{self._task_id}`")

        return problem_statement

    @property
    def base_commit(self) -> str:
        base_commit = self._row.get("base_commit")
        if not base_commit:
            raise ValueError(f"Base commit not found for task `{self._task_id}`")

        return base_commit

    @property
    def docker_image(self) -> str:
        return f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{self._task_id}:latest"

    @property
    def patch(self) -> str:
        patch = self._row.get("patch")
        if not patch:
            raise ValueError(f"Patch not found for task `{self._task_id}`")

        return patch

    @property
    def pre_install_script(self) -> list[str]:
        specs: dict[str, Any] = cast(
            dict[str, Any],
            MAP_REPO_VERSION_TO_SPECS.get(self._row.get("repo"), {}).get(self._row.get("version"), {}),  # type: ignore
        )

        return specs.get("pre_install", [])

    @staticmethod
    async def validate_docker_image(image_name: str) -> bool:
        try:
            result = await asyncio.create_subprocess_exec(
                "docker",
                "manifest",
                "inspect",
                image_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            _, stderr = await result.communicate()

            return result.returncode == 0 or (
                len(stderr) > 0 and "unsupported manifest format" in stderr.decode("utf-8")
            )
        except Exception as e:
            raise ValueError(f"Error validating docker image: {e}")


@asynccontextmanager
async def create_sandbox(daytona: AsyncDaytona, sandbox_name: str, image: str) -> AsyncGenerator[AsyncSandbox, Any]:
    # TODO: Take in resources as a parameter
    sandbox = await daytona.create(
        CreateSandboxFromImageParams(
            name=sandbox_name,
            image=image,
            network_block_all=False,
            resources=Resources(
                cpu=4,
                memory=8,
                disk=10,
            ),
        ),
        timeout=360,
    )

    try:
        yield sandbox
    finally:
        await daytona.delete(sandbox)


def fetch_test_spec(task_id: str) -> TestSpec:
    dataset_map = load_dataset_from_disk()
    if task_id not in dataset_map:
        raise ValueError(f"Task `{task_id}` not found in dataset")

    return make_test_spec(dataset_map[task_id])  # type: ignore


def create_evaluation_script(task_id: str) -> str:
    test_spec: TestSpec = fetch_test_spec(task_id)

    evaluation_script: str = test_spec.eval_script

    if "django" in task_id:
        evaluation_script = evaluation_script.replace("locale-gen", "locale-gen en_US.UTF-8")

    return evaluation_script


def create_run_command(task_id: str) -> str:
    run_command = "cd /testbed"
    if "pylint" in task_id:
        run_command += " && PYTHONPATH="

    run_command += " && python3 -c 'import sys; sys.setrecursionlimit(10000)'"
    run_command += " && /bin/bash /root/eval.sh 2>&1"

    return run_command


async def run_tests(sandbox: AsyncSandbox, task_id: str) -> tuple[str, str | None]:
    evaluation_script: str = create_evaluation_script(task_id)

    # Get the prediction from the harness
    prediction_result = await sandbox.process.exec(
        command="git add -N . && git diff HEAD",
        cwd="/testbed",
    )

    await sandbox.fs.upload_file(
        evaluation_script.encode("utf-8"),
        "/root/eval.sh",
    )

    run_command: str = create_run_command(task_id)

    # TODO: Swap to session command
    eval_result: ExecuteResponse = await sandbox.process.exec(
        command=run_command,
        timeout=0,  # Run indefinitely
    )

    if eval_result.exit_code != 0:
        raise ValueError(f"Error running tests for task {task_id}: {eval_result.result}")

    return eval_result.result, (prediction_result.result or None)


def create_final_score(resolved_tasks: int, tasks_evaluated: int) -> float:
    return round((resolved_tasks / tasks_evaluated) * 100, 6)
