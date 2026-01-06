import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from datasets import Dataset, load_from_disk  # type: ignore
from daytona import AsyncDaytona, AsyncSandbox, CreateSandboxFromImageParams, ExecuteResponse, Image, Resources
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec

from src.logger import get_logger
from src.types import TaskFilter

logger = get_logger(__name__)

_DISK_PATH: Path = Path("/tmp/swe-bench-verified")


def load_dataset_from_disk() -> Dataset:
    return load_from_disk(_DISK_PATH)  # type: ignore


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
        raise FileNotFoundError(
            f"SWE-bench tasks not found at `{_DISK_PATH}`. Run `make task-setup` to download tasks."
        )

    dataset = load_dataset_from_disk()

    task_ids: list[str] = [row["instance_id"] for row in dataset]  # type: ignore

    print(f"Found `{len(task_ids)}` tasks in `{_DISK_PATH}`")

    if not task_ids:
        raise ValueError(f"No tasks found in `{_DISK_PATH}`. Run `make task-setup` to download tasks.")

    if not filter.task_ids:
        return task_ids

    intersection = [task_id for task_id in filter.task_ids if task_id in task_ids]

    if len(intersection) != len(filter.task_ids):
        missing_task_ids = set(filter.task_ids) - set(intersection)

        raise ValueError(
            f"Some task ids in the filter are not found in the tasks. Expected `{len(filter.task_ids)}` tasks, but found `{len(intersection)}`. Missing task ids: `{missing_task_ids}`."
        )

    return list(intersection)


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
        dataset = load_dataset_from_disk()
        row = dataset.filter(lambda x: x["instance_id"] == self._task_id)  # type: ignore
        if not row:
            raise ValueError(f"Task `{self._task_id}` not found in dataset")

        return row[0]  # type: ignore

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
    sandbox = await daytona.create(
        CreateSandboxFromImageParams(
            name=sandbox_name,
            image=Image.base(image),
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
    dataset = load_dataset_from_disk()
    row = dataset.filter(lambda x: x["instance_id"] == task_id)  # type: ignore
    if not row:
        raise ValueError(f"Task `{task_id}` not found in dataset")

    return make_test_spec(row[0])  # type: ignore


async def fetch_patch(sandbox: AsyncSandbox) -> str:
    patch = await sandbox.process.exec(
        command="git add -A && git diff --cached",
        cwd="/testbed",
    )

    if patch.exit_code != 0:
        raise ValueError(f"Error fetching patch: {patch.result}")

    return patch.result


async def apply_patch(sandbox: AsyncSandbox, patch_path: str) -> str:
    GIT_APPLY_CMDS = [
        "git apply --verbose",
        "git apply --verbose --reject",
        "patch --batch --fuzz=5 -p1 -i",
    ]

    for git_apply_cmd in GIT_APPLY_CMDS:
        result: ExecuteResponse = await sandbox.process.exec(
            command=f"{git_apply_cmd} {patch_path}",
            cwd="/testbed",
        )

        if result.exit_code == 0:
            return result.result
        else:
            logger.warning(f"Failed to apply patch command `{git_apply_cmd}`:{result.result}")

    raise ValueError(f"Failed to apply patch `{patch_path}`")


def create_evaluation_script(task_id: str, instance_id: str) -> str:
    test_spec: TestSpec = fetch_test_spec(task_id)

    evaluation_script: str = test_spec.eval_script

    if "django" in instance_id:
        evaluation_script = evaluation_script.replace("locale-gen", "locale-gen en_US.UTF-8")

    return evaluation_script


def create_run_command(instance_id: str) -> str:
    run_command = "cd /testbed"
    if "pylint" in instance_id:
        run_command += " && PYTHONPATH="

    run_command += " && python3 -c 'import sys; sys.setrecursionlimit(10000)'"
    run_command += " && /bin/bash /root/eval.sh"

    return run_command


async def run_tests(sandbox: AsyncSandbox, task_id: str, instance_id: str) -> str:
    # Fetch the patch from the container
    container_patch = await fetch_patch(sandbox)

    # Save the patch to /tmp/patch.diff
    await sandbox.fs.upload_file(
        container_patch.encode("utf-8"),
        "/tmp/patch.diff",
    )

    # Reset the current state of the repository to the base commit
    await sandbox.process.exec(
        command="git reset --hard HEAD && git clean -fd",
        cwd="/testbed",
    )

    # Apply the patch to the repository
    await apply_patch(sandbox, "/tmp/patch.diff")

    evaluation_script: str = create_evaluation_script(task_id, instance_id)

    await sandbox.fs.upload_file(
        evaluation_script.encode("utf-8"),
        "/root/eval.sh",
    )

    run_command: str = create_run_command(instance_id)

    result: ExecuteResponse = await sandbox.process.exec(
        command=run_command,
    )

    return result.result


def create_final_score(resolved_tasks: int, tasks_evaluated: int) -> float:
    return round((resolved_tasks / tasks_evaluated) * 100, 6)
