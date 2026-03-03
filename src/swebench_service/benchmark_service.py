"""SWE-bench benchmark service implementation."""

import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from benchmark_service import BenchmarkService
from benchmark_service.schemas import (
    EvaluateResponseRequest,
    FinalScoreResult,
    Resources,
    RetrieveTaskResponse,
    StreamChunk,
    StreamMessageChunk,
    StreamResultChunk,
)
from benchmark_service.utils import stream_command
from daytona import AsyncSandbox
from swebench.harness.test_spec.test_spec import make_test_spec

from swebench_service import (
    DISK_PATH,
    create_evaluation_script,
    create_run_command,
    get_pre_install_commands,
    grade_test_output,
    load_dataset_from_disk,
)

logger = logging.getLogger(__name__)


class SWEBenchService(BenchmarkService):
    """SWE-bench benchmark implementation."""

    def load_dataset(self) -> dict[str, Any]:
        """Load SWE-bench_Verified dataset from disk."""
        if not DISK_PATH.exists():
            raise FileNotFoundError(f"Dataset not found at {DISK_PATH}. Run 'make setup' first.")

        return load_dataset_from_disk()

    def retrieve_task(self, task_id: str, skip_validation: bool = False) -> RetrieveTaskResponse:
        """Retrieve task metadata for SWE-bench task."""
        if not skip_validation:
            self.validate_task_ids([task_id])

        task = self.tasks[task_id]
        id_docker_compatible = task_id.replace("__", "_1776_")
        docker_image = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest"
        problem_statement = task.get("problem_statement", "")

        # Default: 2 vCPU, 4GB memory
        resources = Resources(vcpu=2, memory=4, disk=10)

        # Larger tasks need more resources
        if task_id in ["scikit-learn__scikit-learn-14710", "psf__requests-2317"]:
            resources.vcpu = 4
            resources.memory = 8

        return RetrieveTaskResponse(
            docker_image=docker_image,
            problem_statement=problem_statement,
            request_setup=True,
            cwd="/testbed",
            resources=resources,
        )

    async def setup_task(self, task_id: str, sandbox: AsyncSandbox) -> AsyncGenerator[StreamChunk, None]:
        """Setup SWE-bench task environment in sandbox."""
        task = self.tasks[task_id]
        base_commit = task["base_commit"]

        # Build setup script: base + repo-specific pre-install
        setup_script = Path("setup.sh").read_text()
        pre_install = get_pre_install_commands(task["repo"], task["version"])
        if pre_install:
            setup_script += "\n" + "\n".join(pre_install)

        # Upload setup script
        await sandbox.fs.upload_file(setup_script.encode("utf-8"), "/setup.sh")
        yield StreamMessageChunk(type="message", data="Uploaded setup script")

        # Execute setup with streaming (ignore errors as some pre-install commands may fail)
        command = "chmod +x /setup.sh && bash /setup.sh {}".format(base_commit)

        async for line in stream_command(sandbox, command, cwd="/testbed", ignore_error=True):
            yield StreamMessageChunk(type="message", data=line)

        yield StreamResultChunk(type="result", data={"status": "ok"})

    def evaluate_response(self, request: EvaluateResponseRequest) -> Any:
        """SWE-bench requires sandbox evaluation."""
        raise NotImplementedError("SWE-bench evaluation requires sandbox. Use /ws/evaluate-instance endpoint.")

    async def evaluate_instance(self, task_id: str, sandbox: AsyncSandbox) -> AsyncGenerator[StreamChunk, None]:
        """Evaluate SWE-bench solution in sandbox."""
        self.validate_task_ids([task_id])
        task = self.tasks[task_id]

        # Get agent's prediction (git diff)
        yield StreamMessageChunk(type="message", data="Capturing agent's changes...")
        result = await sandbox.process.exec(command="git add -N . && git diff HEAD", cwd="/testbed")
        prediction = result.result or None

        # Create and upload evaluation script
        test_spec = make_test_spec(task)
        eval_script = create_evaluation_script(test_spec, task_id)
        await sandbox.fs.upload_file(eval_script.encode("utf-8"), "/root/eval.sh")
        yield StreamMessageChunk(type="message", data="Uploaded evaluation script")

        # Execute tests with streaming
        run_command = create_run_command(task_id)
        test_output: list[str] = []

        yield StreamMessageChunk(type="message", data="Running tests...")
        async for line in stream_command(sandbox, run_command, cwd="/testbed", ignore_error=True):
            yield StreamMessageChunk(type="message", data=line)
            test_output.append(line)

        # Grade results
        evaluation_result = grade_test_output("".join(test_output), test_spec, prediction)

        yield StreamResultChunk(type="result", data=evaluation_result.model_dump())

    def calculate_final_score(self, evaluation_results: dict[str, Any]) -> FinalScoreResult:
        """Calculate final score as percentage of resolved tasks."""
        total = len(evaluation_results)

        resolved_tasks: list[str] = []
        unresolved_tasks: list[str] = []

        for task_id, result in evaluation_results.items():
            if result and result.get("resolved", False):
                resolved_tasks.append(task_id)
            else:
                unresolved_tasks.append(task_id)

        resolved = len(resolved_tasks)
        score = round((resolved / total) * 100, 6) if total > 0 else 0.0

        metadata = {
            "resolved_tasks": resolved_tasks,
            "unresolved_tasks": unresolved_tasks,
        }

        return FinalScoreResult(score=score, metadata=metadata)
