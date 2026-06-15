"""SWE-bench benchmark service implementation."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from benchmark_service import BenchmarkService
from benchmark_service.sandbox import ImageSource, Sandbox, SandboxCommandError, SandboxError
from benchmark_service.schemas import (
    EvaluateResponseRequest,
    FinalScoreResult,
    Resources,
    RetrieveTaskResponse,
    StreamChunk,
    StreamMessageChunk,
    StreamResultChunk,
)
from swebench.harness.test_spec.test_spec import make_test_spec

from swebench_service import (
    DISK_PATH,
    create_evaluation_script,
    create_run_command,
    get_pre_install_commands,
    grade_test_output,
    load_dataset_from_disk,
    load_vals_index_subset,
)
from swebench_service.utils import with_retry

logger = logging.getLogger(__name__)

PROBLEM_STATEMENT_PATH = "/tmp/problem_statement.txt"
COMMAND_QUIET_SECONDS = 300.0


class SWEBenchService(BenchmarkService):
    """SWE-bench benchmark implementation."""

    async def _stream_command_with_retry(
        self, sandbox: Sandbox, command: str, cwd: str, retries: int = 3
    ) -> AsyncGenerator[str, None]:
        """Stream command output with retry on transient sandbox errors."""
        for attempt in range(retries):
            try:
                async for line in self.stream_command_with_watchdog(sandbox, command, cwd=cwd):
                    yield line
                return
            except SandboxCommandError:
                return
            except (SandboxError, RuntimeError):
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2**attempt)
                yield f"Stream interrupted, retrying (attempt {attempt + 2}/{retries})..."

    async def stream_command_with_watchdog(
        self,
        sandbox: Sandbox,
        command: str,
        *,
        cwd: str,
        quiet_seconds: float = COMMAND_QUIET_SECONDS,
    ) -> AsyncGenerator[str, None]:
        output: asyncio.Queue[str | None] = asyncio.Queue()

        async def stream_command() -> None:
            try:
                async for line in sandbox.command(command, cwd=cwd):
                    await output.put(line)
            finally:
                await output.put(None)

        stream_task = asyncio.create_task(stream_command())
        try:
            while True:
                try:
                    line = await asyncio.wait_for(output.get(), timeout=quiet_seconds)
                except TimeoutError:
                    yield (
                        f"[Debug]: No logs have been produced in the last {quiet_seconds:g} seconds, "
                        "evaluation may be stuck"
                    )
                    continue
                if line is None:
                    break
                yield line
            await stream_task
        finally:
            if not stream_task.done():
                stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream_task

    async def load_datasets(self) -> dict[str, dict[str, Any]]:
        """Load SWE-bench_Verified dataset from disk."""
        if not DISK_PATH.exists():
            raise FileNotFoundError(f"Dataset not found at {DISK_PATH}. Run 'make setup' first.")

        return {
            "default": load_dataset_from_disk(),
            "vals_index": load_vals_index_subset(),
        }

    async def retrieve_task(
        self, task_id: str, skip_validation: bool = False, dataset: str | None = None
    ) -> RetrieveTaskResponse:
        """Retrieve task metadata for SWE-bench task."""
        if not skip_validation:
            await self.validate_task_ids([task_id], dataset=dataset)

        id_docker_compatible = task_id.replace("__", "_1776_")
        docker_image = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest"

        # Default: 2 vCPU, 4GB memory
        resources = Resources(vcpu=2, memory=4, disk=10)

        # Larger tasks need more resources
        if task_id in ["scikit-learn__scikit-learn-14710", "psf__requests-2317"]:
            resources.vcpu = 4
            resources.memory = 8

        return RetrieveTaskResponse(
            source=ImageSource(image=docker_image),
            problem_path=PROBLEM_STATEMENT_PATH,
            cwd="/testbed",
            agent_timeout=None,
            resources=resources,
        )

    async def setup_task(
        self, task_id: str, sandbox: Sandbox, dataset: str | None = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Setup SWE-bench task environment in sandbox."""
        ds = self.get_dataset(dataset)
        task = ds[task_id]
        base_commit = task["base_commit"]

        # Write problem statement to sandbox
        problem_statement = task.get("problem_statement", "")
        await with_retry(sandbox, lambda: sandbox.upload_file(PROBLEM_STATEMENT_PATH, problem_statement.encode()))
        yield StreamMessageChunk(type="message", data="Uploaded problem statement")

        # Build setup script: base + repo-specific pre-install
        setup_script = Path("setup.sh").read_text()
        pre_install = get_pre_install_commands(task["repo"], task["version"])
        if pre_install:
            setup_script += "\n" + "\n".join(pre_install)

        # Upload setup script
        await with_retry(sandbox, lambda: sandbox.upload_file("/setup.sh", setup_script.encode()))
        yield StreamMessageChunk(type="message", data="Uploaded setup script")

        # Execute setup with streaming (ignore errors as some pre-install commands may fail)
        command = "chmod +x /setup.sh && bash /setup.sh {}".format(base_commit)

        async for line in self._stream_command_with_retry(sandbox, command, cwd="/testbed"):
            yield StreamMessageChunk(type="message", data=line)

        yield StreamResultChunk(type="result", data={"status": "ok"})

    async def evaluate_response(self, request: EvaluateResponseRequest, dataset: str | None = None) -> Any:
        """SWE-bench requires sandbox evaluation."""
        raise NotImplementedError("SWE-bench evaluation requires sandbox. Use /ws/evaluate-instance endpoint.")

    async def evaluate_instance(
        self, task_id: str, sandbox: Sandbox, dataset: str | None = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Evaluate SWE-bench solution in sandbox."""
        await self.validate_task_ids([task_id], dataset=dataset)
        task = self.get_dataset(dataset)[task_id]

        # Get agent's prediction (git diff)
        yield StreamMessageChunk(type="message", data="Capturing agent's changes...")
        result = await with_retry(sandbox, lambda: sandbox.exec("git add -N . && git diff HEAD", cwd="/testbed"))
        prediction = result.output or None

        # Create and upload evaluation script
        test_spec = make_test_spec(task)
        eval_script = create_evaluation_script(test_spec, task_id)
        await with_retry(sandbox, lambda: sandbox.upload_file("/root/eval.sh", eval_script.encode()))
        yield StreamMessageChunk(type="message", data="Uploaded evaluation script")

        # Execute tests with streaming. Reset output on retry so grading only sees a complete run.
        run_command = create_run_command(task_id)
        MAX_RETRIES = 3

        test_output: list[str] = []
        for attempt in range(MAX_RETRIES):
            test_output = []
            msg = "Running tests..." if attempt == 0 else f"Stream interrupted, retrying (attempt {attempt + 1}/{MAX_RETRIES})..."
            yield StreamMessageChunk(type="message", data=msg)
            try:
                async for line in self.stream_command_with_watchdog(sandbox, run_command, cwd="/testbed"):
                    test_output.append(line)
                    yield StreamMessageChunk(type="message", data=line)
                break
            except SandboxCommandError:
                break
            except (SandboxError, RuntimeError):
                if attempt == MAX_RETRIES - 1:
                    raise
            await asyncio.sleep(2**attempt)

        # Grade results
        evaluation_result = grade_test_output("".join(test_output), test_spec, prediction)

        yield StreamResultChunk(type="result", data=evaluation_result.model_dump())

    async def calculate_final_score(
        self, evaluation_results: dict[str, Any], dataset: str | None = None
    ) -> FinalScoreResult:
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
