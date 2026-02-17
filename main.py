"""
SWE-bench Benchmark Service.

This service implements the SWE-bench benchmark using the benchmark service template framework.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from daytona import AsyncSandbox, SessionExecuteRequest
from swebench.harness.test_spec.test_spec import make_test_spec

from benchmark_service import BenchmarkService, create_app
from benchmark_service.schemas import (
    EvaluateResponseRequest,
    FinalScoreResult,
    Resources,
    RetrieveTaskResponse,
    StreamChunk,
    StreamMessageChunk,
    StreamResultChunk,
)
from swebench_utils import (
    DISK_PATH,
    create_evaluation_script,
    create_run_command,
    get_pre_install_commands,
    grade_test_output,
    load_dataset_from_disk,
)

logger = logging.getLogger(__name__)


async def stream_command(
    sandbox: AsyncSandbox, command: str, cwd: str, ignore_error: bool = False
) -> AsyncGenerator[str, None]:
    """
    Execute a command in a sandbox and stream output line by line in real-time.

    Args:
        sandbox: The sandbox instance
        command: Command to execute
        cwd: Working directory
        ignore_error: Whether to ignore non-zero exit codes

    Yields:
        Output lines from the command as they are produced
    """
    session_id = f"{sandbox.id}-{str(uuid.uuid4())}"

    try:
        await sandbox.process.create_session(session_id)

        session_exec_resp = await sandbox.process.execute_session_command(
            session_id, SessionExecuteRequest(command=f"cd {cwd} && {command}", run_async=True)
        )

        cmd_id = session_exec_resp.cmd_id
        if not cmd_id:
            raise ValueError(f"Failed to execute command in session {session_id}")

        output_queue: asyncio.Queue[str] = asyncio.Queue()

        # Queue lines as they arrive
        def on_output(text: str) -> None:
            if text.strip():
                output_queue.put_nowait(text)

        # Start command with streaming logs
        log_task = asyncio.create_task(
            sandbox.process.get_session_command_logs_async(
                session_id=session_id,
                command_id=cmd_id,
                on_stdout=on_output,
                on_stderr=on_output,
            )
        )

        # Yield lines as they arrive in queue
        while not log_task.done():
            try:
                line = await asyncio.wait_for(output_queue.get(), timeout=0.1)
                yield line
            except asyncio.TimeoutError:
                # Keep polling on timeout error
                continue

        # Drain queue after command completes
        while not output_queue.empty():
            yield output_queue.get_nowait()

        cmd = await sandbox.process.get_session_command(session_id, cmd_id)
        if cmd.exit_code != 0 and not ignore_error:
            raise ValueError(f"Command failed with exit code {cmd.exit_code}")

    finally:
        try:
            await sandbox.process.delete_session(session_id)
        except Exception:
            # NOTE: If we kill the sandbox this sometimes errors
            logger.error(f"Caught failure to delete session `{session_id}`")
            pass


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
        docker_image = f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{task_id}:latest"
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
        full_output = "\n".join(test_output)
        evaluation_result = grade_test_output(full_output, test_spec, prediction)

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


# Create the FastAPI app with SWE-bench implementation
app = create_app(SWEBenchService())
