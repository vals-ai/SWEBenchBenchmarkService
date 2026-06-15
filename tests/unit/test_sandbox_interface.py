import asyncio
from collections.abc import AsyncGenerator

import pytest
from benchmark_service.sandbox import ExecResult, Sandbox
from benchmark_service.schemas import StreamMessageChunk

from swebench_service.benchmark_service import COMMAND_QUIET_SECONDS, PROBLEM_STATEMENT_PATH, SWEBenchService, watchdog_message
from swebench_service.schemas import EvaluationResult


class FakeSandbox(Sandbox):
    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}
        self.commands: list[tuple[str, str | None]] = []

    @property
    def id(self) -> str:
        return "fake-id"

    @property
    def name(self) -> str:
        return "fake-name"

    @property
    def state(self) -> str:
        return "started"

    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> ExecResult:
        self.commands.append((command, cwd))
        return ExecResult(exit_code=0, output="")

    async def command(
        self, command: str, *, cwd: str | None = None, timeout: float | None = None
    ) -> AsyncGenerator[str, None]:
        self.commands.append((command, cwd))
        yield "setup complete"

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        self.uploads[remote_path] = content

    async def download_file(self, remote_path: str) -> bytes:
        return self.uploads[remote_path]


class QuietThenOutputSandbox(FakeSandbox):
    async def command(
        self, command: str, *, cwd: str | None = None, timeout: float | None = None
    ) -> AsyncGenerator[str, None]:
        self.commands.append((command, cwd))
        await asyncio.sleep(0.02)
        yield "command output"


async def test_setup_task_uses_framework_sandbox_interface() -> None:
    """Setup must use the provider-neutral framework sandbox interface.

    Test cases:
    - Problem statement and setup script are uploaded through Sandbox.upload_file.
    - Setup command is streamed through Sandbox.command.
    """
    service = SWEBenchService()
    service.datasets = {
        "default": {
            "task-1": {
                "base_commit": "abc123",
                "problem_statement": "Fix the bug",
                "repo": "django/django",
                "version": "4.2",
            }
        }
    }
    sandbox = FakeSandbox()

    messages = [chunk async for chunk in service.setup_task("task-1", sandbox)]

    assert sandbox.uploads[PROBLEM_STATEMENT_PATH] == b"Fix the bug"
    assert "/setup.sh" in sandbox.uploads
    assert sandbox.commands == [("chmod +x /setup.sh && bash /setup.sh abc123", "/testbed")]
    assert [message.type for message in messages] == ["message", "message", "message", "result"]


async def test_stream_command_emits_watchdog_when_sandbox_is_quiet() -> None:
    """Quiet sandbox commands should still send heartbeat messages upstream.

    Test cases:
    - A watchdog message is emitted while the command stream is quiet.
    - Real command output is still forwarded when it arrives.
    """
    service = SWEBenchService()
    sandbox = QuietThenOutputSandbox()

    messages = [
        message
        async for message in service.stream_command_with_watchdog(
            sandbox,
            "run-tests",
            cwd="/testbed",
            quiet_seconds=0.01,
        )
    ]

    assert any(
        message == "[Debug]: No logs have been produced in the last 0.01 seconds, evaluation may be stuck"
        for message in messages
    )
    assert messages[-1] == "command output"


async def test_evaluate_instance_excludes_watchdog_messages_from_grading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evaluation watchdog messages must not be mixed into SWE-bench test output.

    Test cases:
    - The watchdog message is still streamed to the client.
    - Only real command output is passed to the grader.
    """
    service = SWEBenchService()
    service.datasets = {"default": {"task-1": {"repo": "django/django", "version": "4.2"}}}
    sandbox = FakeSandbox()
    test_spec = object()
    graded_outputs: list[str] = []

    async def stream_with_watchdog(
        sandbox: Sandbox, command: str, *, cwd: str, quiet_seconds: float = COMMAND_QUIET_SECONDS
    ) -> AsyncGenerator[str, None]:
        yield watchdog_message(quiet_seconds)
        yield "test output"

    def grade_test_output(test_output: str, test_spec: object, prediction: str | None) -> EvaluationResult:
        graded_outputs.append(test_output)
        return EvaluationResult(patch_successfully_applied=True, resolved=True, resolution_status="FULL")

    def make_test_spec(task: object) -> object:
        return test_spec

    def create_evaluation_script(spec: object, task_id: str) -> str:
        return ""

    monkeypatch.setattr("swebench_service.benchmark_service.make_test_spec", make_test_spec)
    monkeypatch.setattr("swebench_service.benchmark_service.create_evaluation_script", create_evaluation_script)
    monkeypatch.setattr("swebench_service.benchmark_service.grade_test_output", grade_test_output)
    monkeypatch.setattr(service, "stream_command_with_watchdog", stream_with_watchdog)

    chunks = [chunk async for chunk in service.evaluate_instance("task-1", sandbox)]
    messages = [chunk.data for chunk in chunks if isinstance(chunk, StreamMessageChunk)]

    assert watchdog_message(COMMAND_QUIET_SECONDS) in messages
    assert graded_outputs == ["test output"]
