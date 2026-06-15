import asyncio
from collections.abc import AsyncGenerator

from benchmark_service.sandbox import ExecResult, Sandbox

from swebench_service.benchmark_service import PROBLEM_STATEMENT_PATH, SWEBenchService


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

    assert messages == [
        "[Debug]: No logs have been produced in the last 0.01 seconds, evaluation may be stuck",
        "command output",
    ]
