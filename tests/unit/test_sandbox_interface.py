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
