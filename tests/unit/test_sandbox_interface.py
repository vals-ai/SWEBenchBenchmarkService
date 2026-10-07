import asyncio
import base64
from collections.abc import AsyncGenerator, Mapping
import re
from types import SimpleNamespace
from typing import Any, cast

import pytest
from benchmark_service.sandbox import ExecResult, Sandbox
from benchmark_service.schemas import StreamMessageChunk

from swebench.harness.constants import TESTS_TIMEOUT

from swebench_service.benchmark_service import (
    MULTIMODAL_REPO_RESOURCES,
    MULTIMODAL_RESOURCES,
    EvaluationStalled,
    COMMAND_QUIET_SECONDS,
    PREDICTION_CAPTURE_COMMAND,
    PROBLEM_STATEMENT_PATH,
    SWEBenchService,
    watchdog_message,
)
from swebench_service.schemas import EvaluationResult


class FakeSandbox(Sandbox):
    def __init__(self) -> None:
        self._sandbox = SimpleNamespace(
            labels={
                "Id": "00000000-0000-0000-0000-000000000001",
                "Benchmark": "swebench",
            }
        )
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
        if PREDICTION_CAPTURE_COMMAND in command:
            match = re.search(r">\s*(\S*swebench-prediction-capture\S*)", command)
            capture_path = match.group(1) if match is not None else "/tmp/swebench-prediction.patch"
            self.uploads[capture_path] = b""
            return ExecResult(exit_code=0, output="0")
        return ExecResult(exit_code=0, output="")

    async def command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env_vars: Mapping[str, str] | None = None,
    ) -> AsyncGenerator[str, None]:
        del env_vars
        self.commands.append((command, cwd))
        if "base64 " in command:
            path = command.split("base64 ", 1)[1].split(" && ", 1)[0].strip()
            yield "SWEBENCH_PREDICTION_BASE64_BEGIN\r\n"
            yield base64.b64encode(self.uploads[path]).decode()
            yield "\r\nSWEBENCH_PREDICTION_BASE64_END\r\n"
            return
        yield "setup complete"

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        self.uploads[remote_path] = content

    async def download_file(self, remote_path: str) -> bytes:
        return self.uploads[remote_path]


class QuietThenOutputSandbox(FakeSandbox):
    async def command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env_vars: Mapping[str, str] | None = None,
    ) -> AsyncGenerator[str, None]:
        del env_vars
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
    service.datasets = {"default": {"task-1": {"base_commit": "abc123", "repo": "django/django", "version": "4.2"}}}
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

    def create_evaluation_script(spec: object, task_id: str, restore_commands: list[str] | None = None) -> str:
        return ""

    monkeypatch.setattr("swebench_service.benchmark_service.make_test_spec", make_test_spec)
    monkeypatch.setattr("swebench_service.benchmark_service.create_evaluation_script", create_evaluation_script)
    monkeypatch.setattr("swebench_service.benchmark_service.grade_test_output", grade_test_output)
    monkeypatch.setattr(service, "stream_command_with_watchdog", stream_with_watchdog)

    chunks = [chunk async for chunk in service.evaluate_instance("task-1", sandbox)]
    messages = [chunk.data for chunk in chunks if isinstance(chunk, StreamMessageChunk)]

    assert watchdog_message(COMMAND_QUIET_SECONDS) in messages
    assert graded_outputs == ["test output"]


async def test_evaluate_instance_grades_captured_log_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grading must use the faithfully-captured log file, not the PTY stream.

    The eval runs inside an interactive PTY, on which TTY-sensitive reporters
    (e.g. sympy's `bin/test`) render a carriage-return progress bar with no
    parseable per-test lines. `create_run_command` tees a non-TTY copy to
    EVAL_OUTPUT_PATH; `evaluate_instance` grades that file.

    Test cases:
    - The captured file contents are read back via Sandbox.exec.
    - The file contents (not the reassembled stream) are passed to the grader.
    """
    from swebench_service.test_spec import EVAL_OUTPUT_PATH

    service = SWEBenchService()
    service.datasets = {"default": {"task-1": {"base_commit": "abc123", "repo": "sympy/sympy", "version": "1.9"}}}
    graded_outputs: list[str] = []

    class LogFileSandbox(FakeSandbox):
        async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> ExecResult:
            if command == f"cat {EVAL_OUTPUT_PATH}":
                self.commands.append((command, cwd))
                return ExecResult(exit_code=0, output="test_Mul ok\ntest_Abs ok\n")
            return await super().exec(command, cwd=cwd, timeout=timeout)

    sandbox = LogFileSandbox()
    test_spec = object()

    async def stream_with_watchdog(
        sandbox: Sandbox, command: str, *, cwd: str, quiet_seconds: float = COMMAND_QUIET_SECONDS
    ) -> AsyncGenerator[str, None]:
        # The PTY-mangled stream carries no parseable per-test lines.
        yield "sympy/printing/tests/test_str.py[100] \r"

    def grade_test_output(test_output: str, test_spec: object, prediction: str | None) -> EvaluationResult:
        graded_outputs.append(test_output)
        return EvaluationResult(patch_successfully_applied=True, resolved=True, resolution_status="FULL")

    def make_test_spec(task: object) -> object:
        return test_spec

    def create_evaluation_script(spec: object, task_id: str, restore_commands: list[str] | None = None) -> str:
        return ""

    monkeypatch.setattr("swebench_service.benchmark_service.make_test_spec", make_test_spec)
    monkeypatch.setattr("swebench_service.benchmark_service.create_evaluation_script", create_evaluation_script)
    monkeypatch.setattr("swebench_service.benchmark_service.grade_test_output", grade_test_output)
    monkeypatch.setattr(service, "stream_command_with_watchdog", stream_with_watchdog)

    _ = [chunk async for chunk in service.evaluate_instance("task-1", sandbox)]

    assert (f"cat {EVAL_OUTPUT_PATH}", "/testbed") in sandbox.commands
    assert graded_outputs == ["test_Mul ok\ntest_Abs ok\n"]


class NeverOutputsSandbox(FakeSandbox):
    async def command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env_vars: Mapping[str, str] | None = None,
    ) -> AsyncGenerator[str, None]:
        del command, cwd, timeout, env_vars
        await asyncio.sleep(3600)
        yield "never"


async def test_stream_command_gives_up_on_a_stalled_command() -> None:
    """A hung test suite (a Jest worker dead of a heap overflow, a rendering runner waiting on a page that threw)
    produced only watchdog lines for hours; after the stall budget the stream ends with EvaluationStalled."""
    service = SWEBenchService()
    sandbox = NeverOutputsSandbox()
    messages: list[str] = []

    with pytest.raises(EvaluationStalled, match="no output for"):
        async for message in service.stream_command_with_watchdog(
            sandbox, "sleep", cwd="/testbed", quiet_seconds=0.01, stall_seconds=0.03
        ):
            messages.append(message)

    assert messages and all(message == watchdog_message(0.01) for message in messages)
    assert len(messages) == 2  # two quiet windows reported, the third trips the bound


async def test_stalled_evaluation_is_graded_as_a_test_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    service = SWEBenchService()
    service.datasets = {"default": {"task-1": {"base_commit": "abc123", "repo": "openlayers/openlayers", "version": "7.1"}}}
    graded_outputs: list[str] = []

    async def stall(
        sandbox: Sandbox, command: str, *, cwd: str, quiet_seconds: float = COMMAND_QUIET_SECONDS
    ) -> AsyncGenerator[str, None]:
        yield "Running 12 rendering cases\n"
        raise EvaluationStalled("no output for 1800 seconds")

    def grade_test_output(test_output: str, test_spec: object, prediction: object) -> EvaluationResult:
        graded_outputs.append(test_output)
        return EvaluationResult(patch_successfully_applied=False, resolved=False, resolution_status="NO")

    def make_test_spec(task: object) -> object:
        return object()

    def create_evaluation_script(spec: object, task_id: str, restore_commands: list[str] | None = None) -> str:
        return ""

    monkeypatch.setattr("swebench_service.benchmark_service.make_test_spec", make_test_spec)
    monkeypatch.setattr("swebench_service.benchmark_service.create_evaluation_script", create_evaluation_script)
    monkeypatch.setattr("swebench_service.benchmark_service.grade_test_output", grade_test_output)
    monkeypatch.setattr(service, "stream_command_with_watchdog", stall)

    chunks = [chunk async for chunk in service.evaluate_instance("task-1", FakeSandbox())]

    assert graded_outputs and graded_outputs[0].endswith(f"\n{TESTS_TIMEOUT}\n")
    assert "Running 12 rendering cases" in graded_outputs[0]
    assert any("stalled" in str(chunk.data) and "test timeout" in str(chunk.data) for chunk in chunks if chunk.type == "message")
    result = chunks[-1]
    assert result.type == "result"
    assert isinstance(result.data, dict) and result.data["resolved"] is False


async def test_multimodal_resources_give_carbon_more_memory() -> None:
    service = SWEBenchService()
    service.datasets = {
        "multimodal": {
            "carbon-design-system__carbon-11352": {"base_commit": "a", "repo": "carbon-design-system/carbon", "version": "16.15", "image": "swebench/x:latest"},
            "openlayers__openlayers-14332": {"base_commit": "b", "repo": "openlayers/openlayers", "version": "7.1", "image": "swebench/y:latest"},
        }
    }

    carbon = await service.retrieve_task("carbon-design-system__carbon-11352", skip_validation=True, dataset="multimodal")
    openlayers = await service.retrieve_task("openlayers__openlayers-14332", skip_validation=True, dataset="multimodal")

    assert carbon.resources == MULTIMODAL_REPO_RESOURCES["carbon-design-system/carbon"]
    assert carbon.resources.memory == 16
    assert openlayers.resources == MULTIMODAL_RESOURCES


def _multimodal_service() -> SWEBenchService:
    service = SWEBenchService()
    service.datasets = {
        "multimodal": {"task-1": {"base_commit": "abc123", "repo": "openlayers/openlayers", "version": "7.1"}}
    }
    return service


def _patch_eval_helpers(monkeypatch: pytest.MonkeyPatch, graded: list[tuple[str, dict[str, object]]]) -> None:
    def grade_test_output(test_output: str, test_spec: object, prediction: object, **kwargs: object) -> EvaluationResult:
        graded.append((test_output, kwargs))
        return EvaluationResult(patch_successfully_applied=True, resolved=True, resolution_status="FULL")

    def make_test_spec(task: object) -> object:
        return object()

    def create_evaluation_script(spec: object, task_id: str, restore_commands: list[str] | None = None) -> str:
        return ""

    monkeypatch.setattr("swebench_service.benchmark_service.make_test_spec", make_test_spec)
    monkeypatch.setattr("swebench_service.benchmark_service.create_evaluation_script", create_evaluation_script)
    monkeypatch.setattr("swebench_service.benchmark_service.grade_test_output", grade_test_output)


class PatchedSandbox(FakeSandbox):
    """A sandbox whose agent left a one-line patch, and whose eval log file reads back as `log_file`."""

    def __init__(self, log_file: str | None) -> None:
        super().__init__()
        self.log_file = log_file

    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> ExecResult:
        if PREDICTION_CAPTURE_COMMAND in command:
            self.commands.append((command, cwd))
            match = re.search(r">\s*(\S*swebench-prediction-capture\S*)", command)
            self.uploads[match.group(1) if match else "/tmp/p"] = b"diff --git a/x b/x\n"
            return ExecResult(exit_code=0, output=str(len(b"diff --git a/x b/x\n")))
        if command.startswith("cat ") and self.log_file is not None:
            self.commands.append((command, cwd))
            return ExecResult(exit_code=0, output=self.log_file)
        return await super().exec(command, cwd=cwd, timeout=timeout)


async def test_an_empty_multimodal_patch_is_counted_unresolved_without_running_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The harness drops an empty patch before it runs anything; running it would hand out fail_only passes."""
    service = _multimodal_service()
    graded: list[tuple[str, dict[str, object]]] = []
    _patch_eval_helpers(monkeypatch, graded)
    ran: list[str] = []

    async def stream(sandbox: Sandbox, command: str, *, cwd: str, **kwargs: object) -> AsyncGenerator[str, None]:
        ran.append(command)
        yield "should not run"

    monkeypatch.setattr(service, "stream_command_with_watchdog", stream)
    sandbox = FakeSandbox()

    chunks = [chunk async for chunk in service.evaluate_instance("task-1", sandbox, dataset="multimodal")]

    result = cast(dict[str, Any], [chunk.data for chunk in chunks if chunk.type == "result"][-1])
    assert result["resolved"] is False and result["resolution_status"] == "NO"
    assert result["patch_successfully_applied"] is False
    assert ran == [] and graded == []
    assert "/root/eval.sh" not in sandbox.uploads


async def test_an_empty_verified_patch_is_still_evaluated(monkeypatch: pytest.MonkeyPatch) -> None:
    service = SWEBenchService()
    service.datasets = {"default": {"task-1": {"base_commit": "abc123", "repo": "django/django", "version": "4.2"}}}
    graded: list[tuple[str, dict[str, object]]] = []
    _patch_eval_helpers(monkeypatch, graded)

    async def stream(sandbox: Sandbox, command: str, *, cwd: str, **kwargs: object) -> AsyncGenerator[str, None]:
        assert kwargs == {}
        yield "output"

    monkeypatch.setattr(service, "stream_command_with_watchdog", stream)

    _ = [chunk async for chunk in service.evaluate_instance("task-1", FakeSandbox())]

    assert graded == [("output", {})]


async def test_multimodal_grades_the_log_file_exactly_and_caps_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from swebench_service.benchmark_service import EVAL_TOTAL_SECONDS

    service = _multimodal_service()
    graded: list[tuple[str, dict[str, object]]] = []
    _patch_eval_helpers(monkeypatch, graded)
    limits: list[object] = []

    async def stream(sandbox: Sandbox, command: str, *, cwd: str, **kwargs: object) -> AsyncGenerator[str, None]:
        limits.append(kwargs.get("total_seconds"))
        yield "stream\r\n"

    monkeypatch.setattr(service, "stream_command_with_watchdog", stream)

    _ = [chunk async for chunk in service.evaluate_instance("task-1", PatchedSandbox("file\n"), dataset="multimodal")]

    assert len(limits) == 1
    assert isinstance(limits[0], float) and 0 < limits[0] <= EVAL_TOTAL_SECONDS
    assert graded == [("file\n", {"upstream_exact": True})]


async def test_multimodal_cleans_the_pty_stream_when_the_log_file_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _multimodal_service()
    graded: list[tuple[str, dict[str, object]]] = []
    _patch_eval_helpers(monkeypatch, graded)

    async def stream(sandbox: Sandbox, command: str, *, cwd: str, **kwargs: object) -> AsyncGenerator[str, None]:
        yield "\x1b[31mFAILED\x1b[0m a\r\n"

    monkeypatch.setattr(service, "stream_command_with_watchdog", stream)

    _ = [chunk async for chunk in service.evaluate_instance("task-1", PatchedSandbox(None), dataset="multimodal")]

    assert graded == [("FAILED a\n", {"upstream_exact": True})]


async def test_a_run_that_keeps_printing_still_hits_the_total_cap() -> None:
    from swebench_service.benchmark_service import EvaluationTimedOut

    class ChattySandbox(FakeSandbox):
        async def command(
            self,
            command: str,
            *,
            cwd: str | None = None,
            timeout: float | None = None,
            env_vars: Mapping[str, str] | None = None,
        ) -> AsyncGenerator[str, None]:
            del command, cwd, timeout, env_vars
            while True:
                await asyncio.sleep(0.005)
                yield "still going\n"

    service = SWEBenchService()
    with pytest.raises(EvaluationTimedOut, match="still running after"):
        async for _ in service.stream_command_with_watchdog(
            ChattySandbox(), "x", cwd="/testbed", quiet_seconds=0.05, stall_seconds=5.0, total_seconds=0.05
        ):
            pass
    assert issubclass(EvaluationTimedOut, EvaluationStalled)
