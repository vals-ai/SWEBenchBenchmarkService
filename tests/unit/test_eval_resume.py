from collections.abc import AsyncGenerator
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from benchmark_service.sandbox import (
    DaytonaProviderConfig,
    ExecResult,
    Sandbox,
    SandboxCreateRequest,
    SandboxProvider,
    SandboxQuery,
)
from benchmark_service.schemas import EvaluateResponseRequest, StreamChunk, StreamResultChunk
from pydantic import ValidationError

from swebench_service.benchmark_service import (
    PREDICTION_CAPTURE_COMMAND,
    PREDICTION_PATH,
    SWEBenchService,
    _resume_sandbox_name,  # pyright: ignore[reportPrivateUsage]
)
from swebench_service.eval_resume import EvalResumeState, load_prediction, persist_prediction
from swebench_service.schemas import EvaluationResult


class FakeSandbox(Sandbox):
    def __init__(
        self,
        sandbox_id: str = "original-sandbox",
        captured_prediction: bytes = b"diff --git a/a.py b/a.py\n+fixed\n",
    ) -> None:
        self._id = sandbox_id
        self.captured_prediction = captured_prediction
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
        return self._id

    @property
    def name(self) -> str:
        return self._id

    @property
    def state(self) -> str:
        return "started"

    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> ExecResult:
        self.commands.append((command, cwd))
        if command == PREDICTION_CAPTURE_COMMAND:
            self.uploads[PREDICTION_PATH] = self.captured_prediction
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


class FakeProvider(SandboxProvider):
    def __init__(self) -> None:
        self.sandbox = FakeSandbox("resume-sandbox")
        self.create_request: SandboxCreateRequest | None = None
        self.deleted: list[str] = []

    async def create_sandbox(self, request: SandboxCreateRequest) -> Sandbox:
        self.create_request = request
        return self.sandbox

    async def get_sandbox(self, instance_id: str) -> Sandbox:
        raise AssertionError("resume must create a fresh sandbox")

    async def delete_sandbox(self, instance_id: str) -> None:
        self.deleted.append(instance_id)

    async def list_sandboxes(self, query: SandboxQuery) -> AsyncGenerator[Sandbox, None]:
        if False:
            yield self.sandbox


def service() -> SWEBenchService:
    instance = SWEBenchService()
    instance.datasets = {
        "default": {
            "task-1": {
                "base_commit": "abc123",
                "problem_statement": "Fix it",
                "repo": "django/django",
                "version": "4.2",
            }
        }
    }
    return instance


def sandbox_provider_config() -> DaytonaProviderConfig:
    return DaytonaProviderConfig(
        DAYTONA_API_KEY="key",
        DAYTONA_API_URL="url",
        DAYTONA_TARGET="target",
    )


def use_provider(monkeypatch: pytest.MonkeyPatch, provider: FakeProvider) -> None:
    def create_provider(_config: DaytonaProviderConfig) -> SandboxProvider:
        return provider

    monkeypatch.setattr(DaytonaProviderConfig, "create_provider", create_provider)


@pytest.mark.parametrize(
    ("patch_bytes", "expected_prediction"),
    [
        (b"diff --git a/a.py b/a.py\n+fixed\n", "diff --git a/a.py b/a.py\n+fixed\n"),
        (b"diff --git a/a.py b/a.py\n+\xff\n", "diff --git a/a.py b/a.py\n+\ufffd\n"),
    ],
)
async def test_failed_evaluation_resumes_from_exact_persisted_patch(
    monkeypatch: pytest.MonkeyPatch,
    patch_bytes: bytes,
    expected_prediction: str,
) -> None:
    benchmark = service()
    original_sandbox = FakeSandbox(captured_prediction=patch_bytes)
    evaluation_started = False

    async def fail_evaluation(
        task_id: str,
        sandbox: Sandbox,
        prediction: str | None,
        dataset: str | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        nonlocal evaluation_started
        evaluation_started = True
        if False:
            yield StreamResultChunk(type="result", data={})
        raise RuntimeError("injected evaluator failure")

    monkeypatch.setattr(benchmark, "_evaluate_prediction", fail_evaluation)

    emitted: list[StreamChunk] = []
    with pytest.raises(RuntimeError, match="injected evaluator failure"):
        async for chunk in benchmark.evaluate_instance("task-1", original_sandbox):
            emitted.append(chunk)

    assert evaluation_started
    assert [chunk.type for chunk in emitted] == ["message", "eval_resume_state"]
    state = EvalResumeState.model_validate(emitted[-1].data)
    prediction = await load_prediction(state)
    assert prediction == patch_bytes

    provider = FakeProvider()
    provider.sandbox.captured_prediction = patch_bytes
    use_provider(monkeypatch, provider)
    evaluated_predictions: list[str | None] = []

    async def succeed_evaluation(
        task_id: str,
        sandbox: Sandbox,
        prediction: str | None,
        dataset: str | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        evaluated_predictions.append(prediction)
        yield StreamResultChunk(
            type="result",
            data=EvaluationResult(
                prediction=prediction,
                patch_successfully_applied=True,
                resolved=True,
                resolution_status="FULL",
            ).model_dump(),
        )

    monkeypatch.setattr(benchmark, "_evaluate_prediction", succeed_evaluation)
    request = EvaluateResponseRequest(
        task_id="task-1",
        eval_resume_state=state.model_dump(mode="json"),
        sandbox_provider=sandbox_provider_config(),
    )
    resumed = [chunk async for chunk in benchmark.stream_evaluate_response(request)]

    assert resumed[-1].type == "result"
    assert evaluated_predictions == [expected_prediction]
    assert provider.sandbox.uploads[PREDICTION_PATH] == prediction
    assert (
        f"git apply --binary {PREDICTION_PATH}",
        "/testbed",
    ) in provider.sandbox.commands
    assert provider.deleted == [provider.sandbox.id]
    assert provider.create_request is not None
    assert provider.create_request.labels == {
        "Benchmark": "swebench",
        "Id": "00000000-0000-0000-0000-000000000001",
        "Task": "task-1",
        "EvalResume": "true",
    }


async def test_resume_deletes_sandbox_when_evaluation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = service()
    original_sandbox = FakeSandbox(captured_prediction=b"patch")
    state = await persist_prediction(original_sandbox, "task-1", None, b"patch")
    provider = FakeProvider()
    provider.sandbox.captured_prediction = b"patch"
    use_provider(monkeypatch, provider)

    async def fail_evaluation(
        task_id: str,
        sandbox: Sandbox,
        prediction: str | None,
        dataset: str | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        if False:
            yield StreamResultChunk(type="result", data={})
        raise RuntimeError("resume failed")

    monkeypatch.setattr(benchmark, "_evaluate_prediction", fail_evaluation)
    request = EvaluateResponseRequest(
        task_id="task-1",
        eval_resume_state=state.model_dump(mode="json"),
        sandbox_provider=sandbox_provider_config(),
    )

    with pytest.raises(RuntimeError, match="resume failed"):
        _ = [chunk async for chunk in benchmark.stream_evaluate_response(request)]

    assert provider.deleted == [provider.sandbox.id]


async def test_resume_rejects_mismatched_task_before_loading_artifact() -> None:
    benchmark = service()
    state = await persist_prediction(FakeSandbox(), "task-1", None, b"patch")
    request = EvaluateResponseRequest(task_id="other-task", eval_resume_state=state.model_dump(mode="json"))

    with pytest.raises(ValueError, match="task_id mismatch"):
        _ = [chunk async for chunk in benchmark.stream_evaluate_response(request)]


async def test_resume_rejects_mismatched_dataset_before_loading_artifact() -> None:
    benchmark = service()
    state = await persist_prediction(FakeSandbox(), "task-1", "default", b"patch")
    request = EvaluateResponseRequest(
        task_id="task-1",
        dataset="vals_index",
        eval_resume_state=state.model_dump(mode="json"),
    )

    with pytest.raises(ValueError, match="dataset mismatch"):
        _ = [chunk async for chunk in benchmark.stream_evaluate_response(request, dataset="vals_index")]


async def test_resume_requires_request_scoped_sandbox_provider() -> None:
    benchmark = service()
    state = await persist_prediction(FakeSandbox(), "task-1", None, b"patch")
    request = EvaluateResponseRequest(task_id="task-1", eval_resume_state=state.model_dump(mode="json"))

    with pytest.raises(ValueError, match="requires sandbox_provider"):
        _ = [chunk async for chunk in benchmark.stream_evaluate_response(request)]


@pytest.mark.parametrize(
    ("modified_content", "error"),
    [(b"x", "byte-length"), (b"other", "SHA-256")],
)
async def test_resume_rejects_modified_persisted_patch(modified_content: bytes, error: str) -> None:
    state = await persist_prediction(FakeSandbox(), "task-1", None, b"patch")
    local_root = Path(os.environ["SWEBENCH_EVAL_STATE_LOCAL_DIR"])
    (local_root / state.prediction_s3_key).write_bytes(modified_content)

    with pytest.raises(ValueError, match=error):
        await load_prediction(state)


def test_resume_state_rejects_path_components() -> None:
    with pytest.raises(ValidationError):
        EvalResumeState(
            benchmark_id=UUID("00000000-0000-0000-0000-000000000001"),
            task_id="../other-task",
            dataset="default",
            prediction_s3_key="swebench/eval-resume/other",
            prediction_sha256="0" * 64,
            prediction_size_bytes=1,
        )


async def test_resume_sandbox_names_are_unique() -> None:
    state = await persist_prediction(FakeSandbox(), "task-1", None, b"patch")

    assert _resume_sandbox_name(state) != _resume_sandbox_name(state)


async def test_resume_state_rejects_noncanonical_object_key() -> None:
    state = await persist_prediction(FakeSandbox(), "task-1", None, b"patch")
    data = state.model_dump(mode="json")
    data["prediction_s3_key"] = "swebench/eval-resume/other.patch"

    with pytest.raises(ValidationError, match="canonical"):
        EvalResumeState.model_validate(data)


async def test_upload_failure_does_not_start_evaluation_or_emit_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = service()
    evaluation_started = False

    async def fail_upload(
        sandbox: Sandbox,
        task_id: str,
        dataset: str | None,
        prediction: bytes,
    ) -> EvalResumeState:
        raise RuntimeError("injected upload failure")

    async def evaluation(
        task_id: str,
        sandbox: Sandbox,
        prediction: str | None,
        dataset: str | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        nonlocal evaluation_started
        evaluation_started = True
        if False:
            yield StreamResultChunk(type="result", data={})

    monkeypatch.setattr("swebench_service.benchmark_service.persist_prediction", fail_upload)
    monkeypatch.setattr(benchmark, "_evaluate_prediction", evaluation)

    emitted: list[StreamChunk] = []
    with pytest.raises(RuntimeError, match="injected upload failure"):
        async for chunk in benchmark.evaluate_instance("task-1", FakeSandbox()):
            emitted.append(chunk)

    assert not evaluation_started
    assert [chunk.type for chunk in emitted] == ["message"]


async def test_empty_prediction_resumes_without_git_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = service()
    state = await persist_prediction(
        FakeSandbox(captured_prediction=b""),
        "task-1",
        None,
        b"",
    )
    provider = FakeProvider()
    provider.sandbox.captured_prediction = b""
    use_provider(monkeypatch, provider)

    async def evaluation(
        task_id: str,
        sandbox: Sandbox,
        prediction: str | None,
        dataset: str | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        assert prediction is None
        yield StreamResultChunk(type="result", data={"resolved": False})

    monkeypatch.setattr(benchmark, "_evaluate_prediction", evaluation)
    request = EvaluateResponseRequest(
        task_id="task-1",
        eval_resume_state=state.model_dump(mode="json"),
        sandbox_provider=sandbox_provider_config(),
    )

    chunks = [chunk async for chunk in benchmark.stream_evaluate_response(request)]

    assert chunks[-1].type == "result"
    assert not any(command.startswith("git apply") for command, _ in provider.sandbox.commands)
    assert provider.deleted == [provider.sandbox.id]
