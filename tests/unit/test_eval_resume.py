import asyncio
import base64
from collections.abc import AsyncGenerator, Mapping
import os
from pathlib import Path
import re
import subprocess
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from benchmark_service.sandbox import (
    DaytonaProviderConfig,
    ExecResult,
    Sandbox,
    SandboxCreateRequest,
    SandboxError,
    SandboxProvider,
    SandboxQuery,
    SnapshotSource,
)
from benchmark_service.schemas import EvaluateResponseRequest, Resources, StreamChunk, StreamResultChunk
from pydantic import ValidationError

import swebench_service.benchmark_service as service_module
from swebench_service.benchmark_service import (
    PREDICTION_CAPTURE_COMMAND,
    PREDICTION_PATH,
    SWEBenchService,
    _resume_sandbox_name,  # pyright: ignore[reportPrivateUsage]
)
from swebench_service.eval_resume import EvalResumeState, load_prediction, persist_prediction
from swebench_service.schemas import EvaluationResult

TEST_TASK_CONTRACT_SHA256 = "0" * 64


def test_setup_script_requires_baseline_trap_anchor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "setup.sh").write_text("#!/bin/sh\nset -eu\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="baseline trap"):
        service_module._build_setup_script(  # pyright: ignore[reportPrivateUsage]
            {"repo": "astropy/astropy", "version": "main"}
        )


class FakeSandbox(Sandbox):
    def __init__(
        self,
        sandbox_id: str = "original-sandbox",
        captured_prediction: bytes = b"diff --git a/a.py b/a.py\n+fixed\n",
    ) -> None:
        self._id = sandbox_id
        self.captured_prediction = captured_prediction
        self.labels = {
            "Id": "00000000-0000-0000-0000-000000000001",
            "Benchmark": "swebench",
        }
        self.uploads: dict[str, bytes] = {}
        self.commands: list[tuple[str, str | None]] = []
        self.downloads: list[str] = []

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
        if PREDICTION_CAPTURE_COMMAND in command:
            match = re.search(r">\s*(\S+)", command)
            capture_path = match.group(1) if match is not None else PREDICTION_PATH
            self.uploads[capture_path] = self.captured_prediction
            return ExecResult(exit_code=0, output=str(len(self.captured_prediction)))
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
            encoded = base64.b64encode(self.uploads[path]).decode()
            yield "SWEBENCH_PREDICTION_BASE64_BEGIN\r\n"
            for offset in range(0, len(encoded), 73):
                yield encoded[offset : offset + 73]
            yield "\r\nSWEBENCH_PREDICTION_BASE64_END\r\n"
            return
        yield "setup complete"

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        self.uploads[remote_path] = content

    async def download_file(self, remote_path: str) -> bytes:
        self.downloads.append(remote_path)
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


class GitSandbox(Sandbox):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.repo = root / "testbed"
        self.remote_tmp = root / "tmp"
        self.remote_tmp.mkdir(parents=True)
        self.labels = {
            "Id": "00000000-0000-0000-0000-000000000001",
            "Benchmark": "swebench",
        }
        self.commands: list[tuple[str, str | None]] = []

    @property
    def id(self) -> str:
        return "git-sandbox"

    @property
    def name(self) -> str:
        return "git-sandbox"

    @property
    def state(self) -> str:
        return "started"

    def _local_path(self, remote_path: str) -> Path:
        if remote_path == "/setup.sh":
            return self.root / "setup.sh"
        if remote_path.startswith("/tmp/"):
            return self.remote_tmp / remote_path.removeprefix("/tmp/")
        if remote_path.startswith("/testbed/"):
            return self.repo / remote_path.removeprefix("/testbed/")
        raise AssertionError(f"unexpected remote path: {remote_path}")

    def _localize(self, value: str) -> str:
        localized = (
            value.replace("/tmp/", f"{self.remote_tmp}/")
            .replace("/setup.sh", str(self.root / "setup.sh"))
            .replace("/testbed", str(self.repo))
        )
        return re.sub(r"stat -c %s -- (\S+)", r"wc -c < \1", localized)

    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> ExecResult:
        del timeout
        self.commands.append((command, cwd))
        completed = subprocess.run(
            self._localize(command),
            cwd=self._localize(cwd) if cwd is not None else None,
            shell=True,
            executable="/bin/bash",
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return ExecResult(exit_code=completed.returncode, output=completed.stdout)

    async def command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env_vars: Mapping[str, str] | None = None,
    ) -> AsyncGenerator[str, None]:
        del env_vars
        if "base64 " in command:
            self.commands.append((command, cwd))
            path = command.split("base64 ", 1)[1].split(" && ", 1)[0].strip()
            encoded = base64.b64encode(self._local_path(path).read_bytes()).decode()
            yield "SWEBENCH_PREDICTION_BASE64_BEGIN\r\n"
            for offset in range(0, len(encoded), 73):
                yield encoded[offset : offset + 73]
            yield "\r\nSWEBENCH_PREDICTION_BASE64_END\r\n"
            return
        result = await self.exec(command, cwd=cwd, timeout=timeout)
        if result.output:
            yield result.output

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        local_path = self._local_path(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if remote_path == "/setup.sh":
            content = self._localize(content.decode()).encode()
        local_path.write_bytes(content)

    async def download_file(self, remote_path: str) -> bytes:
        return self._local_path(remote_path).read_bytes()


def test_git_sandbox_localizes_paths_once_when_fixture_root_is_under_tmp() -> None:
    sandbox = object.__new__(GitSandbox)
    sandbox.root = Path("/tmp/swebench-fixture")
    sandbox.repo = sandbox.root / "testbed"
    sandbox.remote_tmp = sandbox.root / "tmp"

    assert sandbox._localize("/setup.sh") == str(  # pyright: ignore[reportPrivateUsage]
        sandbox.root / "setup.sh"
    )
    assert sandbox._localize("/testbed") == str(sandbox.repo)  # pyright: ignore[reportPrivateUsage]


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


async def persist_for_service(
    benchmark: SWEBenchService,
    sandbox: Sandbox,
    dataset: str | None,
    prediction: bytes,
) -> EvalResumeState:
    task_data = await benchmark.retrieve_task("task-1", skip_validation=True, dataset=dataset)
    return await persist_prediction(
        sandbox,
        "task-1",
        dataset,
        prediction,
        task_contract_sha256=benchmark._task_contract_sha256(  # pyright: ignore[reportPrivateUsage]
            "task-1",
            dataset,
            task_data,
        ),
    )


def sandbox_provider_config() -> DaytonaProviderConfig:
    return DaytonaProviderConfig(
        DAYTONA_API_KEY="key",
        DAYTONA_API_URL="url",
        DAYTONA_TARGET="target",
    )


def resume_sandbox_request() -> SandboxCreateRequest:
    return SandboxCreateRequest(
        source=SnapshotSource(snapshot="snapshot"),
        resources=Resources(vcpu=1, memory=1, disk=1),
        name="resume",
        labels={},
        env_vars={},
        auto_stop_interval=15,
        create_timeout=600,
    )


def use_provider(monkeypatch: pytest.MonkeyPatch, provider: FakeProvider) -> None:
    def create_provider(_config: DaytonaProviderConfig) -> SandboxProvider:
        return provider

    monkeypatch.setattr(DaytonaProviderConfig, "create_provider", create_provider)


async def setup_git_sandbox_with_setup_owned_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[SWEBenchService, GitSandbox]:
    sandbox = GitSandbox(tmp_path)
    sandbox.repo.mkdir()
    subprocess.run(["git", "init", "-q", str(sandbox.repo)], check=True)
    subprocess.run(["git", "-C", str(sandbox.repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(sandbox.repo), "config", "user.name", "Test"], check=True)
    (sandbox.repo / "setup_owned.txt").write_text("base setup value\n")
    (sandbox.repo / "agent_owned.txt").write_text("base agent value\n")
    subprocess.run(["git", "-C", str(sandbox.repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(sandbox.repo), "commit", "-qm", "base"], check=True)
    base_commit = subprocess.run(
        ["git", "-C", str(sandbox.repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    benchmark = service()
    benchmark.datasets["default"]["task-1"]["base_commit"] = base_commit

    def setup_pre_install(_repo: str, _version: str) -> list[str]:
        return ["printf 'setup change\\n' > setup_owned.txt"]

    monkeypatch.setattr(service_module, "get_pre_install_commands", setup_pre_install)

    _ = [chunk async for chunk in benchmark.setup_task("task-1", sandbox)]
    return benchmark, sandbox


async def test_capture_is_empty_before_agent_work_after_setup_owned_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark, sandbox = await setup_git_sandbox_with_setup_owned_change(monkeypatch, tmp_path)

    prediction = await benchmark._capture_prediction(sandbox)  # pyright: ignore[reportPrivateUsage]

    assert prediction == b""


async def test_capture_uses_post_setup_baseline_for_agent_patch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark, sandbox = await setup_git_sandbox_with_setup_owned_change(monkeypatch, tmp_path)
    (sandbox.repo / "agent_owned.txt").write_text("agent change\n")

    prediction = await benchmark._capture_prediction(sandbox)  # pyright: ignore[reportPrivateUsage]

    assert b"agent_owned.txt" in prediction
    assert b"+agent change" in prediction
    assert b"setup_owned.txt" not in prediction


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
    state = await persist_for_service(benchmark, original_sandbox, None, b"patch")
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
    state = await persist_prediction(
        FakeSandbox(),
        "task-1",
        None,
        b"patch",
        task_contract_sha256=TEST_TASK_CONTRACT_SHA256,
    )
    request = EvaluateResponseRequest(task_id="other-task", eval_resume_state=state.model_dump(mode="json"))

    with pytest.raises(ValueError, match="task_id mismatch"):
        _ = [chunk async for chunk in benchmark.stream_evaluate_response(request)]


async def test_resume_rejects_mismatched_dataset_before_loading_artifact() -> None:
    benchmark = service()
    state = await persist_prediction(
        FakeSandbox(),
        "task-1",
        "default",
        b"patch",
        task_contract_sha256=TEST_TASK_CONTRACT_SHA256,
    )
    request = EvaluateResponseRequest(
        task_id="task-1",
        dataset="vals_index",
        eval_resume_state=state.model_dump(mode="json"),
    )

    with pytest.raises(ValueError, match="dataset mismatch"):
        _ = [chunk async for chunk in benchmark.stream_evaluate_response(request, dataset="vals_index")]


@pytest.mark.parametrize("changed_contract", ["task_image", "setup_inputs", "evaluator"])
async def test_resume_rejects_changed_task_contract_before_loading_artifact(
    monkeypatch: pytest.MonkeyPatch,
    changed_contract: str,
) -> None:
    benchmark = service()

    async def stop_after_checkpoint(
        task_id: str,
        sandbox: Sandbox,
        prediction: str | None,
        dataset: str | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        del task_id, sandbox, prediction, dataset
        raise RuntimeError("stop after checkpoint")
        yield StreamResultChunk(type="result", data={})

    monkeypatch.setattr(benchmark, "_evaluate_prediction", stop_after_checkpoint)
    emitted: list[StreamChunk] = []
    with pytest.raises(RuntimeError, match="stop after checkpoint"):
        async for chunk in benchmark.evaluate_instance("task-1", FakeSandbox()):
            emitted.append(chunk)
    state = EvalResumeState.model_validate(emitted[-1].data)

    if changed_contract == "task_image":
        monkeypatch.setitem(service_module.IMAGE_DIGEST_OVERRIDES, "task-1", f"sha256:{'1' * 64}")
    elif changed_contract == "setup_inputs":

        def changed_pre_install(_repo: str, _version: str) -> list[str]:
            return ["printf 'changed setup\\n' > setup-owned.txt"]

        monkeypatch.setattr(service_module, "get_pre_install_commands", changed_pre_install)
    else:

        def changed_run_command(_task_id: str) -> str:
            return "changed evaluator command"

        monkeypatch.setattr(service_module, "create_run_command", changed_run_command)

    async def unexpected_load(_state: EvalResumeState) -> bytes:
        raise AssertionError("artifact must not load before task-contract validation")

    monkeypatch.setattr(service_module, "load_prediction", unexpected_load)
    provider = FakeProvider()
    use_provider(monkeypatch, provider)
    request = EvaluateResponseRequest(
        task_id="task-1",
        eval_resume_state=state.model_dump(mode="json"),
        sandbox_provider=sandbox_provider_config(),
    )

    with pytest.raises(ValueError, match="task contract"):
        _ = [chunk async for chunk in benchmark.stream_evaluate_response(request)]

    assert provider.create_request is None


async def test_resume_requires_request_scoped_sandbox_provider() -> None:
    benchmark = service()
    state = await persist_prediction(
        FakeSandbox(),
        "task-1",
        None,
        b"patch",
        task_contract_sha256=TEST_TASK_CONTRACT_SHA256,
    )
    request = EvaluateResponseRequest(task_id="task-1", eval_resume_state=state.model_dump(mode="json"))

    with pytest.raises(ValueError, match="requires sandbox_provider"):
        _ = [chunk async for chunk in benchmark.stream_evaluate_response(request)]


@pytest.mark.parametrize(
    ("modified_content", "error"),
    [(b"x", "byte-length"), (b"other", "SHA-256")],
)
async def test_resume_rejects_modified_persisted_patch(modified_content: bytes, error: str) -> None:
    state = await persist_prediction(
        FakeSandbox(),
        "task-1",
        None,
        b"patch",
        task_contract_sha256=TEST_TASK_CONTRACT_SHA256,
    )
    local_root = Path(os.environ["SWEBENCH_EVAL_STATE_LOCAL_DIR"])
    (local_root / state.prediction_s3_key).write_bytes(modified_content)

    with pytest.raises(ValueError, match=error):
        await load_prediction(state)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("version", 1.0),
        ("version", "1"),
        ("prediction_size_bytes", True),
        ("prediction_size_bytes", 5.0),
        ("prediction_size_bytes", "5"),
        ("prediction_size_bytes", 1024 * 1024 * 257),
    ],
)
async def test_resume_state_rejects_non_exact_or_oversized_integer_fields(
    field: str,
    value: object,
) -> None:
    state = await persist_prediction(
        FakeSandbox(),
        "task-1",
        None,
        b"patch",
        task_contract_sha256=TEST_TASK_CONTRACT_SHA256,
    )
    data = state.model_dump(mode="json")
    data[field] = value

    with pytest.raises(ValidationError, match=field):
        EvalResumeState.model_validate(data)


async def test_resume_verifies_artifact_before_reemitting_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = service()
    state = await persist_for_service(benchmark, FakeSandbox(), None, b"patch")
    local_root = Path(os.environ["SWEBENCH_EVAL_STATE_LOCAL_DIR"])
    (local_root / state.prediction_s3_key).write_bytes(b"tampered")
    provider = FakeProvider()
    use_provider(monkeypatch, provider)
    request = EvaluateResponseRequest(
        task_id="task-1",
        eval_resume_state=state.model_dump(mode="json"),
        sandbox_provider=sandbox_provider_config(),
    )

    emitted: list[StreamChunk] = []
    with pytest.raises(ValueError, match="integrity check"):
        async for chunk in benchmark.stream_evaluate_response(request):
            emitted.append(chunk)

    assert emitted == []
    assert provider.create_request is None


async def test_resume_honors_dataset_carried_by_request(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = service()
    benchmark.datasets["candidate"] = benchmark.datasets["default"]
    state = await persist_for_service(benchmark, FakeSandbox(), "candidate", b"patch")
    provider = FakeProvider()
    provider.sandbox.captured_prediction = b"patch"
    use_provider(monkeypatch, provider)
    evaluated_datasets: list[str | None] = []

    async def evaluate_prediction(
        task_id: str,
        sandbox: Sandbox,
        prediction: str | None,
        dataset: str | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        evaluated_datasets.append(dataset)
        yield StreamResultChunk(type="result", data={})

    monkeypatch.setattr(benchmark, "_evaluate_prediction", evaluate_prediction)
    request = EvaluateResponseRequest(
        task_id="task-1",
        dataset="candidate",
        eval_resume_state=state.model_dump(mode="json"),
        sandbox_provider=sandbox_provider_config(),
    )

    _ = [chunk async for chunk in benchmark.stream_evaluate_response(request)]

    assert evaluated_datasets == ["candidate"]


def test_resume_state_rejects_path_components() -> None:
    with pytest.raises(ValidationError):
        EvalResumeState(
            benchmark_id=UUID("00000000-0000-0000-0000-000000000001"),
            task_id="../other-task",
            dataset="default",
            prediction_s3_key="swebench/eval-resume/other",
            prediction_sha256="0" * 64,
            prediction_size_bytes=1,
            task_contract_sha256=TEST_TASK_CONTRACT_SHA256,
        )


async def test_resume_sandbox_names_are_unique() -> None:
    state = await persist_prediction(
        FakeSandbox(),
        "task-1",
        None,
        b"patch",
        task_contract_sha256=TEST_TASK_CONTRACT_SHA256,
    )

    assert _resume_sandbox_name(state) != _resume_sandbox_name(state)


async def test_resume_state_rejects_noncanonical_object_key() -> None:
    state = await persist_prediction(
        FakeSandbox(),
        "task-1",
        None,
        b"patch",
        task_contract_sha256=TEST_TASK_CONTRACT_SHA256,
    )
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
        *,
        task_contract_sha256: str,
    ) -> EvalResumeState:
        del task_contract_sha256
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
    state = await persist_for_service(
        benchmark,
        FakeSandbox(captured_prediction=b""),
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


async def test_capture_rejects_oversized_remote_patch_before_download() -> None:
    benchmark = service()
    sandbox = FakeSandbox(captured_prediction=b"small fixture")

    original_exec = sandbox.exec

    async def report_oversized(
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        result = await original_exec(command, cwd=cwd, timeout=timeout)
        if PREDICTION_CAPTURE_COMMAND in command:
            return ExecResult(exit_code=0, output=str(256 * 1024 * 1024 + 1))
        return result

    sandbox.exec = report_oversized  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="size limit"):
        await benchmark._capture_prediction(sandbox)  # pyright: ignore[reportPrivateUsage]

    assert sandbox.downloads == []


async def test_capture_rejects_stream_growth_without_unbounded_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = service()
    limit = 8
    sandbox = FakeSandbox(captured_prediction=b"x" * limit)

    async def growing_command(
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[str, None]:
        del command, cwd, timeout
        yield "SWEBENCH_PREDICTION_BASE64_BEGIN\r\n"
        yield base64.b64encode(b"x" * (limit + 1)).decode()
        yield "\r\nSWEBENCH_PREDICTION_BASE64_END\r\n"

    sandbox.command = growing_command  # type: ignore[method-assign]
    sandbox.download_file = AsyncMock(side_effect=AssertionError("download_file must not be awaited"))  # type: ignore[method-assign]
    monkeypatch.setattr(service_module, "MAX_PREDICTION_BYTES", limit)

    with pytest.raises(ValueError, match="size limit"):
        await benchmark._capture_prediction(sandbox)  # pyright: ignore[reportPrivateUsage]

    sandbox.download_file.assert_not_awaited()


async def test_capture_ignores_daytona_pty_noise() -> None:
    benchmark = service()
    sandbox = FakeSandbox(captured_prediction=b"bounded")

    async def noisy_command(
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[str, None]:
        del command, cwd, timeout
        yield "root@sandbox:~# "
        yield "stty -echo\r\n"
        yield "root@sandbox:~# SWEBENCH_PREDICTION_BASE64_BEGIN\r\n"
        yield f"{base64.b64encode(sandbox.captured_prediction).decode()}\r\n"
        yield "SWEBENCH_PREDICTION_BASE64_END\r\n"

    sandbox.command = noisy_command  # type: ignore[method-assign]

    assert await benchmark._capture_prediction(sandbox) == b"bounded"  # pyright: ignore[reportPrivateUsage]


async def test_capture_propagates_command_failure_after_complete_frame() -> None:
    benchmark = service()
    sandbox = FakeSandbox(captured_prediction=b"bounded")

    async def failing_command(
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[str, None]:
        del command, cwd, timeout
        yield "SWEBENCH_PREDICTION_BASE64_BEGIN\r\n"
        yield f"{base64.b64encode(sandbox.captured_prediction).decode()}\r\n"
        yield "SWEBENCH_PREDICTION_BASE64_END\r\n"
        raise RuntimeError("sandbox command failed")

    sandbox.command = failing_command  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="sandbox command failed"):
        await benchmark._capture_prediction(sandbox)  # pyright: ignore[reportPrivateUsage]


async def test_capture_stream_supplies_finite_command_timeout() -> None:
    benchmark = service()
    sandbox = FakeSandbox(captured_prediction=b"bounded")
    observed_timeouts: list[float | None] = []
    original_command = sandbox.command

    async def recording_command(
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[str, None]:
        observed_timeouts.append(timeout)
        async for chunk in original_command(command, cwd=cwd, timeout=timeout):
            yield chunk

    sandbox.command = recording_command  # type: ignore[method-assign]

    assert await benchmark._capture_prediction(sandbox) == b"bounded"  # pyright: ignore[reportPrivateUsage]
    assert len(observed_timeouts) == 1
    timeout = observed_timeouts[0]
    assert timeout is not None and 0 < timeout < float("inf")


async def test_capture_rejects_patch_that_changes_declared_size_during_stream() -> None:
    benchmark = service()
    sandbox = FakeSandbox(captured_prediction=b"changed")

    original_exec = sandbox.exec

    async def report_smaller_size(
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        result = await original_exec(command, cwd=cwd, timeout=timeout)
        if PREDICTION_CAPTURE_COMMAND in command:
            return ExecResult(exit_code=0, output=str(len(sandbox.captured_prediction) - 1))
        return result

    sandbox.exec = report_smaller_size  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="changed size"):
        await benchmark._capture_prediction(sandbox)  # pyright: ignore[reportPrivateUsage]

    assert sandbox.downloads == []


async def test_capture_retry_replaces_read_only_file_after_lost_response() -> None:
    benchmark = service()
    sandbox = FakeSandbox(captured_prediction=b"bounded")
    attempts = 0
    original_exec = sandbox.exec

    async def lose_first_response(
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        nonlocal attempts
        if PREDICTION_CAPTURE_COMMAND in command:
            attempts += 1
            match = re.search(r">\s*(\S+)", command)
            assert match is not None
            capture_path = match.group(1)
            if attempts == 1:
                sandbox.uploads[capture_path] = sandbox.captured_prediction
                raise SandboxError("lost response")
            if not command.startswith("rm -f -- "):
                return ExecResult(exit_code=1, output="Permission denied")
            sandbox.uploads[capture_path] = sandbox.captured_prediction
            return ExecResult(exit_code=0, output=str(len(sandbox.captured_prediction)))
        return await original_exec(command, cwd=cwd, timeout=timeout)

    sandbox.exec = lose_first_response  # type: ignore[method-assign]

    assert await benchmark._capture_prediction(sandbox) == b"bounded"  # pyright: ignore[reportPrivateUsage]
    assert attempts == 2


async def test_cancelled_resume_sandbox_creation_deletes_late_created_sandbox() -> None:
    provider = FakeProvider()
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_create(_request: SandboxCreateRequest) -> Sandbox:
        started.set()
        await release.wait()
        return provider.sandbox

    provider.create_sandbox = delayed_create  # type: ignore[method-assign]
    task = asyncio.create_task(
        service_module._create_owned_sandbox(  # pyright: ignore[reportPrivateUsage]
            provider,
            resume_sandbox_request(),
        )
    )
    await started.wait()
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider.deleted == [provider.sandbox.id]


async def test_cancelled_resume_sandbox_creation_preserves_cancellation_when_creation_fails() -> None:
    provider = FakeProvider()
    started = asyncio.Event()
    release = asyncio.Event()

    async def failing_create(_request: SandboxCreateRequest) -> Sandbox:
        started.set()
        await release.wait()
        raise SandboxError("provider failed")

    provider.create_sandbox = failing_create  # type: ignore[method-assign]
    task = asyncio.create_task(
        service_module._create_owned_sandbox(  # pyright: ignore[reportPrivateUsage]
            provider,
            resume_sandbox_request(),
        )
    )
    await started.wait()
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_repeated_cancellation_still_deletes_late_created_resume_sandbox() -> None:
    provider = FakeProvider()
    started = asyncio.Event()
    release = asyncio.Event()
    deleted = asyncio.Event()

    async def delayed_create(_request: SandboxCreateRequest) -> Sandbox:
        started.set()
        await release.wait()
        return provider.sandbox

    async def record_delete(instance_id: str) -> None:
        provider.deleted.append(instance_id)
        deleted.set()

    provider.create_sandbox = delayed_create  # type: ignore[method-assign]
    provider.delete_sandbox = record_delete  # type: ignore[method-assign]
    task = asyncio.create_task(
        service_module._create_owned_sandbox(  # pyright: ignore[reportPrivateUsage]
            provider,
            resume_sandbox_request(),
        )
    )
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(deleted.wait(), timeout=1)
    assert provider.deleted == [provider.sandbox.id]


async def test_cancelled_resume_sandbox_deletion_finishes_cleanup() -> None:
    provider = FakeProvider()
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_delete(instance_id: str) -> None:
        started.set()
        await release.wait()
        provider.deleted.append(instance_id)

    provider.delete_sandbox = delayed_delete  # type: ignore[method-assign]
    task = asyncio.create_task(
        service_module._delete_owned_sandbox(  # pyright: ignore[reportPrivateUsage]
            provider,
            provider.sandbox.id,
        )
    )
    await started.wait()
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider.deleted == [provider.sandbox.id]
