"""SWE-bench benchmark service implementation."""

import asyncio
import contextlib
import hashlib
import json
import logging
import shlex
from urllib.parse import urlsplit
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from benchmark_service import BenchmarkService
from benchmark_service.sandbox import (
    ExecResult,
    ImageSource,
    Sandbox,
    SandboxCommandError,
    SandboxCreateRequest,
    SandboxError,
    SandboxProvider,
)
from benchmark_service.schemas import (
    EvaluateResponseRequest,
    FinalScoreResult,
    Resources,
    RetrieveTaskResponse,
    StreamChunk,
    StreamEvalResumeStateChunk,
    StreamMessageChunk,
    StreamResultChunk,
)
import httpx

from swebench.harness.constants import TESTS_TIMEOUT

from swebench_service import (
    DISK_PATH,
    EVAL_OUTPUT_PATH,
    MULTIMODAL_DEV_DISK_PATH,
    MULTIMODAL_DISK_PATH,
    EchoedPrediction,
    asset_restore_commands,
    asset_sandbox_path,
    create_evaluation_script,
    create_run_command,
    echo_prediction,
    get_pre_install_commands,
    grade_test_output,
    load_dataset_from_disk,
    load_multimodal_dataset_from_disk,
    load_multimodal_dev_dataset_from_disk,
    load_vals_index_subset,
    make_test_spec,
    task_row_summary,
    test_patch_assets,
)
from swebench_service.test_spec import TestSpec
from swebench_service.eval_resume import MAX_PREDICTION_BYTES, EvalResumeState, load_prediction, persist_prediction
from swebench_service.utils import with_retry

logger = logging.getLogger(__name__)

PROBLEM_STATEMENT_PATH = "/tmp/problem_statement.txt"
PREDICTION_PATH = "/tmp/swebench-prediction.patch"
AGENT_BASELINE_TREE_PATH = "/tmp/swebench-agent-baseline.tree"
AGENT_BASELINE_INDEX_PATH = "/tmp/swebench-agent-baseline.index"
# Untracked files above this size are build or report artifacts (Lighthouse writes its `latest-run`
# output into the repo, test runs leave logs), never source changes: gold patches are kilobytes.
# Leaving them out keeps the patch, its result frame, and the eval script's `git apply` bounded, and
# keeps `git diff` off files a leftover agent process may still be writing.
MAX_PATCH_FILE_BYTES = 1024 * 1024
# `git add -N` / `git diff` fail when a file changes size under them (an agent's background test run
# still writing); the tree settles within seconds once the agent has exited.
CAPTURE_ATTEMPTS = 3
CAPTURE_RETRY_DELAY_SECONDS = 5.0
PREDICTION_CAPTURE_COMMAND = (
    f"umask 077; baseline=$(cat {AGENT_BASELINE_TREE_PATH}) "
    "&& exclude=$(mktemp) "
    "&& git ls-files --others --exclude-standard | while IFS= read -r f; do "
    f'[ -f "$f" ] && [ "$(stat -c %s -- "$f")" -gt {MAX_PATCH_FILE_BYTES} ] '
    # gitignore syntax: anchor at the repo root and escape its glob characters
    + r"&& printf '/%s\n' " + '"$f"' + r" | sed 's/[][*?\\]/\\&/g'; "
    + 'done > "$exclude"; '
    + "sed 's/^/excluded /' \"$exclude\" "
    + '&& git -c core.excludesFile="$exclude" add -N . '
    + '&& git diff --binary --full-index --no-ext-diff --no-textconv --no-color "$baseline" --'
)
PREDICTION_CAPTURE_PATH_PREFIX = "/tmp/swebench-prediction-capture"
COMMAND_QUIET_SECONDS = 300.0
# A test suite that stops producing output for this long is treated as timed out, as the SWE-bench
# harness treats a run over its 30-minute budget. Without a bound a hung suite (a Jest worker that
# died of a V8 heap overflow, a rendering runner waiting on a page that threw) blocks the task
# forever: the watchdog above only reports silence.
EVAL_STALL_SECONDS = 1800.0
EVAL_SANDBOX_CREATE_TIMEOUT_SECONDS = 600
EVAL_SANDBOX_AUTO_STOP_MINUTES = 15
IMAGE_DIGEST_OVERRIDES = {
    "scikit-learn__scikit-learn-12585": "sha256:438346134907344bb2444ac8f0764ffa90384cf9a4bcfc2b4b398ed95847308e",
}
MULTIMODAL_DATASET = "multimodal"
MULTIMODAL_DEV_DATASET = "multimodal_dev"
MULTIMODAL_DATASETS = frozenset({MULTIMODAL_DATASET, MULTIMODAL_DEV_DATASET})
# The Multimodal repositories run heavier suites than the Python ones: browser suites under
# Xvfb (Chart.js, openlayers, lighthouse), Puppeteer, and Jest over monorepos. They get the
# allocation the large Verified tasks already use.
MULTIMODAL_RESOURCES = Resources(vcpu=4, memory=8, disk=10)
# carbon's eval runs Jest with four workers over a monorepo; in 8 GB a worker dies of a V8 heap
# overflow and the suite hangs, and agents polling their own Jest runs crashed the same way.
MULTIMODAL_REPO_RESOURCES = {"carbon-design-system/carbon": Resources(vcpu=4, memory=16, disk=10)}
# Binary test assets (rendering baselines) are fetched from the dataset's URLs at grading time.
# Only https URLs on these hosts are fetched, redirects are refused, and a body over the cap is
# rejected, so a dataset revision cannot steer the service at internal endpoints or exhaust it.
ASSET_FETCH_TIMEOUT_SECONDS = 60.0
ASSET_HOSTS = frozenset({"raw.githubusercontent.com"})
MAX_ASSET_BYTES = 20 * 1024 * 1024
_AGENT_BASELINE_TRAP = f"""
_record_agent_baseline() {{
    setup_status=$?
    trap - EXIT
    set +e
    rm -f -- {AGENT_BASELINE_INDEX_PATH}
    GIT_INDEX_FILE={AGENT_BASELINE_INDEX_PATH} git read-tree HEAD &&
        GIT_INDEX_FILE={AGENT_BASELINE_INDEX_PATH} git add -A &&
        GIT_INDEX_FILE={AGENT_BASELINE_INDEX_PATH} git write-tree > {AGENT_BASELINE_TREE_PATH}
    baseline_status=$?
    rm -f -- {AGENT_BASELINE_INDEX_PATH}
    if [ "$baseline_status" -ne 0 ]; then
        exit "$baseline_status"
    fi
    exit "$setup_status"
}}
trap _record_agent_baseline EXIT
"""


def _build_setup_script(task: dict[str, Any]) -> str:
    setup_script = Path("setup.sh").read_text()
    if "set -euo pipefail" not in setup_script:
        raise RuntimeError("SWE-bench setup script is missing the agent baseline trap anchor")
    setup_script = setup_script.replace("set -euo pipefail", f"set -euo pipefail\n{_AGENT_BASELINE_TRAP}", 1)
    pre_install = get_pre_install_commands(task["repo"], task["version"])
    if pre_install:
        setup_script += "\n" + "\n".join(pre_install)
    return setup_script


def _resume_sandbox_name(state: EvalResumeState) -> str:
    identity = f"{state.benchmark_id}:{state.task_id}:{state.prediction_sha256}"
    task_hash = hashlib.sha256(identity.encode()).hexdigest()[:8]
    return f"swebench-eval-resume-{task_hash}-{uuid4().hex[:8]}"


class EvaluationStalled(RuntimeError):
    """The test command produced no output for EVAL_STALL_SECONDS."""


def watchdog_message(quiet_seconds: float) -> str:
    return f"[Debug]: No logs have been produced in the last {quiet_seconds:g} seconds, evaluation may be stuck"


async def _delete_owned_sandbox(provider: SandboxProvider, sandbox_id: str) -> None:
    cleanup = asyncio.create_task(provider.delete_sandbox(sandbox_id))
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        await asyncio.shield(cleanup)
        raise
    except Exception:
        logger.exception("Failed to delete SWE-bench eval-resume sandbox %s", sandbox_id)


async def _cleanup_created_sandbox(
    provider: SandboxProvider,
    creation: asyncio.Task[Sandbox],
) -> None:
    try:
        sandbox = await creation
    except Exception:
        return
    await _delete_owned_sandbox(provider, sandbox.id)


async def _create_owned_sandbox(
    provider: SandboxProvider,
    request: SandboxCreateRequest,
) -> Sandbox:
    creation = asyncio.create_task(provider.create_sandbox(request))
    try:
        return await asyncio.shield(creation)
    except asyncio.CancelledError:
        cleanup = asyncio.create_task(_cleanup_created_sandbox(provider, creation))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            pass
        raise


async def _download_bounded(sandbox: Sandbox, path: str, *, limit: int) -> bytes:
    """Download a sandbox file, refusing to buffer more than `limit` bytes.

    The capture file is read-only and its size was checked first, but a process still
    running in the sandbox could unlink and replace the path in between. Providers that
    stream (Daytona does) let the transfer stop at the limit; the plain download is the
    fallback for a provider that only offers whole-file reads.
    """
    stream = getattr(sandbox, "stream_download", None)
    if stream is None:
        return await sandbox.download_file(path)
    data = bytearray()
    async for chunk in cast(AsyncGenerator[bytes, None], stream(path)):
        data.extend(chunk)
        if len(data) > limit:
            raise ValueError(f"SWE-bench prediction exceeds the {limit}-byte size limit")
    return bytes(data)


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
        stall_seconds: float = EVAL_STALL_SECONDS,
    ) -> AsyncGenerator[str, None]:
        output: asyncio.Queue[str | None] = asyncio.Queue()
        quiet_for = 0.0

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
                    quiet_for += quiet_seconds
                    if quiet_for >= stall_seconds:
                        raise EvaluationStalled(f"no output for {quiet_for:.0f} seconds") from None
                    yield watchdog_message(quiet_seconds)
                    continue
                quiet_for = 0.0
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
        """Load SWE-bench_Verified and both SWE-bench Multimodal splits from disk."""
        for disk_path in (DISK_PATH, MULTIMODAL_DISK_PATH, MULTIMODAL_DEV_DISK_PATH):
            if not disk_path.exists():
                raise FileNotFoundError(f"Dataset not found at {disk_path}. Run 'make setup' first.")

        return {
            "default": load_dataset_from_disk(),
            "vals_index": load_vals_index_subset(),
            MULTIMODAL_DATASET: load_multimodal_dataset_from_disk(),
            MULTIMODAL_DEV_DATASET: load_multimodal_dev_dataset_from_disk(),
        }

    async def retrieve_task(
        self, task_id: str, skip_validation: bool = False, dataset: str | None = None
    ) -> RetrieveTaskResponse:
        """Retrieve task metadata for SWE-bench task."""
        if not skip_validation:
            await self.validate_task_ids([task_id], dataset=dataset)

        # The dataset row names its evaluation image (lowercase, as the harness builds them);
        # the constructed name is the fallback for rows that predate that column.
        task = self.get_dataset(dataset)[task_id]
        id_docker_compatible = task_id.replace("__", "_1776_").lower()
        image_reference: str = cast(str, task.get("image")) or f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest"
        image_repository: str = image_reference.split("@", 1)[0].rsplit(":", 1)[0]
        image_digest = IMAGE_DIGEST_OVERRIDES.get(task_id)
        docker_image = f"{image_repository}@{image_digest}" if image_digest else image_reference

        # Default: 2 vCPU, 4GB memory
        resources = Resources(vcpu=2, memory=4, disk=10)

        # Larger tasks need more resources
        if task_id in ["scikit-learn__scikit-learn-14710", "psf__requests-2317"]:
            resources.vcpu = 4
            resources.memory = 8
        if (dataset or "default") in MULTIMODAL_DATASETS:
            resources = MULTIMODAL_REPO_RESOURCES.get(cast(str, task.get("repo", "")), MULTIMODAL_RESOURCES).model_copy()

        return RetrieveTaskResponse(
            source=ImageSource(image=docker_image),
            problem_path=PROBLEM_STATEMENT_PATH,
            cwd="/testbed",
            agent_timeout=None,
            resources=resources,
        )

    def _task_contract_sha256(
        self,
        task_id: str,
        dataset: str | None,
        task_data: RetrieveTaskResponse,
    ) -> str:
        task = self.get_dataset(dataset)[task_id]
        service_dir = Path(__file__).parent
        payload = {
            "version": 2,
            "task_id": task_id,
            "dataset": dataset or "default",
            "image": task_data.source.model_dump(mode="json"),
            "setup": {
                "base_commit": task["base_commit"],
                "script_sha256": hashlib.sha256(_build_setup_script(task).encode()).hexdigest(),
            },
            "evaluator": {
                "run_command": create_run_command(task_id),
                "test_patch": task.get("test_patch"),
                "fail_to_pass": task.get("FAIL_TO_PASS"),
                "pass_to_pass": task.get("PASS_TO_PASS"),
                **task_row_summary(task),
                "benchmark_service_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "evaluation_sha256": hashlib.sha256((service_dir / "evaluation.py").read_bytes()).hexdigest(),
                "test_spec_sha256": hashlib.sha256((service_dir / "test_spec.py").read_bytes()).hexdigest(),
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

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

        # Build setup script: base + repo-specific pre-install + post-setup Git baseline.
        setup_script = _build_setup_script(task)

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

    async def stream_evaluate_response(
        self, request: EvaluateResponseRequest, dataset: str | None = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Resume evaluation from a benchmark-owned durable patch."""
        if request.response is not None or request.eval_resume_state is None:
            raise ValueError("SWE-bench eval resume requires eval_resume_state")

        state = EvalResumeState.model_validate(request.eval_resume_state)
        requested_dataset = dataset or request.dataset or "default"
        if state.task_id != request.task_id:
            raise ValueError(f"eval_resume_state task_id mismatch: {state.task_id} != {request.task_id}")
        if state.dataset != requested_dataset:
            raise ValueError(f"eval_resume_state dataset mismatch: {state.dataset} != {requested_dataset}")
        if request.sandbox_provider is None:
            raise ValueError("SWE-bench eval resume requires sandbox_provider")

        await self.validate_task_ids([request.task_id], dataset=requested_dataset)
        task_data = await self.retrieve_task(request.task_id, skip_validation=True, dataset=requested_dataset)
        task_contract_sha256 = self._task_contract_sha256(request.task_id, requested_dataset, task_data)
        if state.task_contract_sha256 != task_contract_sha256:
            raise ValueError("SWE-bench eval resume task contract does not match the current evaluator")
        prediction_bytes = await load_prediction(state)
        yield StreamEvalResumeStateChunk(type="eval_resume_state", data=state.model_dump(mode="json"))
        prediction = echo_prediction(prediction_bytes)

        async with request.sandbox_provider.create_provider() as provider:
            sandbox = await _create_owned_sandbox(
                provider,
                SandboxCreateRequest(
                    source=task_data.source,
                    resources=task_data.resources,
                    name=_resume_sandbox_name(state),
                    labels={
                        "Benchmark": "swebench",
                        "Id": str(state.benchmark_id),
                        "Task": state.task_id,
                        "EvalResume": "true",
                    },
                    env_vars={},
                    auto_stop_interval=EVAL_SANDBOX_AUTO_STOP_MINUTES,
                    create_timeout=EVAL_SANDBOX_CREATE_TIMEOUT_SECONDS,
                ),
            )
            try:
                async for chunk in self.setup_task(request.task_id, sandbox, dataset=requested_dataset):
                    if isinstance(chunk, StreamMessageChunk):
                        yield chunk

                await sandbox.upload_file(PREDICTION_PATH, prediction_bytes)
                if prediction_bytes:
                    result = await sandbox.exec(
                        f"git apply --binary {PREDICTION_PATH}",
                        cwd="/testbed",
                    )
                    if result.exit_code != 0:
                        raise RuntimeError(f"Failed to restore persisted SWE-bench prediction:\n{result.output}")

                restored_prediction = await self._capture_prediction(sandbox)
                if restored_prediction != prediction_bytes:
                    raise RuntimeError("Restored SWE-bench prediction does not match the persisted generation")
                yield StreamMessageChunk(type="message", data="Restored persisted agent prediction")

                async for chunk in self._evaluate_prediction(
                    request.task_id,
                    sandbox,
                    prediction,
                    dataset=requested_dataset,
                ):
                    yield chunk
            finally:
                await _delete_owned_sandbox(provider, sandbox.id)

    async def evaluate_instance(
        self, task_id: str, sandbox: Sandbox, dataset: str | None = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Evaluate SWE-bench solution in sandbox."""
        await self.validate_task_ids([task_id], dataset=dataset)

        # Get agent's prediction (git diff)
        yield StreamMessageChunk(type="message", data="Capturing agent's changes...")
        excluded: list[str] = []
        prediction_bytes = await self._capture_prediction(sandbox, excluded=excluded)
        if excluded:
            shown = ", ".join(excluded[:20]) + (" ..." if len(excluded) > 20 else "")
            yield StreamMessageChunk(
                type="message",
                data=f"Left {len(excluded)} untracked file(s) over {MAX_PATCH_FILE_BYTES} bytes out of the patch: {shown}",
            )

        task_data = await self.retrieve_task(task_id, skip_validation=True, dataset=dataset)
        resume_state = await persist_prediction(
            sandbox,
            task_id,
            dataset,
            prediction_bytes,
            task_contract_sha256=self._task_contract_sha256(task_id, dataset, task_data),
        )
        yield StreamEvalResumeStateChunk(type="eval_resume_state", data=resume_state.model_dump(mode="json"))
        prediction = echo_prediction(prediction_bytes)

        async for chunk in self._evaluate_prediction(task_id, sandbox, prediction, dataset=dataset):
            yield chunk

    async def _capture_prediction(self, sandbox: Sandbox, *, excluded: list[str] | None = None) -> bytes:
        """Capture the agent's patch against the post-setup baseline.

        `excluded`, when given, receives the untracked paths left out for exceeding
        MAX_PATCH_FILE_BYTES (in gitignore form: root-anchored, glob characters escaped).
        """
        capture_path = f"{PREDICTION_CAPTURE_PATH_PREFIX}-{uuid4().hex}.patch"
        quoted_capture_path = shlex.quote(capture_path)
        capture_command = (
            f"rm -f -- {quoted_capture_path} && "
            f"{PREDICTION_CAPTURE_COMMAND} > {quoted_capture_path} "
            f"&& chmod 0400 {quoted_capture_path} && stat -c %s -- {quoted_capture_path}"
        )
        try:
            result = await self._run_capture_command(sandbox, capture_command)
            lines = [line.strip() for line in result.output.strip().splitlines()]
            if excluded is not None:
                excluded.extend(line.removeprefix("excluded ") for line in lines[:-1] if line.startswith("excluded "))
            try:
                expected_size = int(lines[-1])
            except (IndexError, ValueError) as exc:
                raise RuntimeError("Failed to read captured SWE-bench prediction size") from exc
            if not 0 <= expected_size <= MAX_PREDICTION_BYTES:
                raise ValueError(f"SWE-bench prediction exceeds the {MAX_PREDICTION_BYTES}-byte size limit")
            if expected_size == 0:
                # An agent that changed nothing leaves an empty patch, and Daytona's
                # streaming download raises "No file data received" on a zero-byte file.
                return b""

            # The capture file is read-only from here on, so the size just reported bounds
            # the download. It goes through the sandbox file API: the PTY stream used before
            # could lose output across a Daytona websocket reconnect and end a large patch
            # short with no command error, which surfaced as invalid base64 on the test-split
            # tasks whose agents leave multi-megabyte artifacts in the working tree.
            prediction = cast(
                bytes, await with_retry(sandbox, lambda: _download_bounded(sandbox, capture_path, limit=MAX_PREDICTION_BYTES))
            )
            if len(prediction) != expected_size:
                raise RuntimeError(
                    "Captured SWE-bench prediction changed size between capture and download: "
                    f"expected {expected_size} bytes, got {len(prediction)}"
                )
            return prediction
        finally:
            try:
                await sandbox.exec(f"rm -f -- {capture_path}", cwd="/testbed")
            except Exception:
                logger.exception("Failed to remove SWE-bench prediction capture %s", capture_path)

    async def _run_capture_command(self, sandbox: Sandbox, capture_command: str) -> ExecResult:
        for attempt in range(1, CAPTURE_ATTEMPTS + 1):
            result = cast(ExecResult, await with_retry(sandbox, lambda: sandbox.exec(capture_command, cwd="/testbed")))
            if result.exit_code == 0:
                return result
            if attempt == CAPTURE_ATTEMPTS:
                raise RuntimeError(f"Failed to capture SWE-bench prediction:\n{result.output}")
            logger.warning("SWE-bench prediction capture attempt %d failed, retrying:\n%s", attempt, result.output)
            await asyncio.sleep(CAPTURE_RETRY_DELAY_SECONDS)
        raise AssertionError("unreachable")

    async def _stage_image_assets(self, sandbox: Sandbox, test_spec: TestSpec) -> list[str]:
        """Upload the binary assets the test patch needs and return the lines that restore them.

        A text patch cannot carry files such as expected.png rendering baselines, so the
        dataset lists them with a source URL. They are fetched here and copied into the
        working tree only after the eval script's `git apply`, so test data is never baked
        into the task image. An asset that cannot be fetched fails the evaluation: grading
        without it would score the model zero for an infrastructure fault.
        """
        assets = test_patch_assets(test_spec)
        if not assets:
            return []
        async with httpx.AsyncClient(timeout=ASSET_FETCH_TIMEOUT_SECONDS, follow_redirects=False) as client:
            for asset in assets:
                url = asset["url"]
                parts = urlsplit(url)
                if parts.scheme != "https" or parts.hostname not in ASSET_HOSTS:
                    raise RuntimeError(
                        f"Test asset {asset['path']} for {test_spec.instance_id} has a source outside the allowed "
                        f"hosts ({', '.join(sorted(ASSET_HOSTS))}): {url or '<none>'}"
                    )
                try:
                    async with client.stream("GET", url) as response:
                        response.raise_for_status()
                        data = bytearray()
                        async for chunk in response.aiter_bytes():
                            data.extend(chunk)
                            if len(data) > MAX_ASSET_BYTES:
                                raise RuntimeError(
                                    f"Test asset {asset['path']} for {test_spec.instance_id} exceeds {MAX_ASSET_BYTES} bytes"
                                )
                except httpx.HTTPError as exc:
                    raise RuntimeError(
                        f"Could not fetch test asset {asset['path']} for {test_spec.instance_id}: {exc}"
                    ) from exc
                payload = bytes(data)
                await with_retry(sandbox, lambda: sandbox.upload_file(asset_sandbox_path(asset["path"]), payload))
        return asset_restore_commands(assets)

    async def _evaluate_prediction(
        self,
        task_id: str,
        sandbox: Sandbox,
        prediction: EchoedPrediction,
        dataset: str | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Run the existing atomic SWE-bench evaluator for one captured patch."""
        task = self.get_dataset(dataset)[task_id]

        # Create and upload evaluation script, with the patch's binary assets staged so the
        # script can copy them into place after its own `git apply`.
        test_spec = make_test_spec(task)
        restore_commands = await self._stage_image_assets(sandbox, test_spec)
        if restore_commands:
            yield StreamMessageChunk(type="message", data=f"Staged {len(restore_commands)} test asset(s)")
        eval_script = create_evaluation_script(test_spec, task_id, restore_commands)
        await with_retry(sandbox, lambda: sandbox.upload_file("/root/eval.sh", eval_script.encode()))
        yield StreamMessageChunk(type="message", data="Uploaded evaluation script")

        # Execute tests with streaming. Reset output on retry so grading only sees a complete run.
        run_command = create_run_command(task_id)
        MAX_RETRIES = 3

        test_output: list[str] = []
        stalled: EvaluationStalled | None = None
        for attempt in range(MAX_RETRIES):
            test_output = []
            msg = (
                "Running tests..."
                if attempt == 0
                else f"Stream interrupted, retrying (attempt {attempt + 1}/{MAX_RETRIES})..."
            )
            yield StreamMessageChunk(type="message", data=msg)
            try:
                async for line in self.stream_command_with_watchdog(sandbox, run_command, cwd="/testbed"):
                    if line != watchdog_message(COMMAND_QUIET_SECONDS):
                        test_output.append(line)
                    yield StreamMessageChunk(type="message", data=line)
                break
            except SandboxCommandError:
                break
            except EvaluationStalled as exc:
                stalled = exc
                break
            except (SandboxError, RuntimeError):
                if attempt == MAX_RETRIES - 1:
                    raise
            await asyncio.sleep(2**attempt)

        # Grade from the faithfully-captured log file rather than the reassembled
        # PTY stream. `sandbox.command` runs the eval inside an interactive PTY, on
        # which TTY-sensitive reporters (e.g. sympy's `bin/test`) emit a
        # carriage-return progress bar with no parseable per-test lines; the tee'd
        # file (a plain pipe) preserves them. Fall back to the stream if the file
        # is unavailable (e.g. the run was interrupted before it was written).
        graded_output = "".join(test_output)
        try:
            log_file = await with_retry(sandbox, lambda: sandbox.exec(f"cat {EVAL_OUTPUT_PATH}", cwd="/testbed"))
            if log_file.output and log_file.output.strip():
                graded_output = log_file.output
        except SandboxError:
            pass
        if stalled is not None:
            # The harness marks a run over its time budget with this line and grades it unresolved.
            graded_output = f"{graded_output}\n{TESTS_TIMEOUT}\n"
            yield StreamMessageChunk(
                type="message",
                data=f"Evaluation stalled ({stalled}); graded as a test timeout, as the SWE-bench harness does",
            )

        evaluation_result = grade_test_output(graded_output, test_spec, prediction)

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
