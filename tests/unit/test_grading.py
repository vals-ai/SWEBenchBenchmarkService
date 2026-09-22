"""Dataset-driven grading: eval types, parser lookup, asset restore, and pre-install parity."""

from typing import Any

import httpx
import pytest
from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT, TEST_EXIT_CODE
from swebench.harness.utils import TestSpec

from benchmark_service.schemas import StreamResultChunk

from swebench_service import (
    asset_restore_commands,
    asset_sandbox_path,
    create_evaluation_script,
    get_pre_install_commands,
    grade_test_output,
    test_patch_assets as patch_assets,
    trim_log_preamble,
)
from swebench_service.benchmark_service import SWEBenchService
from swebench_service.evaluation import MAX_ECHOED_PREDICTION_BYTES, echo_prediction


def _spec(eval_type: str, *, f2p: list[str], p2p: list[str], image_assets: dict[str, Any] | None = None) -> TestSpec:
    return TestSpec(
        instance_id="repo__project-1",
        image="swebench/sweb.eval.x86_64.repo_1776_project-1:latest",
        eval_script_list=["cd /testbed", f": '{START_TEST_OUTPUT}'", "pytest -rA", f": '{END_TEST_OUTPUT}'"],
        repo="repo/project",
        version="1.0",
        FAIL_TO_PASS=f2p,
        PASS_TO_PASS=p2p,
        log_parser="parse_log_pytest",
        eval_type=eval_type,
        image_assets=image_assets or {},
    )


def _log(*lines: str, exit_code: int | None = None) -> str:
    trailer = [f"{TEST_EXIT_CODE}: {exit_code}"] if exit_code is not None else []
    return "\n".join(["setup", START_TEST_OUTPUT, *lines, END_TEST_OUTPUT, *trailer, "done"])


class TestGradeTestOutput:
    def test_pass_and_fail_requires_every_pass_to_pass_test(self) -> None:
        spec = _spec("pass_and_fail", f2p=["tests/a.py::test_fix"], p2p=["tests/a.py::test_old"])
        result = grade_test_output(_log("PASSED tests/a.py::test_fix"), spec, "diff")
        # test_old never ran, so under pass_and_fail it is not a success
        assert result.patch_successfully_applied is True
        assert result.resolved is False
        assert result.fail_to_pass == {"success": ["tests/a.py::test_fix"], "failure": []}

    def test_fail_only_treats_an_absent_pass_to_pass_test_as_success(self) -> None:
        spec = _spec("fail_only", f2p=["tests/a.py::test_fix"], p2p=["tests/a.py::test_old"])
        result = grade_test_output(_log("PASSED tests/a.py::test_fix"), spec, "diff")
        assert result.resolved is True
        assert result.resolution_status == "RESOLVED_FULL"

    def test_a_failing_fail_to_pass_test_is_unresolved(self) -> None:
        spec = _spec("fail_only", f2p=["tests/a.py::test_fix"], p2p=[])
        result = grade_test_output(_log("FAILED tests/a.py::test_fix"), spec, "diff")
        assert result.resolved is False
        assert result.f2p_score == 0.0

    def test_a_suite_that_never_ran_is_not_a_pass_even_under_fail_only(self) -> None:
        """No parsed results and no sign the suite ran must not grade as resolved."""
        spec = _spec("fail_only", f2p=["tests/a.py::test_fix"], p2p=[])
        result = grade_test_output(_log("bash: pytest: command not found"), spec, "diff")
        assert result.patch_successfully_applied is False
        assert result.resolved is False

    def test_a_nonzero_exit_with_only_passes_reported_invalidates_the_run(self) -> None:
        """A patch can print its own PASSED lines; the recorded exit status exposes that."""
        spec = _spec("fail_only", f2p=["tests/a.py::test_fix"], p2p=[])
        result = grade_test_output(_log("PASSED tests/a.py::test_fix", exit_code=1), spec, "diff")
        assert result.patch_successfully_applied is False
        assert result.resolved is False

    def test_a_nonzero_exit_with_a_reported_failure_is_graded_normally(self) -> None:
        spec = _spec("pass_and_fail", f2p=["tests/a.py::test_fix"], p2p=["tests/a.py::test_old"])
        result = grade_test_output(
            _log("PASSED tests/a.py::test_fix", "FAILED tests/a.py::test_old", exit_code=1), spec, "diff"
        )
        assert result.patch_successfully_applied is True
        assert result.resolved is False
        assert result.pass_to_pass == {"success": [], "failure": ["tests/a.py::test_old"]}

    def test_a_zero_exit_grades_normally(self) -> None:
        spec = _spec("fail_only", f2p=["tests/a.py::test_fix"], p2p=[])
        assert grade_test_output(_log("PASSED tests/a.py::test_fix", exit_code=0), spec, "diff").resolved is True

    def test_missing_markers_mean_the_patch_did_not_apply(self) -> None:
        spec = _spec("pass_and_fail", f2p=["tests/a.py::test_fix"], p2p=[])
        result = grade_test_output("error: patch failed", spec, "diff")
        assert result.patch_successfully_applied is False


class TestEvaluationScript:
    def test_restore_commands_land_before_the_test_output_marker(self) -> None:
        spec = _spec("pass_and_fail", f2p=["t"], p2p=[])
        restore = asset_restore_commands([{"path": "cases/a/expected.png", "url": "https://x/e.png"}])

        script = create_evaluation_script(spec, spec.instance_id, restore)

        lines = script.split("\n")
        marker = next(i for i, line in enumerate(lines) if START_TEST_OUTPUT in line)
        assert lines[marker - 1] == restore[0]
        assert restore[0] == "mkdir -p $(dirname cases/a/expected.png) && cp /image_assets/cases__a__expected.png cases/a/expected.png"
        assert asset_sandbox_path("cases/a/expected.png") == "/image_assets/cases__a__expected.png"

    def test_asset_paths_are_shell_quoted(self) -> None:
        """A dataset path is data, not shell; it must not be able to run commands in the sandbox."""
        [line] = asset_restore_commands([{"path": "a b/$(touch pwned).png", "url": "https://x/e.png"}])
        assert line == (
            "mkdir -p $(dirname 'a b/$(touch pwned).png') && "
            "cp '/image_assets/a b__$(touch pwned).png' 'a b/$(touch pwned).png'"
        )

    def test_without_assets_the_script_is_the_rows_eval_script(self) -> None:
        spec = _spec("pass_and_fail", f2p=["t"], p2p=[])
        assert create_evaluation_script(spec, spec.instance_id, []) == spec.eval_script
        assert create_evaluation_script(spec, spec.instance_id) == spec.eval_script

    def test_the_preamble_logs_headers_instead_of_the_whole_repository(self) -> None:
        """The images have a squashed history, so `git show` there is the entire repo as one diff."""
        base = "716923f458c2ba90b5a4ec3ab41dcae8bc0a9917"
        preamble = [
            "#!/bin/bash",
            "set -uxo pipefail",
            "cd /testbed",
            "git config --global --add safe.directory /testbed",
            "source $NVM_DIR/nvm.sh",
            "git status",
            "git show",
            f"git -c core.fileMode=false diff {base}",
            f"git checkout {base} tests/languages/fsharp/keyword_feature.test",
        ]
        spec = _spec("pass_and_fail", f2p=["t"], p2p=[])
        spec.eval_script_list = [*preamble, *spec.eval_script_list]
        script = create_evaluation_script(spec, spec.instance_id)
        lines = script.split("\n")
        assert "git show --no-patch" in lines and "git show" not in lines
        assert f"git -c core.fileMode=false diff --stat {base}" in lines
        assert f"git -c core.fileMode=false diff {base}" not in lines
        # Everything else, including the test-file checkout that names the same commit, is untouched.
        assert script.endswith("\n".join(spec.eval_script_list[-4:]) + "\n")
        assert f"git checkout {base} tests/languages/fsharp/keyword_feature.test" in lines

    def test_trim_leaves_other_git_commands_alone(self) -> None:
        script = "git show --stat\ngit diff HEAD -- package.json\ngit show HEAD:file\n: 'marker'"
        assert trim_log_preamble(script) == script

    def test_patch_assets_read_path_and_url_from_both_patch_lists(self) -> None:
        spec = _spec(
            "pass_and_fail",
            f2p=["t"],
            p2p=[],
            image_assets={
                "problem_statement": ["https://x/shot.png"],
                "test_patch": [{"path": "a.png", "url": "https://x/a.png"}],
                "patch": [{"path": "b.png", "url": "https://x/b.png"}, {"url": "https://x/no-path.png"}],
            },
        )
        assert patch_assets(spec) == [
            {"path": "a.png", "url": "https://x/a.png"},
            {"path": "b.png", "url": "https://x/b.png"},
        ]


class TestPreInstall:
    def test_verified_pre_install_table_matches_the_old_harness(self) -> None:
        assert get_pre_install_commands("django/django", "2.2") == [
            "apt-get update && apt-get install -y locales",
            "echo 'en_US UTF-8' > /etc/locale.gen",
            "locale-gen en_US.UTF-8",
        ]
        assert get_pre_install_commands("astropy/astropy", "5.1")[0].startswith("sed -i")
        assert get_pre_install_commands("django/django", "4.2") == []
        assert get_pre_install_commands("chartjs/Chart.js", "2.0") == []


GOOD = "https://raw.githubusercontent.com/org/repo/abc123/cases/a/expected.png"
MISSING = "https://raw.githubusercontent.com/org/repo/abc123/cases/a/missing.png"


class _FakeSandbox:
    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}
        self.id = "sandbox"

    async def upload_file(self, path: str, data: bytes) -> None:
        self.uploads[path] = data


class TestAssetStaging:
    @pytest.fixture
    def service(self) -> SWEBenchService:
        return SWEBenchService.__new__(SWEBenchService)

    def _transport(self, responses: dict[str, tuple[int, bytes]]) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            status, body = responses.get(str(request.url), (404, b""))
            return httpx.Response(status, content=body)

        return httpx.MockTransport(handler)

    async def _stage(
        self, service: SWEBenchService, spec: TestSpec, responses: dict[str, tuple[int, bytes]], monkeypatch: pytest.MonkeyPatch
    ) -> tuple[list[str], _FakeSandbox]:
        transport = self._transport(responses)
        real_client = httpx.AsyncClient

        def patched_client(**kwargs: Any) -> httpx.AsyncClient:
            return real_client(transport=transport, **kwargs)

        monkeypatch.setattr("swebench_service.benchmark_service.httpx.AsyncClient", patched_client)
        sandbox = _FakeSandbox()
        restore = await service._stage_image_assets(sandbox, spec)  # pyright: ignore[reportPrivateUsage, reportArgumentType]
        return restore, sandbox

    async def test_assets_are_uploaded_and_restore_lines_returned(self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = _spec("pass_and_fail", f2p=["t"], p2p=[], image_assets={"test_patch": [{"path": "c/e.png", "url": GOOD}]})
        restore, sandbox = await self._stage(service, spec, {GOOD: (200, b"PNG")}, monkeypatch)
        assert sandbox.uploads == {"/image_assets/c__e.png": b"PNG"}
        assert restore == ["mkdir -p $(dirname c/e.png) && cp /image_assets/c__e.png c/e.png"]

    async def test_no_assets_means_no_uploads(self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = _spec("pass_and_fail", f2p=["t"], p2p=[])
        restore, sandbox = await self._stage(service, spec, {}, monkeypatch)
        assert restore == [] and sandbox.uploads == {}

    async def test_an_unfetchable_asset_fails_the_evaluation(self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch) -> None:
        """Grading without the baseline would score the model zero for an infrastructure fault."""
        spec = _spec("pass_and_fail", f2p=["t"], p2p=[], image_assets={"test_patch": [{"path": "c/e.png", "url": MISSING}]})
        with pytest.raises(RuntimeError, match="Could not fetch test asset c/e.png"):
            await self._stage(service, spec, {}, monkeypatch)

    @pytest.mark.parametrize(
        "url",
        [
            "http://raw.githubusercontent.com/o/r/c/e.png",
            "https://evil.example.com/e.png",
            "https://169.254.169.254/latest/meta-data/",
            "file:///etc/passwd",
        ],
    )
    async def test_an_asset_outside_the_allowed_hosts_is_refused_without_a_request(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch, url: str
    ) -> None:
        spec = _spec("pass_and_fail", f2p=["t"], p2p=[], image_assets={"test_patch": [{"path": "c/e.png", "url": url}]})
        with pytest.raises(RuntimeError, match="outside the allowed hosts"):
            await self._stage(service, spec, {url: (200, b"PNG")}, monkeypatch)

    async def test_a_redirect_is_refused(self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = _spec("pass_and_fail", f2p=["t"], p2p=[], image_assets={"test_patch": [{"path": "c/e.png", "url": GOOD}]})
        with pytest.raises(RuntimeError, match="Could not fetch test asset"):
            await self._stage(service, spec, {GOOD: (302, b"")}, monkeypatch)

    async def test_an_oversized_asset_is_refused(self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("swebench_service.benchmark_service.MAX_ASSET_BYTES", 8)
        spec = _spec("pass_and_fail", f2p=["t"], p2p=[], image_assets={"test_patch": [{"path": "c/e.png", "url": GOOD}]})
        with pytest.raises(RuntimeError, match="exceeds 8 bytes"):
            await self._stage(service, spec, {GOOD: (200, b"PNG" * 10)}, monkeypatch)

    async def test_an_asset_without_a_url_fails_the_evaluation(self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = _spec("pass_and_fail", f2p=["t"], p2p=[], image_assets={"test_patch": [{"path": "c/e.png"}]})
        with pytest.raises(RuntimeError, match="outside the allowed hosts .*<none>"):
            await self._stage(service, spec, {}, monkeypatch)



class TestEchoedPrediction:
    """The result is one WebSocket frame and the framework client drops frames over 10 MiB."""

    def test_a_small_patch_is_echoed_whole(self) -> None:
        spec = _spec("fail_only", f2p=["tests/a.py::test_fix"], p2p=[])
        patch = "diff --git a b\n+é\n".encode()
        result = grade_test_output(_log("PASSED tests/a.py::test_fix"), spec, echo_prediction(patch))
        assert result.prediction == "diff --git a b\n+é\n"
        assert result.prediction_bytes == len(patch)
        assert result.prediction_truncated is False

    def test_a_plain_string_is_echoed_as_is(self) -> None:
        spec = _spec("fail_only", f2p=["tests/a.py::test_fix"], p2p=[])
        result = grade_test_output(_log("PASSED tests/a.py::test_fix"), spec, "diff")
        assert (result.prediction, result.prediction_bytes, result.prediction_truncated) == ("diff", 4, False)

    def test_no_patch_stays_none(self) -> None:
        spec = _spec("fail_only", f2p=["tests/a.py::test_fix"], p2p=[])
        for prediction in (None, echo_prediction(b"")):
            result = grade_test_output(_log("bash: pytest: command not found"), spec, prediction)
            assert result.prediction is None
            assert result.prediction_bytes is None
            assert result.prediction_truncated is False

    def test_a_patch_bloated_by_build_artifacts_is_cut_and_the_frame_stays_under_the_limit(self) -> None:
        spec = _spec("fail_only", f2p=["tests/a.py::test_fix"], p2p=[])
        patch = b"diff --git a/latest-run/artifacts.json b/latest-run/artifacts.json\n" + b"+x" * (12 * 1024 * 1024)
        result = grade_test_output(_log("FAILED tests/a.py::test_fix"), spec, echo_prediction(patch))
        assert result.prediction is not None
        assert result.prediction.startswith("diff --git a/latest-run/artifacts.json")
        assert len(result.prediction.encode()) == MAX_ECHOED_PREDICTION_BYTES
        assert result.prediction_bytes == len(patch)
        assert result.prediction_truncated is True
        assert result.resolved is False
        frame = StreamResultChunk(type="result", data=result.model_dump()).model_dump_json()
        assert len(frame.encode()) < 10 * 1024 * 1024

    def test_the_cut_never_splits_a_multibyte_character(self) -> None:
        # One ASCII byte shifts every two-byte "é" off the bound, so the cut lands inside one of them.
        patch = ("a" + "é" * (MAX_ECHOED_PREDICTION_BYTES // 2)).encode()
        echoed = echo_prediction(patch)
        assert echoed.text == "a" + "é" * (MAX_ECHOED_PREDICTION_BYTES // 2 - 1)
        assert echoed.text is not None and "\ufffd" not in echoed.text
        assert echoed == (echoed.text, len(patch), True)

    def test_a_cut_on_a_character_boundary_keeps_the_whole_head(self) -> None:
        patch = ("é" * (MAX_ECHOED_PREDICTION_BYTES // 2 + 1)).encode()
        echoed = echo_prediction(patch)
        assert echoed.text == "é" * (MAX_ECHOED_PREDICTION_BYTES // 2)
        assert echoed.truncated is True
