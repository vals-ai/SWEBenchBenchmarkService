"""Dataset-driven grading: eval types, parser lookup, asset restore, and pre-install parity."""

from typing import Any

import httpx
import pytest
from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
from swebench.harness.utils import TestSpec

from swebench_service import (
    asset_restore_commands,
    asset_sandbox_path,
    create_evaluation_script,
    get_pre_install_commands,
    grade_test_output,
    test_patch_assets as patch_assets,
)
from swebench_service.benchmark_service import SWEBenchService


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


def _log(*lines: str) -> str:
    return "\n".join(["setup", START_TEST_OUTPUT, *lines, END_TEST_OUTPUT, "done"])


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

    def test_without_assets_the_script_is_the_rows_eval_script(self) -> None:
        spec = _spec("pass_and_fail", f2p=["t"], p2p=[])
        assert create_evaluation_script(spec, spec.instance_id, []) == spec.eval_script
        assert create_evaluation_script(spec, spec.instance_id) == spec.eval_script

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
        spec = _spec("pass_and_fail", f2p=["t"], p2p=[], image_assets={"test_patch": [{"path": "c/e.png", "url": "https://x/e.png"}]})
        restore, sandbox = await self._stage(service, spec, {"https://x/e.png": (200, b"PNG")}, monkeypatch)
        assert sandbox.uploads == {"/image_assets/c__e.png": b"PNG"}
        assert restore == ["mkdir -p $(dirname c/e.png) && cp /image_assets/c__e.png c/e.png"]

    async def test_no_assets_means_no_uploads(self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = _spec("pass_and_fail", f2p=["t"], p2p=[])
        restore, sandbox = await self._stage(service, spec, {}, monkeypatch)
        assert restore == [] and sandbox.uploads == {}

    async def test_an_unfetchable_asset_fails_the_evaluation(self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch) -> None:
        """Grading without the baseline would score the model zero for an infrastructure fault."""
        spec = _spec("pass_and_fail", f2p=["t"], p2p=[], image_assets={"test_patch": [{"path": "c/e.png", "url": "https://x/missing.png"}]})
        with pytest.raises(RuntimeError, match="Could not fetch test asset c/e.png"):
            await self._stage(service, spec, {}, monkeypatch)

    async def test_an_asset_without_a_url_fails_the_evaluation(self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = _spec("pass_and_fail", f2p=["t"], p2p=[], image_assets={"test_patch": [{"path": "c/e.png"}]})
        with pytest.raises(RuntimeError, match="has no source URL"):
            await self._stage(service, spec, {}, monkeypatch)

