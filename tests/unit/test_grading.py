"""Dataset-driven grading: eval types, parser lookup, asset restore, and pre-install parity."""

import ipaddress
import json
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
from swebench_service.benchmark_service import MAX_PROBLEM_IMAGE_BYTES, SWEBenchService
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

    @staticmethod
    def _browser_spec(repo: str, test_line: str) -> TestSpec:
        spec = _spec("pass_and_fail", f2p=["t"], p2p=[])
        spec.repo = repo
        spec.eval_script_list = ["#!/bin/bash", "set -uxo pipefail", "cd /testbed", START_TEST_OUTPUT, test_line]
        return spec

    def test_an_openlayers_browser_row_without_a_chrome_path_gets_one(self) -> None:
        """Without it the runner cannot launch a browser in the image, whatever the patch does."""
        spec = self._browser_spec("openlayers/openlayers", 'su chromeuser -c "npm run test-browser"')
        lines = create_evaluation_script(spec, spec.instance_id).split("\n")
        export = "export PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome-stable"
        assert lines.count(export) == 1
        assert lines[lines.index("set -uxo pipefail") + 1] == export

    def test_other_rows_keep_their_eval_script_without_a_chrome_path(self) -> None:
        test_browser = 'su chromeuser -c "npm run test-browser"'
        cases = [
            ("openlayers/openlayers", "PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome-stable " + test_browser),
            ("openlayers/openlayers", "npm run test-node"),
            ("other/project", test_browser),
        ]
        for repo, test_line in cases:
            spec = self._browser_spec(repo, test_line)
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


_MockResponse = tuple[int, bytes] | tuple[int, bytes, dict[str, str]]


class TestProblemImageStaging:
    """Screenshots are staged during setup, when the sandbox still has egress.

    The agent runs with its network restricted to the model gateway, so an image it cannot
    fetch itself has to already be on disk. The URLs come from the dataset's own
    `image_assets.problem_statement`, which is the list upstream SWE-agent stages.
    """

    PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 32
    JPG = b"\xff\xd8\xff" + b"y" * 16

    @pytest.fixture
    def service(self) -> SWEBenchService:
        return SWEBenchService.__new__(SWEBenchService)

    @pytest.fixture(autouse=True)
    def public_dns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolve every test host to a public address unless a test says otherwise."""
        monkeypatch.setattr(
            "swebench_service.benchmark_service._resolve_addresses",
            self._resolver({}),
        )

    @staticmethod
    def _resolver(overrides: dict[str, str], default: str = "93.184.216.34") -> Any:
        async def resolve(host: str, port: int) -> list[Any]:
            return [ipaddress.ip_address(overrides.get(host, default))]

        return resolve

    @staticmethod
    def _transport(responses: "dict[str, _MockResponse]") -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            # The fetch connects to a checked address, so the request URL holds that address
            # and the name it stands for is in the Host header.
            host = request.headers.get("Host") or request.url.netloc.decode()
            path = request.url.raw_path.decode()
            entry = responses.get(f"{request.url.scheme}://{host}{path}")
            if entry is None:
                return httpx.Response(404, content=b"<html>not found</html>")
            status, body = entry[0], entry[1]
            headers: dict[str, str] = entry[2] if len(entry) == 3 else {}
            return httpx.Response(status, content=body, headers=headers)

        return httpx.MockTransport(handler)

    async def _stage(
        self,
        service: SWEBenchService,
        urls: list[str],
        responses: "dict[str, _MockResponse]",
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[dict[str, str], list[str], _FakeSandbox]:
        transport = self._transport(responses)
        real_client = httpx.AsyncClient

        def patched_client(**kwargs: Any) -> httpx.AsyncClient:
            return real_client(transport=transport, **kwargs)

        monkeypatch.setattr("swebench_service.benchmark_service.httpx.AsyncClient", patched_client)
        sandbox = _FakeSandbox()
        task = {"image_assets": {"problem_statement": urls}}
        manifest, unstaged = await service._stage_problem_images(sandbox, task)  # pyright: ignore[reportPrivateUsage, reportArgumentType]
        return manifest, unstaged, sandbox

    async def test_declared_images_land_in_the_sandbox_with_a_manifest(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = "https://user-images.githubusercontent.com/1/shot.png"

        manifest, unstaged, sandbox = await self._stage(service, [url], {url: (200, self.PNG)}, monkeypatch)

        assert list(manifest) == [url]
        assert unstaged == []
        local = manifest[url]
        assert local.startswith("/problem_images/") and local.endswith(".png")
        assert sandbox.uploads[local] == self.PNG
        assert json.loads(sandbox.uploads["/problem_images/manifest.json"]) == manifest

    async def test_any_host_the_dataset_lists_is_staged(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reporters embed screenshots wherever they like; an allowlist of GitHub CDNs blinded
        the agent on 32 tasks whose images live on Alibaba OSS or GitHub's older CDN."""
        urls = [
            "https://cloud.githubusercontent.com/assets/1/old.png",
            "https://fusion-image.oss-cn-beijing.aliyuncs.com/shot.png",
            "https://img.alicdn.com/tfs/shot.png",
            "https://guoxicheng.top/images/shot.png",
        ]

        manifest, unstaged, _ = await self._stage(
            service, urls, {url: (200, self.PNG) for url in urls}, monkeypatch
        )

        assert list(manifest) == urls
        assert unstaged == []

    async def test_an_image_that_cannot_be_fetched_is_reported_not_fatal(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Some of these links have been dead for years; upstream drops them too."""
        good = "https://user-images.githubusercontent.com/1/good.png"
        bad = "https://user-images.githubusercontent.com/2/gone.png"

        manifest, unstaged, sandbox = await self._stage(
            service, [good, bad], {good: (200, self.PNG), bad: (404, b"")}, monkeypatch
        )

        assert list(manifest) == [good]
        assert unstaged == [bad]
        assert not any("gone" in path for path in sandbox.uploads)

    async def test_a_page_that_is_not_an_image_is_refused(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dead link often answers 200 with HTML, and a tile API answers text/plain."""
        url = "https://tile.nextzen.org/tile/1/2/3.png"

        manifest, unstaged, _ = await self._stage(
            service,
            [url],
            {url: (200, b"<html>gone</html>", {"content-type": "text/html"})},
            monkeypatch,
        )

        assert manifest == {}
        assert unstaged == [url]

    async def test_an_image_type_with_no_magic_bytes_is_kept(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = "https://example.org/diagram.svg"

        manifest, unstaged, _ = await self._stage(
            service,
            [url],
            {url: (200, b"<svg xmlns='http://www.w3.org/2000/svg'/>", {"content-type": "image/svg+xml"})},
            monkeypatch,
        )

        assert unstaged == []
        assert manifest[url].endswith(".svg")

    async def test_the_served_type_wins_over_the_link_extension(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A .png link answering with SVG must not be stored as a PNG."""
        url = "https://images.example.org/preview.png"

        manifest, unstaged, _ = await self._stage(
            service,
            [url],
            {url: (200, b"<svg xmlns='http://www.w3.org/2000/svg'/>", {"content-type": "image/svg+xml"})},
            monkeypatch,
        )

        assert unstaged == []
        assert manifest[url].endswith(".svg")

    async def test_the_extension_comes_from_the_bytes_when_the_url_has_none(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = "https://user-images.githubusercontent.com/1/0bd4-11eb-8a1c"  # GitHub's extensionless form

        manifest, _, _ = await self._stage(service, [url], {url: (200, self.JPG)}, monkeypatch)

        assert manifest[url].endswith(".jpg")

    async def test_a_task_with_no_declared_images_stages_nothing(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest, unstaged, sandbox = await self._stage(service, [], {}, monkeypatch)
        assert manifest == {} and unstaged == [] and sandbox.uploads == {}

    async def test_the_same_image_twice_is_fetched_once(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = "https://user-images.githubusercontent.com/1/shot.png"
        manifest, _, _ = await self._stage(service, [url, url], {url: (200, self.PNG)}, monkeypatch)
        assert list(manifest) == [url]

    async def test_a_github_blob_link_fetches_the_file_not_the_page(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A statement that links a screenshot by its repository page used to stage 230 KB of
        HTML under a .png name; the agent then dropped it for having the wrong magic bytes."""
        blob = "https://github.com/o/r/blob/master/Reference/A.4.1.png"
        raw = "https://raw.githubusercontent.com/o/r/master/Reference/A.4.1.png"

        manifest, unstaged, sandbox = await self._stage(
            service,
            [blob],
            {blob: (200, b"<html>page</html>", {"content-type": "text/html"}), raw: (200, self.PNG)},
            monkeypatch,
        )

        assert unstaged == []
        # The statement references the blob URL, so that is what the agent looks up.
        assert list(manifest) == [blob]
        assert sandbox.uploads[manifest[blob]] == self.PNG

    # --- what the host allowlist used to be standing in for ---------------------------------

    async def test_a_plain_http_url_is_refused(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = "http://user-images.githubusercontent.com/1/shot.png"
        manifest, unstaged, _ = await self._stage(service, [url], {url: (200, self.PNG)}, monkeypatch)
        assert manifest == {} and unstaged == [url]

    @pytest.mark.parametrize(
        "address",
        ["127.0.0.1", "169.254.169.254", "10.0.0.5", "192.168.1.1", "::1", "fd00::1"],
    )
    async def test_a_host_on_our_own_network_is_refused(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch, address: str
    ) -> None:
        """The URL list comes from an external dataset, so this is the control that keeps a
        revision from pointing the service at a metadata endpoint or an internal service."""
        url = "https://metadata.example/shot.png"
        monkeypatch.setattr(
            "swebench_service.benchmark_service._resolve_addresses",
            self._resolver({"metadata.example": address}),
        )

        manifest, unstaged, _ = await self._stage(service, [url], {url: (200, self.PNG)}, monkeypatch)

        assert manifest == {} and unstaged == [url]

    async def test_a_host_is_refused_when_any_of_its_addresses_is_private(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A name can hold several records, so one public answer is not enough."""
        url = "https://split.example/shot.png"

        async def resolve(host: str, port: int) -> list[Any]:
            return [ipaddress.ip_address("93.184.216.34"), ipaddress.ip_address("169.254.169.254")]

        monkeypatch.setattr("swebench_service.benchmark_service._resolve_addresses", resolve)

        manifest, unstaged, _ = await self._stage(service, [url], {url: (200, self.PNG)}, monkeypatch)

        assert manifest == {} and unstaged == [url]

    async def test_the_fetch_connects_to_the_address_that_was_checked(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Connecting by name would resolve it a second time, so a host that answers
        publicly for the check could answer privately for the fetch."""
        url = "https://rebind.example/shot.png"
        seen: list[str] = []

        async def resolve(host: str, port: int) -> list[Any]:
            return [ipaddress.ip_address("93.184.216.34")]

        monkeypatch.setattr("swebench_service.benchmark_service._resolve_addresses", resolve)

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            assert request.headers["Host"] == "rebind.example"
            assert request.extensions.get("sni_hostname") == "rebind.example"
            return httpx.Response(200, content=self.PNG)

        real_client = httpx.AsyncClient

        def patched_client(**kwargs: Any) -> httpx.AsyncClient:
            return real_client(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr("swebench_service.benchmark_service.httpx.AsyncClient", patched_client)
        sandbox = _FakeSandbox()
        manifest, unstaged = await service._stage_problem_images(  # pyright: ignore[reportPrivateUsage]
            sandbox,  # pyright: ignore[reportArgumentType]
            {"image_assets": {"problem_statement": [url]}},
        )

        assert unstaged == []
        assert list(manifest) == [url]
        assert seen == ["https://93.184.216.34/shot.png"]

    async def test_a_redirect_into_our_network_is_refused(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The old fetch followed redirects blindly, so an allowed host could hand the service
        any destination it liked; every hop is checked now."""
        url = "https://user-images.githubusercontent.com/1/shot.png"
        inside = "https://metadata.example/secret"
        monkeypatch.setattr(
            "swebench_service.benchmark_service._resolve_addresses",
            self._resolver({"metadata.example": "169.254.169.254"}),
        )

        manifest, unstaged, _ = await self._stage(
            service,
            [url],
            {url: (302, b"", {"location": inside}), inside: (200, self.PNG)},
            monkeypatch,
        )

        assert manifest == {} and unstaged == [url]

    async def test_a_redirect_to_another_public_host_is_followed(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GitHub's camo and the OSS CDNs redirect in normal operation."""
        url = "https://github.com/o/r/assets/1"
        final = "https://objects.githubusercontent.com/shot.png"

        manifest, unstaged, _ = await self._stage(
            service,
            [url],
            {url: (302, b"", {"location": final}), final: (200, self.PNG)},
            monkeypatch,
        )

        assert unstaged == []
        assert list(manifest) == [url]

    async def test_a_redirect_loop_gives_up(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a = "https://example.org/a.png"
        b = "https://example.org/b.png"

        manifest, unstaged, _ = await self._stage(
            service,
            [a],
            {a: (302, b"", {"location": b}), b: (302, b"", {"location": a})},
            monkeypatch,
        )

        assert manifest == {} and unstaged == [a]

    async def test_an_oversized_image_is_refused(
        self, service: SWEBenchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = "https://example.org/huge.png"
        body = b"\x89PNG\r\n\x1a\n" + b"x" * (MAX_PROBLEM_IMAGE_BYTES + 1)

        manifest, unstaged, _ = await self._stage(service, [url], {url: (200, body)}, monkeypatch)

        assert manifest == {} and unstaged == [url]


class TestUpstreamExactGrading:
    """Multimodal grades the log as the SWE-bench harness does, with none of the stream repairs."""

    _KARMA_FAILURE = "Chrome Headless 120.0.0 (Linux x86_64) ol.Map renders FAILED"

    def _openlayers_spec(self) -> TestSpec:
        spec = _spec("fail_only", f2p=["ol.Map renders"], p2p=[])
        spec.log_parser = "parse_log_openlayers"
        return spec

    def test_crlf_line_ends_are_read_as_the_harness_reads_them(self) -> None:
        log = _log(self._KARMA_FAILURE).replace("\n", "\r\n")
        assert grade_test_output(log, self._openlayers_spec(), "diff", upstream_exact=True).resolved is False

    def test_a_lone_carriage_return_ends_a_line(self) -> None:
        log = _log(self._KARMA_FAILURE).replace("\n", "\r")
        assert grade_test_output(log, self._openlayers_spec(), "diff", upstream_exact=True).resolved is False

    def test_control_characters_in_a_test_name_are_kept(self) -> None:
        spec = _spec("pass_and_fail", f2p=["tests/a.py::test_fix"], p2p=["tests/a.py::test_ok"])
        log = _log("PASSED tests/a.py::test_fix", "PASSED tests/a.py::test_ok​")
        assert grade_test_output(log, spec, "diff").resolved is True
        exact = grade_test_output(log, spec, "diff", upstream_exact=True)
        assert exact.resolved is False
        assert exact.pass_to_pass == {"success": [], "failure": ["tests/a.py::test_ok"]}

    def test_status_words_are_not_split_off_a_preceding_token(self) -> None:
        spec = _spec("pass_and_fail", f2p=["tests/a.py::test_fix"], p2p=[])
        log = _log("tests/a.py::test_fixPASSED tests/a.py::test_fix")
        assert grade_test_output(log, spec, "diff").status_map == {"tests/a.py::test_fix": "PASSED"}
        # The fused token is left alone, so nothing parses and the run is not a pass.
        assert grade_test_output(log, spec, "diff", upstream_exact=True).resolved is False

    def test_a_failed_test_that_never_ran_is_still_a_pass_under_fail_only(self) -> None:
        # Upstream's own behaviour, which this service must reproduce rather than repair.
        log = _log("some unrelated output")
        spec = self._openlayers_spec()
        result = grade_test_output(log + "\nChrome Headless 1.0 (Linux x86_64) other FAILED\n", spec, "diff", upstream_exact=True)
        assert result.resolved is True


def test_clean_pty_stream_gives_back_the_text_the_log_file_would_hold() -> None:
    from swebench_service.evaluation import clean_pty_stream

    assert clean_pty_stream("\x1b[32mok\x1b[0m a\r\nb\rc\n") == "ok a\nb\nc\n"
    assert clean_pty_stream("\x1b]0;title\x07\x1b(Bq\x1b7r") == "qr"
