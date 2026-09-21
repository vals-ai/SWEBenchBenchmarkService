import asyncio
from pathlib import Path
from typing import cast

import pytest

from swebench_service import load_dataset_from_disk, load_multimodal_dataset_from_disk


class TestRegistry:
    def test_setup_script_exists(self) -> None:
        """Verify setup.sh exists in repository root."""
        setup_sh = Path("setup.sh")
        assert setup_sh.exists(), "setup.sh not found"
        assert setup_sh.stat().st_size > 0, "setup.sh is empty"

    def test_problem_statements_exist(self) -> None:
        """Verify all 500 tasks have problem statements."""
        dataset_map = load_dataset_from_disk()

        assert len(dataset_map) == 500, f"Expected 500 tasks, got {len(dataset_map)}"

        missing = [task_id for task_id, task in dataset_map.items() if not task.get("problem_statement")]

        assert len(missing) == 0, f"Missing problem statements: {missing[:10]}"

    def test_base_commit_exists(self) -> None:
        """Verify all tasks have base_commit field."""
        dataset_map = load_dataset_from_disk()

        missing = [task_id for task_id, task in dataset_map.items() if not task.get("base_commit")]

        assert len(missing) == 0, f"Missing base_commit: {missing[:10]}"

    def test_patch_exists(self) -> None:
        """Verify all tasks have patch field."""
        dataset_map = load_dataset_from_disk()

        missing = [task_id for task_id, task in dataset_map.items() if not task.get("patch")]

        assert len(missing) == 0, f"Missing patch: {missing[:10]}"

    def test_repo_exists(self) -> None:
        """Verify all tasks have repo field."""
        dataset_map = load_dataset_from_disk()

        missing = [task_id for task_id, task in dataset_map.items() if not task.get("repo")]

        assert len(missing) == 0, f"Missing repo: {missing[:10]}"

    def test_version_exists(self) -> None:
        """Verify all tasks have version field."""
        dataset_map = load_dataset_from_disk()

        missing = [task_id for task_id, task in dataset_map.items() if not task.get("version")]

        assert len(missing) == 0, f"Missing version: {missing[:10]}"

    def test_task_ids_are_lowercase(self) -> None:
        """Lowercasing the image name is a no-op for every Verified task."""
        mixed_case = [task_id for task_id in load_dataset_from_disk() if task_id != task_id.lower()]

        assert mixed_case == [], f"Mixed-case Verified ids: {mixed_case[:10]}"

    @pytest.mark.experimental
    async def test_images_exist(self) -> None:
        """Verify all Docker images exist in registry (SLOW)."""
        from swebench_service.benchmark_service import SWEBenchService

        service = await SWEBenchService.create()
        task_ids = list(service.get_dataset().keys())

        async def check_image(task_id: str) -> tuple[str, bool]:
            try:
                response = await service.retrieve_task(task_id, skip_validation=True)
                assert response.docker_image
                return task_id, True
            except Exception:
                return task_id, False

        results = await asyncio.gather(*[check_image(tid) for tid in task_ids])
        failed = [tid for tid, success in results if not success]

        assert len(failed) == 0, f"Failed images: {failed[:10]}"


class TestMultimodalRegistry:
    """The pinned SWE-bench Multimodal dev split must be fully gradable by the pinned harness."""

    def test_every_repo_version_has_harness_specs_and_a_log_parser(self) -> None:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
        from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

        dataset_map = load_multimodal_dataset_from_disk()
        assert len(dataset_map) == 100, f"Expected 100 tasks, got {len(dataset_map)}"

        missing_specs = sorted(
            {
                (task["repo"], task["version"])
                for task in dataset_map.values()
                if task["version"] not in MAP_REPO_VERSION_TO_SPECS.get(task["repo"], {})  # type: ignore[reportUnknownMemberType]
            }
        )
        assert missing_specs == [], f"Repo versions without harness specs: {missing_specs}"

        missing_parsers = sorted({task["repo"] for task in dataset_map.values() if task["repo"] not in MAP_REPO_TO_PARSER})
        assert missing_parsers == [], f"Repos without a log parser: {missing_parsers}"

    def test_every_task_builds_an_evaluation_script(self) -> None:
        from swebench.harness.constants import SWEbenchInstance
        from swebench.harness.test_spec.test_spec import make_test_spec

        for task_id, task in load_multimodal_dataset_from_disk().items():
            test_spec = make_test_spec(cast(SWEbenchInstance, task))
            assert test_spec.eval_script, f"Empty eval script for {task_id}"
            assert test_spec.FAIL_TO_PASS, f"No FAIL_TO_PASS tests for {task_id}"

    def test_required_fields_exist(self) -> None:
        dataset_map = load_multimodal_dataset_from_disk()

        for field in ("problem_statement", "base_commit", "patch", "repo", "version", "image"):
            missing = [task_id for task_id, task in dataset_map.items() if not task.get(field)]
            assert missing == [], f"Missing {field}: {missing[:10]}"

    def test_image_assets_reference_the_problem_statement(self) -> None:
        """Every task ships an image_assets map whose problem_statement URLs appear in the text."""
        import json

        dataset_map = load_multimodal_dataset_from_disk()

        with_images = 0
        for task_id, task in dataset_map.items():
            image_assets = task["image_assets"]
            if isinstance(image_assets, str):
                image_assets = json.loads(image_assets)
            assert set(image_assets) == {"problem_statement", "patch", "test_patch"}, task_id
            urls: list[str] = image_assets["problem_statement"]
            with_images += bool(urls)
            for url in urls:
                assert url in task["problem_statement"], f"{task_id}: {url} not in problem statement"

        assert with_images > 90, f"Only {with_images} of {len(dataset_map)} tasks carry images"

    @pytest.mark.experimental
    async def test_images_exist(self) -> None:
        """Verify all multimodal Docker images resolve (SLOW)."""
        from swebench_service.benchmark_service import SWEBenchService

        service = await SWEBenchService.create()
        task_ids = list(service.get_dataset("multimodal").keys())

        async def check_image(task_id: str) -> tuple[str, bool]:
            try:
                response = await service.retrieve_task(task_id, skip_validation=True, dataset="multimodal")
                assert response.docker_image
                return task_id, True
            except Exception:
                return task_id, False

        results = await asyncio.gather(*[check_image(tid) for tid in task_ids])
        failed = [tid for tid, success in results if not success]

        assert len(failed) == 0, f"Failed images: {failed[:10]}"
