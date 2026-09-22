import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from swebench_service import (
    load_dataset_from_disk,
    load_multimodal_dataset_from_disk,
    load_multimodal_dev_dataset_from_disk,
    make_test_spec,
    test_patch_assets as patch_assets,
)


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


MULTIMODAL_DATASETS: dict[str, tuple[Callable[[], dict[str, Any]], int, int]] = {
    # dataset -> (loader, instance count, repository count)
    "multimodal": (load_multimodal_dataset_from_disk, 480, 11),
    "multimodal_dev": (load_multimodal_dev_dataset_from_disk, 100, 5),
}
GRADED_DATASETS: dict[str, Callable[[], dict[str, Any]]] = {
    "default": load_dataset_from_disk,
    **{name: loader for name, (loader, _, _) in MULTIMODAL_DATASETS.items()},
}


class TestGradingMetadata:
    """Every served row must be gradable by the pinned harness from its own columns."""

    @pytest.mark.parametrize("dataset", sorted(GRADED_DATASETS))
    def test_every_row_names_a_registered_log_parser(self, dataset: str) -> None:
        from swebench.harness.grading import PARSER_REGISTRY

        rows = GRADED_DATASETS[dataset]()
        unknown = sorted({task["log_parser"] for task in rows.values() if task["log_parser"] not in PARSER_REGISTRY})
        assert unknown == [], f"{dataset}: log parsers missing from the harness: {unknown}"

    @pytest.mark.parametrize("dataset", sorted(GRADED_DATASETS))
    def test_every_row_builds_a_test_spec(self, dataset: str) -> None:
        from swebench.harness.constants import EvalType

        for task_id, task in GRADED_DATASETS[dataset]().items():
            test_spec = make_test_spec(task)
            assert test_spec.eval_script, f"Empty eval script for {task_id}"
            assert test_spec.FAIL_TO_PASS, f"No FAIL_TO_PASS tests for {task_id}"
            assert EvalType(test_spec.eval_type)
            assert test_spec.image == task["image"]

    @pytest.mark.parametrize("dataset", sorted(GRADED_DATASETS))
    def test_image_column_matches_the_harness_naming(self, dataset: str) -> None:
        """The row's image is the name the harness builds; the service's fallback must agree."""
        mismatched = [
            task_id
            for task_id, task in GRADED_DATASETS[dataset]().items()
            if task["image"] != f"swebench/sweb.eval.x86_64.{task_id.replace('__', '_1776_').lower()}:latest"
        ]
        assert mismatched == [], f"{dataset}: {mismatched[:10]}"


class TestMultimodalRegistry:
    """The pinned SWE-bench Multimodal splits, by the numbers."""

    @pytest.mark.parametrize("dataset", sorted(MULTIMODAL_DATASETS))
    def test_split_sizes(self, dataset: str) -> None:
        loader, count, repos = MULTIMODAL_DATASETS[dataset]
        rows = loader()
        assert len(rows) == count, f"Expected {count} tasks, got {len(rows)}"
        assert len({task["repo"] for task in rows.values()}) == repos

    @pytest.mark.parametrize("dataset", sorted(MULTIMODAL_DATASETS))
    def test_required_fields_exist(self, dataset: str) -> None:
        rows = MULTIMODAL_DATASETS[dataset][0]()
        for field in ("problem_statement", "base_commit", "patch", "repo", "version", "image", "eval_script", "log_parser"):
            missing = [task_id for task_id, task in rows.items() if not task.get(field)]
            assert missing == [], f"Missing {field}: {missing[:10]}"

    @pytest.mark.parametrize("dataset", sorted(MULTIMODAL_DATASETS))
    def test_image_assets_reference_the_problem_statement(self, dataset: str) -> None:
        """Every task ships an image_assets map whose problem_statement URLs appear in the text."""
        rows = MULTIMODAL_DATASETS[dataset][0]()

        with_images = 0
        for task_id, task in rows.items():
            image_assets = task["image_assets"]
            if isinstance(image_assets, str):
                image_assets = json.loads(image_assets)
            assert set(image_assets) == {"problem_statement", "patch", "test_patch"}, task_id
            urls: list[str] = image_assets["problem_statement"]
            with_images += bool(urls)
            for url in urls:
                assert url in task["problem_statement"], f"{task_id}: {url} not in problem statement"

        assert with_images / len(rows) > 0.9, f"Only {with_images} of {len(rows)} tasks carry images"

    def test_test_patch_assets_carry_a_path_and_a_source_url(self) -> None:
        """Binary test assets must name where they land and where to fetch them; the service refuses otherwise."""
        rows = load_multimodal_dataset_from_disk()
        with_assets = 0
        for task in rows.values():
            assets = patch_assets(make_test_spec(task))
            with_assets += bool(assets)
            for asset in assets:
                assert asset["path"] and asset["url"].startswith("https://"), (task["instance_id"], asset)
        assert with_assets == 54

    @pytest.mark.experimental
    @pytest.mark.parametrize("dataset", sorted(MULTIMODAL_DATASETS))
    async def test_images_exist_on_docker_hub(self, dataset: str) -> None:
        """Every multimodal evaluation image is published under the name the service serves (SLOW, network)."""
        import httpx

        from swebench_service.benchmark_service import SWEBenchService

        service = await SWEBenchService.create()
        task_ids = list(service.get_dataset(dataset).keys())
        semaphore = asyncio.Semaphore(8)

        async def check_image(client: httpx.AsyncClient, task_id: str) -> tuple[str, bool]:
            response = await service.retrieve_task(task_id, skip_validation=True, dataset=dataset)
            repository, _, tag = response.docker_image.partition(":")
            async with semaphore:
                probe = await client.get(f"https://hub.docker.com/v2/repositories/{repository}/tags/{tag}")
            return task_id, probe.status_code == 200

        async with httpx.AsyncClient(timeout=30) as client:
            results = await asyncio.gather(*[check_image(client, tid) for tid in task_ids])
        failed = [tid for tid, exists in results if not exists]

        assert len(failed) == 0, f"Images missing from Docker Hub: {failed[:10]}"
