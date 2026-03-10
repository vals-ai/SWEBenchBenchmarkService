import asyncio
from pathlib import Path

import pytest

from swebench_service import load_dataset_from_disk


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
