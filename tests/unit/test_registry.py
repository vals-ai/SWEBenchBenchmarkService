import asyncio
from pathlib import Path
from typing import cast

from datasets import Dataset, load_dataset  # type: ignore

from src.logger import get_logger

logger = get_logger(__name__)


class TestRegistry:
    @property
    def registry_format(self) -> str:
        return "ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"

    async def _check_image_exists(self, image_name: str) -> bool:
        try:
            result = await asyncio.create_subprocess_exec(
                "docker",
                "manifest",
                "inspect",
                image_name,
            )

            await result.communicate()

            return result.returncode == 0
        except Exception:
            return False

    def load_dataset(self) -> Dataset:
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

        return dataset

    async def test_images_exist(self):
        dataset = self.load_dataset()
        instance_ids: list[str] = cast(list[str], [row["instance_id"] for row in dataset])  # type: ignore

        logger.info(f"Checking {len(instance_ids)} images exist")

        async def check_image_exists(instance_id: str) -> bool:
            image_name = self.registry_format.format(instance_id=instance_id)

            return await self._check_image_exists(image_name)

        results = await asyncio.gather(*[check_image_exists(instance_id) for instance_id in instance_ids])

        failed_to_fetch_images: list[str] = [
            instance_id for instance_id, result in zip(instance_ids, results) if not result
        ]

        assert len(failed_to_fetch_images) == 0, f"Failed to fetch images for {failed_to_fetch_images[:10]}..."

    async def test_problem_statements_exist(self) -> None:
        dataset = self.load_dataset()

        logger.info(f"Checking {len(dataset)} problem statements exist")

        instances_without: list[str] = []
        for row in dataset:  # type: ignore
            if not row.get("problem_statement"):  # type: ignore
                instances_without.append(row["instance_id"])  # type: ignore

        assert len(instances_without) == 0, f"Instances without problem statements: {instances_without[:10]}..."

    async def test_setup_script_exists(self) -> None:
        expected_path = Path("setup.sh")

        assert expected_path.exists(), f"Setup script not found at {expected_path}"

    async def test_base_commit_exists(self) -> None:
        dataset = self.load_dataset()

        logger.info(f"Checking {len(dataset)} base commits exist")

        instances_without_base_commit: list[str] = []
        for row in dataset:  # type: ignore
            if not row.get("base_commit"):  # type: ignore
                instances_without_base_commit.append(row["instance_id"])  # type: ignore

        assert len(instances_without_base_commit) == 0, (
            f"Instances without base commits: {instances_without_base_commit[:10]}..."
        )
