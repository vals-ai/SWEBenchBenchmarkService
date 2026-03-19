from collections.abc import Generator

import pytest

from tests.utils import BenchmarkServiceTestClient

EXPECTED_SIZES = {
    "default": 500,
    "vals_index": 102,
}


class TestDatasets:
    @pytest.fixture
    def client(self) -> Generator[BenchmarkServiceTestClient]:
        c = BenchmarkServiceTestClient()
        yield c
        c.close()

    @pytest.mark.parametrize("dataset,expected_size", EXPECTED_SIZES.items())
    async def test_dataset_loading(
        self, client: BenchmarkServiceTestClient, dataset: str, expected_size: int
    ) -> None:
        response = await client.request_verify_task_ids(dataset=dataset)
        assert response.status_code == 200
        task_ids = response.json()["task_ids"]
        assert len(task_ids) == expected_size

        response = await client.request_retrieve_task(task_ids[0], dataset=dataset)
        assert response.status_code == 200
        data = response.json()
        assert data["docker_image"]
        assert data["problem_path"]
        assert data["cwd"] == "/testbed"
