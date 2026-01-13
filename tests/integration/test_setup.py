from pathlib import Path

from pytest import MonkeyPatch

from src.utils import load_dataset_from_disk


class TestSetup:
    async def test_setup(self, monkeypatch: MonkeyPatch, tmp_path: Path, task_directory: Path) -> None:
        monkeypatch.setattr("src.setup.__main__._DISK_PATH", task_directory)

        # Run directly so that monkeypatch is applied
        from src.setup.__main__ import main

        main()

        # Base directory asserts
        assert tmp_path.exists()
        assert task_directory.exists()

        monkeypatch.setattr("src.utils._DISK_PATH", task_directory)

        # Task directory asserts
        dataset_map = load_dataset_from_disk()

        assert len(dataset_map) == 500, "Expected 500 tasks to be available"
