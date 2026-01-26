from pathlib import Path

import pytest
from pytest import MonkeyPatch, TempPathFactory


@pytest.fixture
def task_directory(tmp_path: Path) -> Path:
    return tmp_path / "swe-bench-verified"


@pytest.fixture(scope="function", autouse=True)
def setup_dataset(tmp_path_factory: TempPathFactory, monkeypatch: MonkeyPatch) -> Path:
    tmp_path = tmp_path_factory.mktemp("data")
    task_directory = tmp_path / "swe-bench-verified"

    monkeypatch.setattr("src.setup.__main__._DISK_PATH", task_directory)
    monkeypatch.setattr("src.utils._DISK_PATH", task_directory)

    from src.setup.__main__ import main

    main()

    return task_directory
