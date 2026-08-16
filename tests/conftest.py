from pathlib import Path

import pytest
from pytest import Config, MonkeyPatch, TempPathFactory


@pytest.fixture
def task_directory(tmp_path: Path) -> Path:
    return tmp_path / "swe-bench-verified"


@pytest.fixture(scope="function", autouse=True)
def setup_dataset(tmp_path_factory: TempPathFactory, monkeypatch: MonkeyPatch) -> Path:
    tmp_path = tmp_path_factory.mktemp("data")
    task_directory = tmp_path / "swe-bench-verified"
    monkeypatch.setenv("SWEBENCH_EVAL_STATE_LOCAL_DIR", str(tmp_path / "eval-resume"))

    # Update monkeypatch paths for new structure
    monkeypatch.setattr("swebench_service.dataset.DISK_PATH", task_directory)
    monkeypatch.setattr("swebench_service.benchmark_service.DISK_PATH", task_directory)

    from swebench_service.dataset import setup_dataset

    setup_dataset()

    return task_directory


def pytest_configure(config: Config) -> None:
    """Configure pytest with asyncio mode."""
    config.option.asyncio_mode = "auto"
