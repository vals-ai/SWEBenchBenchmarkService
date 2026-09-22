"""Dataset loading and setup utilities for SWE-bench."""

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_from_disk  # type: ignore

# Every dataset is served from the SWE-bench organisation's HuggingFace copies, which carry
# the per-instance evaluation script, log parser name, eval type, and image that the
# harness (swebench >= 5) grades from. Each is pinned to a revision: upstream reshapes
# splits in place, and a benchmark's task set must not move under its published scores.

# SWE-bench_Verified: the same 500 instances as princeton-nlp/SWE-bench_Verified, with the
# grading columns added. Two instances (astropy__astropy-7606, django__django-10097) lost a
# handful of PASS_TO_PASS tests upstream between the two copies.
VERIFIED_DATASET_NAME = "SWE-bench/SWE-bench_Verified"
VERIFIED_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"
DISK_PATH: Path = Path("/tmp/swe-bench-verified")
VALS_INDEX_PATH: Path = Path(__file__).parent / "vals_index.json"

# SWE-bench Multimodal: issue/PR pairs from JavaScript repositories whose problem statements
# carry screenshots. `multimodal` is the test split (480 instances, 11 repositories), the
# split the public leaderboard reports. `multimodal_dev` is the dev split (100 instances,
# 5 repositories), kept for smokes and parity checks.
MULTIMODAL_DATASET_NAME = "SWE-bench/SWE-bench_Multimodal"
MULTIMODAL_REVISION = "4e6662d51c48e475f7f346e4fa09a6f8b31fcaa5"
MULTIMODAL_DISK_PATH: Path = Path("/tmp/swe-bench-multimodal")
MULTIMODAL_DEV_DISK_PATH: Path = Path("/tmp/swe-bench-multimodal-dev")

# One in-memory copy per dataset, reloaded when its disk path changes (the tests point every
# run at a fresh directory), so the cache never holds more than the datasets being served.
_DATASET_CACHE: dict[str, tuple[Path, dict[str, dict[str, Any]]]] = {}


def _load_from_disk(name: str, disk_path: Path) -> dict[str, dict[str, Any]]:
    """Load a saved dataset from disk and return a mapping of instance_id to row data."""
    cached = _DATASET_CACHE.get(name)
    if cached is None or cached[0] != disk_path:
        dataset = load_from_disk(disk_path)
        rows: dict[str, dict[str, Any]] = {row["instance_id"]: dict(row) for row in dataset}  # type: ignore
        cached = (disk_path, rows)
        _DATASET_CACHE[name] = cached

    return cached[1]


def load_dataset_from_disk() -> dict[str, dict[str, Any]]:
    """
    Load the SWE-bench_Verified dataset from disk and return a mapping of instance_id to row data.

    Returns:
        dict[str, dict[str, Any]]: A dictionary mapping instance_id to the corresponding dataset row
    """
    return _load_from_disk("verified", DISK_PATH)


def load_vals_index_subset() -> dict[str, dict[str, Any]]:
    """Return the subset of the dataset filtered to only vals_index instance IDs."""
    full = load_dataset_from_disk()
    vals_index: list[str] = json.loads(VALS_INDEX_PATH.read_text())
    return {iid: full[iid] for iid in vals_index if iid in full}


def load_multimodal_dataset_from_disk() -> dict[str, dict[str, Any]]:
    """Load the SWE-bench Multimodal test split from disk and return a mapping of instance_id to row data."""
    return _load_from_disk("multimodal", MULTIMODAL_DISK_PATH)


def load_multimodal_dev_dataset_from_disk() -> dict[str, dict[str, Any]]:
    """Load the SWE-bench Multimodal dev split from disk and return a mapping of instance_id to row data."""
    return _load_from_disk("multimodal_dev", MULTIMODAL_DEV_DISK_PATH)


def _download(name: str, *, split: str, disk_path: Path, revision: str) -> None:
    print(f"Downloading {name} ({split} @ {revision[:8]}) to {disk_path}...")
    disk_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(name, split=split, revision=revision)
    dataset.save_to_disk(str(disk_path))  # type: ignore[reportUnknownMemberType]

    print(f"Dataset saved to {disk_path}")
    print(f"Total instances: {len(dataset)}")


def setup_dataset() -> None:
    """
    Download and save every dataset the service serves.

    This should be run once during setup to cache the datasets locally.
    """
    _download(VERIFIED_DATASET_NAME, split="test", disk_path=DISK_PATH, revision=VERIFIED_REVISION)
    _download(MULTIMODAL_DATASET_NAME, split="test", disk_path=MULTIMODAL_DISK_PATH, revision=MULTIMODAL_REVISION)
    _download(MULTIMODAL_DATASET_NAME, split="dev", disk_path=MULTIMODAL_DEV_DISK_PATH, revision=MULTIMODAL_REVISION)


if __name__ == "__main__":
    # Allow running as a module to download the datasets
    setup_dataset()
