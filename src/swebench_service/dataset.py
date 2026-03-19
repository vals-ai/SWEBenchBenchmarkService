"""Dataset loading and setup utilities for SWE-bench."""

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_from_disk  # type: ignore

DISK_PATH: Path = Path("/tmp/swe-bench-verified")
VALS_INDEX_PATH: Path = Path(__file__).parent / "vals_index.json"

_DATASET_CACHE: dict[str, dict[str, Any]] | None = None


def load_dataset_from_disk() -> dict[str, dict[str, Any]]:
    """
    Load the dataset from disk and return a mapping of instance_id to row data.

    Returns:
        dict[str, dict[str, Any]]: A dictionary mapping instance_id to the corresponding dataset row
    """
    global _DATASET_CACHE

    if _DATASET_CACHE is None:
        dataset = load_from_disk(DISK_PATH)
        _DATASET_CACHE = {row["instance_id"]: dict(row) for row in dataset}  # type: ignore

    return _DATASET_CACHE  # type: ignore[reportReturnType]


def load_vals_index_subset() -> dict[str, dict[str, Any]]:
    """Return the subset of the dataset filtered to only vals_index instance IDs."""
    full = load_dataset_from_disk()
    vals_index: list[str] = json.loads(VALS_INDEX_PATH.read_text())
    return {iid: full[iid] for iid in vals_index if iid in full}


def setup_dataset() -> None:
    """
    Download and save the SWE-bench_Verified dataset to disk.

    This should be run once during setup to cache the dataset locally.
    """
    print(f"Downloading SWE-bench_Verified dataset to {DISK_PATH}...")

    # Create parent directory if it doesn't exist
    DISK_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Download and save the dataset
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    dataset.save_to_disk(str(DISK_PATH))  # type: ignore[reportUnknownMemberType]

    print(f"Dataset saved to {DISK_PATH}")
    print(f"Total instances: {len(dataset)}")


if __name__ == "__main__":
    # Allow running as a module to download the dataset
    setup_dataset()
