"""Dataset loading and setup utilities for SWE-bench."""

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_from_disk  # type: ignore

DISK_PATH: Path = Path("/tmp/swe-bench-verified")
VALS_INDEX_PATH: Path = Path(__file__).parent / "vals_index.json"

# SWE-bench Multimodal, dev split: issue/PR pairs from five JavaScript repositories whose
# problem statements carry screenshots. Only the dev split is served. The test split is
# graded exclusively by the hosted SWE-bench evaluation service: the swebench harness has no
# specs or log parsers for its twelve repositories and no evaluation images are published.
# The revision is pinned because the upstream split has been reshaped in place before, and
# a benchmark's task set must not change underneath its published scores.
MULTIMODAL_DISK_PATH: Path = Path("/tmp/swe-bench-multimodal")
MULTIMODAL_DATASET_NAME = "SWE-bench/SWE-bench_Multimodal"
MULTIMODAL_SPLIT = "dev"
MULTIMODAL_REVISION = "4e6662d51c48e475f7f346e4fa09a6f8b31fcaa5"

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
    """Load the SWE-bench Multimodal dev split from disk and return a mapping of instance_id to row data."""
    return _load_from_disk("multimodal", MULTIMODAL_DISK_PATH)


def _download(name: str, *, split: str, disk_path: Path, revision: str | None = None) -> None:
    print(f"Downloading {name} ({split}) to {disk_path}...")
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
    _download("princeton-nlp/SWE-bench_Verified", split="test", disk_path=DISK_PATH)
    _download(
        MULTIMODAL_DATASET_NAME,
        split=MULTIMODAL_SPLIT,
        disk_path=MULTIMODAL_DISK_PATH,
        revision=MULTIMODAL_REVISION,
    )


if __name__ == "__main__":
    # Allow running as a module to download the datasets
    setup_dataset()
