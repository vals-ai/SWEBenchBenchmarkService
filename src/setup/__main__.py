from pathlib import Path

from datasets import load_dataset, load_from_disk  # type: ignore

from src.logger import get_logger

logger = get_logger(__name__)

_DISK_PATH: Path = Path("/tmp/swe-bench-verified")


def main() -> None:
    logger.info("Loading SWE-bench_Verified dataset")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    logger.info(f"Loaded {len(dataset)} rows")
    dataset.save_to_disk(_DISK_PATH)  # type: ignore

    logger.info(f"Saved dataset to {_DISK_PATH}")

    logger.info(f"Checking dataset exists at {_DISK_PATH}")
    dataset = load_from_disk(_DISK_PATH)
    if not dataset or len(dataset) != 500:
        raise ValueError(f"Expected 500 rows, got {len(dataset)} rows")

    logger.info("Dataset saved successfully")


if __name__ == "__main__":
    main()
