"""The final score and the vals-format metadata both lambdas read from it."""

from typing import Any

import pytest

from swebench_service import (
    load_multimodal_dataset_from_disk,
    load_vals_index_subset,
)
from swebench_service.benchmark_service import SWEBenchService


def resolved() -> dict[str, bool]:
    return {"resolved": True}


def unresolved() -> dict[str, bool]:
    return {"resolved": False}


@pytest.fixture
def service() -> SWEBenchService:
    """A bare instance: scoring is pure, and the contract adapter calls it this way."""
    return SWEBenchService()


@pytest.fixture
def index_ids() -> list[str]:
    return list(load_vals_index_subset())


@pytest.fixture
def non_index_id(index_ids: list[str]) -> str:
    from swebench_service import load_dataset_from_disk

    return next(task_id for task_id in load_dataset_from_disk() if task_id not in set(index_ids))


@pytest.fixture
def multimodal_ids() -> list[str]:
    return list(load_multimodal_dataset_from_disk())


async def score(service: SWEBenchService, results: dict[str, Any], dataset: str | None = None) -> Any:
    return await service.calculate_final_score(results, dataset=dataset)


# --- the score itself ------------------------------------------------------------------------


async def test_the_score_is_the_resolved_percentage(service: SWEBenchService, multimodal_ids: list[str]) -> None:
    a, b, c, d = multimodal_ids[:4]
    result = await score(
        service,
        {a: resolved(), b: unresolved(), c: resolved(), d: None},
        dataset="multimodal",
    )
    assert result.score == 50.0


async def test_an_empty_run_scores_zero(service: SWEBenchService) -> None:
    assert (await score(service, {}, dataset="multimodal")).score == 0.0


async def test_the_legacy_view_still_gets_its_task_lists(
    service: SWEBenchService, multimodal_ids: list[str]
) -> None:
    """SWE-bench Verified still runs under swebench-final-view-lambda, which reads these."""
    a, b = multimodal_ids[:2]
    metadata = (await score(service, {a: resolved(), b: unresolved()}, dataset="multimodal")).metadata
    assert metadata["resolved_tasks"] == [a]
    assert metadata["unresolved_tasks"] == [b]


async def test_a_task_outside_the_dataset_is_refused(service: SWEBenchService) -> None:
    with pytest.raises(ValueError, match="Task ID not found"):
        await score(service, {"not-a-real-swe-task": resolved()}, dataset="multimodal")


# --- vals-format ----------------------------------------------------------------------------


async def test_the_metadata_declares_what_the_lambda_needs(
    service: SWEBenchService, multimodal_ids: list[str]
) -> None:
    metadata = (await score(service, {multimodal_ids[0]: resolved()}, dataset="multimodal")).metadata
    assert set(metadata) >= {
        "score_types",
        "results",
        "primary_population",
        "tasks",
        "usage_components",
    }
    assert metadata["score_types"]["score"]["unit"] == "percent"
    assert metadata["usage_components"] == [{"component": "generation.model"}]


async def test_multimodal_scores_one_population(
    service: SWEBenchService, multimodal_ids: list[str]
) -> None:
    """The Vals Index subset is a SWE-bench Verified selection; multimodal has none."""
    a, b = multimodal_ids[:2]
    metadata = (await score(service, {a: resolved(), b: unresolved()}, dataset="multimodal")).metadata

    assert metadata["primary_population"] == "full"
    assert list(metadata["results"]) == ["full"]
    full = metadata["results"]["full"]
    assert full["counts"] == {"total": 2, "by_status": {"resolved": 1, "unresolved": 1}, "extra": {}}
    assert full["scores"]["score"]["value"] == 50.0
    assert full["selection"] is None


async def test_every_submitted_task_gets_a_row_in_order(
    service: SWEBenchService, multimodal_ids: list[str]
) -> None:
    a, b, c = multimodal_ids[:3]
    metadata = (await score(service, {a: resolved(), b: None, c: unresolved()}, dataset="multimodal")).metadata

    assert [task["task_id"] for task in metadata["tasks"]] == [a, b, c]
    assert [task["status"] for task in metadata["tasks"]] == ["resolved", "unresolved", "unresolved"]
    assert metadata["tasks"][0]["scores"]["score"]["value"] == 100.0
    assert metadata["tasks"][1]["scores"]["score"]["value"] == 0.0


async def test_task_rows_carry_no_field_the_schema_rejects(
    service: SWEBenchService, multimodal_ids: list[str]
) -> None:
    """vals_format.v1 forbids unknown keys and drops the whole run when one appears."""
    allowed = {
        "task_id", "category", "tags", "status", "output", "retries",
        "scores", "aggregated_metrics", "evaluations", "error", "turns", "extra",
    }
    metadata = (await score(service, {multimodal_ids[0]: resolved()}, dataset="multimodal")).metadata
    assert set(metadata["tasks"][0]) <= allowed


async def test_population_counts_sum_to_its_total(
    service: SWEBenchService, multimodal_ids: list[str]
) -> None:
    submitted = {task_id: resolved() for task_id in multimodal_ids[:5]}
    submitted[multimodal_ids[5]] = unresolved()
    metadata = (await score(service, submitted, dataset="multimodal")).metadata

    for population in metadata["results"].values():
        assert population["counts"]["total"] == sum(population["counts"]["by_status"].values())


# --- the Vals Index subset --------------------------------------------------------------------


async def test_a_mixed_default_run_scores_both_populations(
    service: SWEBenchService, index_ids: list[str], non_index_id: str
) -> None:
    metadata = (await score(service, {index_ids[0]: resolved(), non_index_id: unresolved()})).metadata

    assert metadata["primary_population"] == "full"
    assert sorted(metadata["results"]) == ["full", "vals_index"]
    assert metadata["results"]["vals_index"]["selection"]["criteria"]["task_ids"] == [index_ids[0]]
    assert metadata["results"]["vals_index"]["scores"]["score"]["value"] == 100.0
    assert metadata["results"]["full"]["scores"]["score"]["value"] == 50.0


async def test_a_run_with_no_index_task_has_no_index_population(
    service: SWEBenchService, non_index_id: str
) -> None:
    metadata = (await score(service, {non_index_id: resolved()})).metadata
    assert list(metadata["results"]) == ["full"]


async def test_submitting_exactly_the_subset_is_an_index_run(
    service: SWEBenchService, index_ids: list[str]
) -> None:
    """However it was requested, a run of exactly that subset is ranked as the subset."""
    metadata = (await score(service, {task_id: resolved() for task_id in index_ids})).metadata

    assert metadata["primary_population"] == "vals_index"
    assert list(metadata["results"]) == ["vals_index"]


async def test_the_index_dataset_scores_only_the_index(
    service: SWEBenchService, index_ids: list[str]
) -> None:
    a, b = index_ids[:2]
    metadata = (await score(service, {a: resolved(), b: None}, dataset="vals_index")).metadata

    assert metadata["primary_population"] == "vals_index"
    assert list(metadata["results"]) == ["vals_index"]
    assert metadata["results"]["vals_index"]["selection"]["criteria"]["task_ids"] == [a, b]


async def test_an_empty_index_run_still_names_the_index(service: SWEBenchService) -> None:
    metadata = (await score(service, {}, dataset="vals_index")).metadata
    assert metadata["primary_population"] == "vals_index"
    assert list(metadata["results"]) == ["vals_index"]
    assert metadata["tasks"] == []


async def test_an_index_run_refuses_a_task_outside_the_subset(
    service: SWEBenchService, index_ids: list[str], non_index_id: str
) -> None:
    """Scoring it would report the subset's number over a different set of tasks."""
    with pytest.raises(ValueError, match="Task ID not found"):
        await score(service, {index_ids[0]: resolved(), non_index_id: resolved()}, dataset="vals_index")
