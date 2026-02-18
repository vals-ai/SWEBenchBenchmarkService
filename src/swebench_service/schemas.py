"""SWE-bench specific data models."""

from pydantic import BaseModel


class EvaluationResult(BaseModel):
    """Result from evaluating a SWE-bench task."""

    prediction: str | None = None
    patch_successfully_applied: bool
    resolved: bool
    resolution_status: str
    fail_to_pass: dict[str, list[str]] | None = None
    pass_to_pass: dict[str, list[str]] | None = None
    f2p_score: float | None = None
    p2p_score: float | None = None
    status_map: dict[str, str] | None = None
