from pydantic import BaseModel


class TaskFilter(BaseModel):
    task_ids: list[str] | None = None


class EvaluateInstanceRequest(BaseModel):
    task_id: str
    instance_id: str


class EvaluateResponseRequest(BaseModel):
    task_id: str
    response: str


class SetupTaskRequest(BaseModel):
    task_id: str
    instance_id: str


class EvaluationResult(BaseModel):
    task_id: str
    instance_id: str
    patch_successfully_applied: bool
    resolved: bool
    resolution_status: str
    fail_to_pass: dict[str, list[str]] | None = None
    pass_to_pass: dict[str, list[str]] | None = None
    f2p_score: float | None = None
    p2p_score: float | None = None
    status_map: dict[str, str] | None = None


class FinalScoreRequest(BaseModel):
    evaluation_results: dict[str, EvaluationResult | None]
