from pydantic import BaseModel


class TaskFilter(BaseModel):
    task_ids: list[str] | None = None
    slice_str: str | None = None

    def parse_slice(self) -> slice:
        if not self.slice_str:
            raise ValueError("Slice is not provided")

        parts = self.slice_str.split(":")

        if not 1 <= len(parts) <= 3:
            raise ValueError("Invalid slice format")

        def int_conversion(p: str) -> int | None:
            return int(p) if p else None

        while len(parts) < 3:
            parts.append("")

        start, stop, step = (int_conversion(p) for p in parts)
        return slice(start, stop, step)


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
    prediction: str | None = None
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


class Metadata(BaseModel):
    resolved_tasks: list[str]
    unresolved_tasks: list[str]


class FinalScoreResponse(BaseModel):
    tasks_evaluated: list[str]
    final_score: float
    metadata: Metadata


class StatusResponse(BaseModel):
    status: str


class SetupTaskResponse(StatusResponse):
    pass


class HealthCheckResponse(StatusResponse):
    pass


class RetrieveTaskResponse(BaseModel):
    docker_image: str
    problem_statement: str
    request_setup: bool
    cwd: str


class VerifyTaskIdsResponse(BaseModel):
    task_ids: list[str]
