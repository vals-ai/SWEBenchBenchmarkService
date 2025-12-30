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
