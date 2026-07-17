"""Durable SWE-bench generation artifacts for eval-only retry."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, TypedDict, cast
from uuid import UUID

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from benchmark_service.sandbox import Sandbox

_ARTIFACT_PREFIX = "swebench/eval-resume"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class _StreamingBody(Protocol):
    def read(self) -> bytes: ...


class _GetObjectResponse(TypedDict):
    Body: _StreamingBody


class _S3Client(Protocol):
    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
    ) -> object: ...

    def get_object(self, *, Bucket: str, Key: str) -> _GetObjectResponse: ...


class EvalResumeState(BaseModel):
    """Small, validated pointer to one immutable generated patch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    benchmark_id: UUID
    task_id: str
    dataset: str
    prediction_s3_key: str
    prediction_sha256: str
    prediction_size_bytes: int = Field(ge=0)

    @field_validator("task_id", "dataset")
    @classmethod
    def validate_component(cls, value: str) -> str:
        if not _SAFE_COMPONENT.fullmatch(value):
            raise ValueError("resume-state identifiers may contain only letters, numbers, '.', '_', and '-'")
        return value

    @field_validator("prediction_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("prediction_sha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_prediction_key(self) -> "EvalResumeState":
        if self.prediction_s3_key != prediction_key(
            self.benchmark_id,
            self.task_id,
            self.prediction_sha256,
        ):
            raise ValueError("prediction_s3_key does not match the canonical SWE-bench artifact path")
        return self


def prediction_key(benchmark_id: UUID, task_id: str, prediction_sha256: str) -> str:
    """Return the only S3 key these validated state fields may address."""
    return f"{_ARTIFACT_PREFIX}/{benchmark_id}/{task_id}/{prediction_sha256}.patch"


async def persist_prediction(
    sandbox: Sandbox,
    task_id: str,
    dataset: str | None,
    prediction: bytes,
) -> EvalResumeState:
    """Persist the exact generated patch before evaluation begins."""
    labels = _sandbox_labels(sandbox)
    benchmark_id = labels.get("Id", "").strip()
    if not benchmark_id:
        raise RuntimeError(f"SWE-bench sandbox {sandbox.id} is missing Valkyrie run label 'Id'")
    benchmark_uuid = UUID(benchmark_id)
    prediction_sha256 = hashlib.sha256(prediction).hexdigest()
    state = EvalResumeState(
        benchmark_id=benchmark_uuid,
        task_id=task_id,
        dataset=dataset or "default",
        prediction_s3_key=prediction_key(benchmark_uuid, task_id, prediction_sha256),
        prediction_sha256=prediction_sha256,
        prediction_size_bytes=len(prediction),
    )
    await _put_object(state.prediction_s3_key, prediction)
    return state


async def load_prediction(state: EvalResumeState) -> bytes:
    """Fetch and integrity-check the generated patch referenced by state."""
    content = await _get_object(state.prediction_s3_key)
    if len(content) != state.prediction_size_bytes:
        raise ValueError("Persisted SWE-bench prediction failed its byte-length integrity check")
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != state.prediction_sha256:
        raise ValueError("Persisted SWE-bench prediction failed its SHA-256 integrity check")
    return content


def _sandbox_labels(sandbox: Sandbox) -> dict[str, str]:
    inner = getattr(sandbox, "_sandbox", None)
    labels = getattr(inner, "labels", None)
    if not isinstance(labels, dict):
        return {}
    raw_labels = cast(dict[object, object], labels)
    return {key: value for key, value in raw_labels.items() if isinstance(key, str) and isinstance(value, str)}


def _local_root() -> Path | None:
    value = os.environ.get("SWEBENCH_EVAL_STATE_LOCAL_DIR")
    return Path(value).expanduser() if value else None


def _local_path(key: str) -> Path:
    key_path = PurePosixPath(key)
    if key_path.is_absolute() or ".." in key_path.parts:
        raise ValueError("Invalid SWE-bench eval-resume artifact key")
    root = _local_root()
    if root is None:
        raise RuntimeError("SWEBENCH_EVAL_STATE_LOCAL_DIR is not configured")
    return root.joinpath(*key_path.parts)


def _bucket() -> str:
    value = os.environ.get("SWEBENCH_EVAL_STATE_BUCKET")
    if not value:
        raise RuntimeError("SWEBENCH_EVAL_STATE_BUCKET is not configured")
    return value


def _s3_client() -> _S3Client:
    return cast(
        _S3Client,
        boto3.client(  # pyright: ignore[reportUnknownMemberType]
            "s3",
            region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
        ),
    )


async def _put_object(key: str, content: bytes) -> None:
    if _local_root() is not None:
        path = _local_path(key)

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

        await asyncio.to_thread(write)
        return

    try:
        await asyncio.to_thread(
            _s3_client().put_object,
            Bucket=_bucket(),
            Key=key,
            Body=content,
            ContentType="text/x-diff",
        )
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to persist SWE-bench prediction at {key}") from exc


async def _get_object(key: str) -> bytes:
    if _local_root() is not None:
        return await asyncio.to_thread(_local_path(key).read_bytes)

    try:
        response = await asyncio.to_thread(_s3_client().get_object, Bucket=_bucket(), Key=key)
        return await asyncio.to_thread(response["Body"].read)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to load SWE-bench prediction at {key}") from exc
