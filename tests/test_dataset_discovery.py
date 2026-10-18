"""Dataset discovery preserves hosted tenant authorization without loading tasks."""

from typing import cast

import httpx
import pytest
from benchmark_service.vals import ValsBenchmarkService, ValsBenchmarkServiceApp, base
from benchmark_service.vals.allowlist import TenantConfig
from fastapi.testclient import TestClient

from swebench_service.benchmark_service import SWEBenchService


@pytest.fixture(autouse=True)
def setup_dataset():
    """Discovery uses names only, so the dataset download fixture is unnecessary."""


def test_dataset_discovery_requires_auth_and_filters_names(monkeypatch: pytest.MonkeyPatch) -> None:
    assert issubclass(SWEBenchService, ValsBenchmarkService)
    service = object.__new__(SWEBenchService)
    service.datasets = {"allowed": {}, "blocked": {}}
    service.dataset_versions = {}

    async def create(cls: type[SWEBenchService]) -> SWEBenchService:
        return service

    async def authenticate(headers: dict[str, str]) -> str | None:
        return "tenant" if headers.get("x-descope-api-key") == "test-key" else None

    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("SUBMISSION_ARTIFACT_BUCKET", "test-artifacts")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setattr(SWEBenchService, "create", classmethod(create))
    monkeypatch.setattr(base, "resolve_caller_tenant", authenticate)

    def tenant_config(tenant: str) -> TenantConfig:
        return TenantConfig(datasets=["allowed"])

    monkeypatch.setattr(base, "get_tenant_config", tenant_config)
    with TestClient(ValsBenchmarkServiceApp(SWEBenchService)) as raw_client:
        client = cast(httpx.Client, raw_client)
        assert client.get("/v1/datasets").status_code == 401
        response = client.get("/v1/datasets", headers={"x-descope-api-key": "test-key"})
        assert response.status_code == 200
        assert response.json() == {"datasets": ["allowed"]}
