"""Tests for the catalog-backed tenant allowlist client.

Proves the service authorizes tenants through the benchmark catalog API
(`BENCHMARK_CATALOG_API_URL` + `SERVICE_NAME`) instead of the injected
local allowlist: request shape, fail-closed on unknown tenants, and that a
legacy allowlist is ignored once catalog mode is configured.
"""

from __future__ import annotations

import json
from collections.abc import Generator, Mapping
from typing import Any

import httpx
import pytest

import benchmark_service.auth as auth_module
from benchmark_service.allowlist import CatalogAllowlistClient
from tests.utils import BenchmarkServiceTestClient

CATALOG_URL = "https://catalog.test"
SERVICE_NAME = "swebench"
DESCOPE_PROJECT_ID = "test-project"
ALLOWED_KEY = "allowed-access-key"
ALLOWED_TENANT = "allowed-tenant"
LEGACY_ONLY_KEY = "legacy-only-access-key"
UNKNOWN_TENANT = "unknown-tenant"


def _catalog_response_handler(requests_log: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests_log.append(request)
        if request.headers.get("x-descope-api-key") == ALLOWED_KEY:
            return httpx.Response(
                200,
                json={
                    "name": SERVICE_NAME,
                    "datasets": ["default", "vals_index"],
                    "evaluation_quota": None,
                    "trial_mode": False,
                },
            )
        return httpx.Response(404, json={"detail": "tenant not found"})

    return handler


@pytest.fixture
def requests_log() -> list[httpx.Request]:
    return []


@pytest.fixture(autouse=True)
def catalog_env(monkeypatch: pytest.MonkeyPatch, requests_log: list[httpx.Request]) -> Generator[None, None, None]:
    monkeypatch.setenv("BENCHMARK_CATALOG_API_URL", CATALOG_URL)
    monkeypatch.setenv("SERVICE_NAME", SERVICE_NAME)
    monkeypatch.setenv("DESCOPE_PROJECT_ID", DESCOPE_PROJECT_ID)
    monkeypatch.delenv("AUTH_DISABLED")
    auth_module.clear_allowlist_cache()
    auth_module.clear_auth_cache()

    async def fake_exchange(project_id: str, access_key: str) -> Mapping[str, Any]:
        tenant = (
            ALLOWED_TENANT
            if access_key in {ALLOWED_KEY, LEGACY_ONLY_KEY}
            else UNKNOWN_TENANT
        )
        return {"tenants": {tenant: {}}}

    monkeypatch.setattr(auth_module, "_exchange_descope_access_key", fake_exchange)
    catalog_client = CatalogAllowlistClient(
        CATALOG_URL,
        SERVICE_NAME,
        transport=httpx.MockTransport(_catalog_response_handler(requests_log)),
    )
    monkeypatch.setattr(auth_module, "_catalog_client", catalog_client)
    yield


@pytest.fixture
def client() -> Generator[BenchmarkServiceTestClient]:
    c = BenchmarkServiceTestClient()
    yield c
    c.close()


async def test_catalog_mode_authorizes_allowed_tenant(
    client: BenchmarkServiceTestClient,
    requests_log: list[httpx.Request],
) -> None:
    """An access key that the catalog allows reaches the protected endpoint."""
    response = await client.request_verify_task_ids(
        ["invalid-task-id"],
        headers={"x-descope-api-key": ALLOWED_KEY},
    )
    assert response.status_code == 400

    assert requests_log, "expected at least one catalog request"
    request = requests_log[-1]
    assert request.url.path == f"/benchmark-services/{SERVICE_NAME}"
    assert request.headers["x-descope-api-key"] == ALLOWED_KEY


async def test_catalog_mode_fails_closed_for_unknown_tenant(client: BenchmarkServiceTestClient) -> None:
    """An access key the catalog does not recognize is rejected."""
    response = await client.request_verify_task_ids(
        ["invalid-task-id"],
        headers={"x-descope-api-key": "unknown-access-key"},
    )
    assert response.status_code == 401


async def test_catalog_mode_denies_missing_access_key(client: BenchmarkServiceTestClient) -> None:
    """A request without any access key is rejected in catalog mode."""
    response = await client.request_verify_task_ids(["invalid-task-id"])
    assert response.status_code == 401


async def test_catalog_mode_ignores_legacy_allowlist(
    client: BenchmarkServiceTestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tenant listed only in the legacy allowlist is denied in catalog mode."""
    legacy_allowlist = json.dumps({"tenants": {ALLOWED_TENANT: {"datasets": ["default", "vals_index"]}}})
    monkeypatch.setenv("DESCOPE_TENANT_ALLOWLIST_JSON", legacy_allowlist)
    assert auth_module.load_allowlist().tenants == {}

    response = await client.request_verify_task_ids(
        ["invalid-task-id"],
        headers={"x-descope-api-key": LEGACY_ONLY_KEY},
    )
    assert response.status_code == 401


async def test_catalog_client_returns_none_on_network_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catalog failures must not allow a tenant through."""
    async def broken_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    client_under_test = CatalogAllowlistClient(
        CATALOG_URL,
        SERVICE_NAME,
        transport=httpx.MockTransport(broken_handler),
    )
    config = await client_under_test.get_tenant_config(ALLOWED_KEY, ALLOWED_TENANT)
    assert config is None
