from collections.abc import Awaitable, Callable
from typing import Any

from benchmark_service.sandbox import Sandbox, SandboxError, SandboxNotFoundError
from tenacity import retry, retry_if_exception_type, retry_if_not_exception_type, stop_after_attempt, wait_exponential


async def with_retry(sandbox: Sandbox, fn: Callable[[], Awaitable[Any]]) -> Any:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=(retry_if_exception_type(SandboxError) & retry_if_not_exception_type(SandboxNotFoundError)),
        reraise=True,
    )
    async def _attempt() -> Any:
        return await fn()

    try:
        return await _attempt()
    except SandboxError as e:
        raise SandboxError(f"{e} | sandbox={sandbox.name} state={sandbox.state}") from e
