from typing import Any, Callable

from daytona import AsyncSandbox, DaytonaError, DaytonaNotFoundError
from tenacity import retry, retry_if_exception_type, retry_if_not_exception_type, stop_after_attempt, wait_exponential


async def with_retry(sandbox: AsyncSandbox, fn: Callable[..., Any]) -> Any:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=(retry_if_exception_type(DaytonaError) & retry_if_not_exception_type(DaytonaNotFoundError)),
        reraise=True,
    )
    async def _attempt() -> Any:
        return await fn()

    try:
        return await _attempt()
    except DaytonaError as e:
        await sandbox.refresh_data()
        raise DaytonaError(f"{e} | sandbox={sandbox.name} state={sandbox.state}") from e
