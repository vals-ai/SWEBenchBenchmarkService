FROM docker:28.3.3-dind

RUN apk add --no-cache \
    python3 \
    py3-pip \
    make \
    curl \
    coreutils \
    bash

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_SYSTEM_PYTHON=1

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY . .

RUN rm -rf .venv __pycache__ .pytest_cache && uv run python -m src.setup

EXPOSE 8000

CMD ["sleep", "infinity"]