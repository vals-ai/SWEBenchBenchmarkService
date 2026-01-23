FROM --platform=linux/amd64 docker:28.3.3-dind

RUN apk add --no-cache \
    python3 \
    py3-pip \
    make \
    curl \
    coreutils \
    bash

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY ./src ./src
COPY setup.sh pyproject.toml uv.lock README.md main.py ./

RUN uv sync --frozen

RUN PYTHONPATH=/app /app/.venv/bin/python src/setup/__main__.py

EXPOSE 8000

CMD ["uv", "run", "fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
