FROM python:3.12-slim
# Pick up distro ca-certificates updates (e.g. the eMudhra intermediate CA distrust) before base images refresh.
RUN apt-get update && apt-get install -y --no-install-recommends --only-upgrade ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install docker CLI for image validation
RUN apt-get update && \
    apt-get install -y curl docker.io && \
    rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
COPY --from=ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Install git for fetching git dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN uv sync --locked

# Copy application code
COPY . .

# Download SWE-bench dataset during build
RUN uv run python -m swebench_service.dataset

# Expose port
EXPOSE 8001

# Run the application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--ws-ping-interval", "30", "--ws-ping-timeout", "600"]
