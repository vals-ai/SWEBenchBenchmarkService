FROM python:3.12-slim

WORKDIR /app

# Install docker CLI for image validation
RUN apt-get update && \
    apt-get install -y curl docker.io && \
    rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml README.md ./

# Install git for fetching git dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN uv sync

# Copy application code
COPY . .

# Download SWE-bench dataset during build
RUN uv run python -m swebench_service.dataset

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "30", "--ws-ping-timeout", "10"]
