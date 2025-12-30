### Local Development

```bash
# Create .venv with python version 12
make install

# Activate virtual environment
source .venv/bin/activate

# Download tasks from github registry locally
make task-setup

# Start fastapi server for local testing
make start-fastapi
```

Documentation can be found at `http://127.0.0.1:8000/docs` once you start the fastapi server locally

also compiled into `make quick-start`

### Testing

Unit tests

`make test-unit`

Integration tests

`make test-integration`

### Required Environment variables

DAYTONA_API_KEY=...
DAYTONA_API_URL=...
DAYTONA_TARGET=...

### Endpoints

- `GET /health` - Health check endpoint
- `GET /verify-task-ids` - Verify task IDs exist in SWE-bench benchmark
- `GET /retrieve-tasks` - Get docker images and setup requirements for tasks
- `POST /setup-task` - Run setup script for a task in a sandbox
- `POST /evaluate-instance` - Execute tests and grade results for an instance
