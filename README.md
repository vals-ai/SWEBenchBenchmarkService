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

Related documentation can be found [here](https://www.daytona.io/docs/configuration#env-file)

```
DAYTONA_API_KEY=...
DAYTONA_API_URL=...
DAYTONA_TARGET=...
```

### Endpoints

- `GET /health` - Health check endpoint
- `GET /verify-task-ids` - Verify task IDs exist in SWE-bench benchmark
- `GET /retrieve-task/` - Get docker image and setup requirements for a single task
- `POST /setup-task` - Run setup script for a task in a sandbox
- `POST /evaluate-instance` - Execute tests and grade results for an instance
- `POST /final-score` - Takes the evaluation results and produces a json containing the final score and evaluation metadata

### Upload to AWS

First time install and publish

- `npm install -g aws-cdk`
- `cdk bootstrap aws://<AWS_ACCOUNT_ID>/<AWS_REGION>`
- `make deploy-ecs`

Redeploy

- `force-deploy-ecs`

Average deploy time is 385.79s
