### Local Development

```bash
# Create .venv with python version 12
make install

# Activate virtual environment
source .venv/bin/activate

# Download tasks from github registry locally
make setup

# Start local server
make benchmark-service-local
```

Documentation can be found at `http://127.0.0.1:8000/docs` once you start the local server

also compiled into `make quick-start`

### Testing

Unit tests

`make test-unit`

Integration tests

`make test-integration`

Experimental tests (flaky)

`uv run pytest tests/integration/test_daytona.py -m experimental`

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

### Known bugs

We use the same test script and evaluation methodology as the originally [SWEBench repository](https://github.com/SWE-bench/SWE-bench).

There have been a few tasks with reported bugs that basically make the task fail no matter what.

-- [astropy__astropy-7606](https://github.com/SWE-bench/SWE-bench/issues/223)
-- [astropy__astropy-8707](https://github.com/SWE-bench/SWE-bench/issues/342)
-- [astropy__astropy-8872](https://github.com/SWE-bench/SWE-bench/issues/343)
-- [django__django-10097](https://github.com/swe-bench/SWE-bench/issues/267)
