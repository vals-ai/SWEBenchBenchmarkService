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

### Methodology changes

both `simple-web-scraper` and `simple-sheets-put` required a docker compose file with multiple services.

Daytona requires a single image and does not support compose. These services were compiled to support that.

### Testing

Unit tests

`make test-unit`

Integration tests

`make test-integration`
