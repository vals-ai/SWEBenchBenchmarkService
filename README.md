# SWE-bench Benchmark Service

A FastAPI-based service for evaluating software engineering agents on the SWE-bench benchmark. This service provides standardized endpoints for task retrieval, environment setup, and test-based evaluation of code changes.

## Overview

SWE-bench is a benchmark for evaluating AI agents on real-world software engineering tasks. This service implements the **SWE-bench_Verified** dataset, which contains 500 carefully verified GitHub issues from popular Python repositories.

Each task requires:
1. Understanding a GitHub issue description
2. Making appropriate code changes to fix the issue
3. Passing the associated test suite

**Paper:** [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)
**Repository:** [princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench)

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (for containerized deployment)

### Installation

1. **Install dependencies:**
   ```bash
   make install
   ```

2. **Download the SWE-bench_Verified dataset** (required before running the service):
   ```bash
   make setup
   ```

   This downloads ~500 task instances to `/tmp/swe-bench-verified/` (approximately 100MB).

3. **Run the development server:**
   ```bash
   make dev
   ```

The service will be available at `http://localhost:8000`. View API documentation at `http://localhost:8000/docs`.

### Docker Deployment

Build and run using Docker:

```bash
make docker-build
make docker-run
```

The dataset is downloaded during the Docker build process, so no separate setup is needed.

## Dataset

- **Name:** SWE-bench_Verified
- **Source:** `princeton-nlp/SWE-bench_Verified` (HuggingFace)
- **Size:** 500 verified instances
- **Cache Location:** `/tmp/swe-bench-verified/`
- **Repositories:** Django, Flask, Matplotlib, Pandas, Pytest, Requests, Scikit-learn, Sphinx, SymPy, and more

Each instance contains:
- `instance_id`: Unique task identifier (e.g., `django__django-12453`)
- `problem_statement`: GitHub issue description
- `base_commit`: Git commit hash to start from
- `test_patch`: Gold test patch
- `patch`: Gold solution patch (for reference)
- `repo`, `version`: Repository and version information

## Docker Images

Each task uses a pre-built Docker image with the repository and dependencies:

```
ghcr.io/epoch-research/swe-bench.eval.x86_64.{task_id}:latest
```

These images are maintained by [Epoch Research](https://github.com/epoch-research/swe-bench-docker) and contain:
- Repository cloned at the base commit
- All dependencies installed
- Testing framework configured

## Resource Requirements

**Default allocation per task:**
- vCPU: 2 cores
- Memory: 4 GB
- Disk: 10 GB

**Large tasks** (require more resources):
- `scikit-learn__scikit-learn-14710`
- `psf__requests-2317`

These receive 4 vCPU and 8 GB memory.

## API Endpoints

### Health Check
```http
GET /health
```

Returns service status.

### Verify Task IDs
```http
POST /verify-task-ids
Content-Type: application/json

{
  "task_ids": ["django__django-12453", "flask__flask-4045"]
}
```

Validates that task IDs exist in the dataset.

### Retrieve Task
```http
GET /retrieve-task/{task_id}
```

Returns task metadata including:
- Docker image name
- Problem statement
- Resource requirements
- Working directory

### Setup Task (WebSocket)
```http
WS /ws/setup-task
```

Connects to a sandbox and:
1. Uploads the setup script
2. Resets the repository to the base commit
3. Applies repository-specific pre-install commands
4. Streams progress logs

**Message format:**
```json
{"type": "message", "data": "log line..."}
{"type": "result", "data": {"status": "ok"}}
```

### Evaluate Instance (WebSocket)
```http
WS /ws/evaluate-instance
```

Evaluates a solution in the sandbox:
1. Captures the agent's changes (git diff)
2. Generates and uploads evaluation script
3. Runs the test suite
4. Grades test results using SWE-bench's grading logic
5. Streams test output and final results

**Message format:**
```json
{"type": "message", "data": "test output..."}
{"type": "result", "data": {
  "resolved": true,
  "patch_successfully_applied": true,
  "resolution_status": "FULL",
  "fail_to_pass": {"success": [...], "failure": [...]},
  "pass_to_pass": {"success": [...], "failure": [...]},
  "f2p_score": 1.0,
  "p2p_score": 1.0,
  "prediction": "diff --git ..."
}}
```

### Evaluate Response
```http
POST /evaluate-response
```

**Not supported** - SWE-bench requires sandbox evaluation. Use `/ws/evaluate-instance` instead.

### Calculate Final Score
```http
POST /final-score
Content-Type: application/json

{
  "evaluation_results": {
    "task_id_1": {"resolved": true, ...},
    "task_id_2": {"resolved": false, ...}
  }
}
```

Returns aggregate score as percentage of resolved tasks:

```json
{
  "tasks_evaluated": ["task_id_1", "task_id_2"],
  "final_score": 50.0,
  "metadata": {
    "resolved_tasks": ["task_id_1"],
    "unresolved_tasks": ["task_id_2"]
  }
}
```

## Evaluation Process

The evaluation follows SWE-bench's official grading methodology:

1. **Capture Prediction:** Extract agent's changes via `git diff`
2. **Run Tests:** Execute the test suite using the task's evaluation script
3. **Parse Output:** Extract test results from output using repository-specific parsers
4. **Grade Results:** Compare against gold test specifications:
   - **Fail-to-Pass (F2P):** Tests that should transition from failing to passing
   - **Pass-to-Pass (P2P):** Tests that should remain passing
5. **Determine Resolution:**
   - `FULL`: All F2P tests pass, all P2P tests pass
   - `PARTIAL`: Some tests pass
   - `NO`: No tests pass or errors occurred

## Development

### Available Commands

```bash
make help          # Show available commands
make install       # Install dependencies
make setup         # Download SWE-bench dataset
make dev           # Start development server
make lint          # Check code style
make format        # Format code
make typecheck     # Type check with basedpyright
make test          # Run tests
make docker-build  # Build Docker image
make docker-run    # Run Docker container
```

### Project Structure

```
.
├── main.py                 # SWEBenchService implementation
├── swebench_utils/         # Utility modules
│   ├── __init__.py
│   ├── schemas.py          # EvaluationResult model
│   ├── evaluation.py       # Test grading logic
│   ├── dataset.py          # Dataset loading
│   └── test_spec.py        # Test spec generation
├── setup.sh                # Environment setup script
├── src/
│   └── benchmark_service/  # Framework (template provided)
├── pyproject.toml
├── Dockerfile
├── Makefile
└── README.md
```

## License

This service uses the SWE-bench benchmark. Please cite the original work:

```bibtex
@inproceedings{jimenez2024swebench,
  title={SWE-bench: Can Language Models Resolve Real-World GitHub Issues?},
  author={Jimenez, Carlos E and Yang, John and Wettig, Alexander and Yao, Shunyu and Pei, Kexin and Press, Ofir and Narasimhan, Karthik},
  booktitle={ICLR},
  year={2024}
}
```
