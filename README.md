# SWE-bench Benchmark Service

A FastAPI-based service for evaluating software engineering agents on the SWE-bench benchmark. This service provides standardized endpoints for task retrieval, environment setup, and test-based evaluation of code changes.

## Overview

SWE-bench is a benchmark for evaluating AI agents on real-world software engineering tasks. This service serves **SWE-bench_Verified**, 500 carefully verified GitHub issues from popular Python repositories, and **SWE-bench Multimodal**, issues from JavaScript repositories whose descriptions include screenshots (the 480-task test split, plus the 100-task dev split for smokes).

Grading is dataset-driven: every served row carries its own `eval_script`, `log_parser`, `eval_type`, and `image`, which is how the pinned `swebench` harness (5.x) grades every SWE-bench dataset. Each dataset is pinned to a HuggingFace revision.

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

2. **Download the datasets** (required before running the service):
   ```bash
   make setup
   ```

   This downloads the 500 Verified instances to `/tmp/swe-bench-verified/`, the 480 Multimodal test instances to `/tmp/swe-bench-multimodal/`, and the 100 Multimodal dev instances to `/tmp/swe-bench-multimodal-dev/`.

3. **Run the development server:**
   ```bash
   make dev
   ```

The service will be available at `http://localhost:8001`. View API documentation at `http://localhost:8001/docs`.

### Docker Deployment

Build and run using Docker:

```bash
make docker-build
make docker-run
```

The datasets are downloaded during the Docker build process, so no separate setup is needed.

## Datasets

### SWE-bench_Verified (`default`)

- **Name:** SWE-bench_Verified
- **Source:** `SWE-bench/SWE-bench_Verified` (HuggingFace), pinned to revision `78f471bf655a3137b2e8a75af1501690ec009ec3`. The same 500 instances, commits, and patches as `princeton-nlp/SWE-bench_Verified`, plus the grading columns the harness now reads; two instances (`astropy__astropy-7606`, `django__django-10097`) lost a few PASS_TO_PASS tests upstream between the two copies.
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

### Vals Index Subset (`vals_index`)
The Vals Index subset can be fetched by passing in `dataset=vals_index` in the [/verify-task-ids](#verify-task-ids) endpoint

### SWE-bench Multimodal (`multimodal`, `multimodal_dev`)

- **Source:** `SWE-bench/SWE-bench_Multimodal` (HuggingFace), pinned to revision `4e6662d51c48e475f7f346e4fa09a6f8b31fcaa5`
- **`multimodal`:** the `test` split, 480 instances from 11 repositories (carbon 133, openlayers 77, lighthouse 54, bpmn-js 53, highlight.js 39, prism 38, next 36, grommet 20, prettier 13, eslint 11, scratch-gui 6), cached at `/tmp/swe-bench-multimodal/`. This is the split the public leaderboard reports.
- **`multimodal_dev`:** the `dev` split, 100 instances from 5 repositories (wp-calypso 37, Chart.js 22, p5.js 16, marked 14, react-pdf 11), cached at `/tmp/swe-bench-multimodal-dev/`. Kept for smokes and parity checks.

Instances carry the same fields as Verified plus `image_assets`, a map of the image URLs found in the `problem_statement`, `patch`, and `test_patch`. The problem statement is served verbatim, so the screenshots appear as GitHub markdown or HTML image links; an agent that wants to see them fetches those URLs itself.

Binary test assets (rendering baselines such as `expected.png`, which a text patch cannot carry) are listed under `image_assets.test_patch` with a source URL. At grading time the service fetches each one, uploads it to `/image_assets/` in the sandbox, and inserts restore commands into the evaluation script just before the test-output marker, so the files land after the script's own `git apply` and never live in the task image. Only https URLs on `raw.githubusercontent.com` (the host every pinned asset uses) are fetched, redirects are refused, and bodies over 20 MB are rejected. An asset that cannot be fetched fails the evaluation rather than scoring the model zero.

The revision is pinned because the upstream split has been reshaped in place before.

## Docker Images

Each task uses a pre-built Docker image with the repository and dependencies:

```
swebench/sweb.eval.x86_64.{task_id}:latest
```

These images are hosted on [Docker Hub](https://hub.docker.com/u/swebench) and contain:
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
- every `multimodal` and `multimodal_dev` task: browser suites under Xvfb (Chart.js, openlayers, lighthouse), Puppeteer, and Jest over monorepos

These receive 4 vCPU and 8 GB memory; `carbon-design-system/carbon` tasks receive 16 GB, because their eval runs Jest with four workers over a monorepo and in 8 GB a worker dies of a V8 heap overflow and the suite hangs.

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
  "prediction": "diff --git ...",
  "prediction_bytes": 1432,
  "prediction_truncated": false
}}
```

`prediction` echoes the captured patch up to 1 MiB; a larger patch is cut there with `prediction_truncated: true`, and `prediction_bytes` is its full size. The complete patch is the persisted prediction artifact.

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

1. **Capture Prediction:** Extract the agent's changes via `git diff` against the post-setup baseline. Untracked files over 1 MiB are left out (they are build or report artifacts such as Lighthouse's `latest-run` output, never source changes, and they made patches of tens of MB); the evaluation stream names each one. The capture is retried up to three times when git fails on a file a leftover agent process is still writing.
2. **Run Tests:** Execute the test suite using the task's evaluation script. A suite that produces no output for 30 minutes is treated as timed out and graded unresolved, as the SWE-bench harness treats a run over its time budget; the evaluation stream says so. The script's own log preamble is trimmed first: its `git show` becomes `git show --no-patch` and its base-commit diff becomes `--stat`, because the task images carry a squashed single-commit history (so `git show` is the whole repository as one diff) and the full diff repeats the agent's patch; neither is graded, and both flood the sandbox stream.
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
make help               # Show available commands
make install            # Install dependencies
make setup              # Download SWE-bench dataset
make dev                # Start development server
make lint               # Check code style
make format             # Format code
make typecheck          # Type check with basedpyright
make test               # Run tests
make test-experimental  # Run experimental tests (slow)
make docker-build       # Build Docker image
make docker-run         # Run Docker container
```

### Testing

The project includes comprehensive tests covering unit and integration tests:

```bash
# Run all tests (excluding experimental tests)
make test

# Run integration tests (requires Daytona credentials)
export DAYTONA_API_KEY="your-key"
export DAYTONA_API_URL="your-url"
export DAYTONA_TARGET="your-target"
make test

# Run experimental tests (slow, tests all 1,080 images)
make test-experimental
```

**Test Structure:**
- `tests/unit/test_registry.py` - Dataset validation, setup script verification
- `tests/unit/test_endpoints.py` - FastAPI endpoint tests
- `tests/integration/test_end_to_end.py` - Full workflow tests with Daytona sandboxes

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
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
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
