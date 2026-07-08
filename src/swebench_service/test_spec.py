"""Test specification and script generation utilities."""

from typing import Any, cast

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.test_spec.test_spec import TestSpec

# Path inside the sandbox where the raw eval output is captured for grading.
# Grading reads this file (a plain pipe, not the interactive PTY) so that
# TTY-sensitive test reporters (e.g. sympy's `bin/test`, which switches to a
# carriage-return progress display on a TTY) still emit the per-line results
# that the SWE-bench log parsers require.
EVAL_OUTPUT_PATH = "/root/eval_output.log"


def get_pre_install_commands(repo: str, version: str) -> list[str]:
    """
    Get pre-install commands for a specific repository and version.

    Args:
        repo: Repository name (e.g., "django/django")
        version: Version string

    Returns:
        List of pre-install commands, or empty list if none specified
    """
    specs: dict[str, Any] = cast(
        dict[str, Any],
        MAP_REPO_VERSION_TO_SPECS.get(repo, {}).get(version, {}),  # type: ignore
    )

    return specs.get("pre_install", [])


def create_evaluation_script(test_spec: TestSpec, task_id: str) -> str:
    """
    Create the evaluation script for running tests.

    Args:
        test_spec: Test specification for the task
        task_id: Task identifier (used for Django locale fix)

    Returns:
        Evaluation script content as a string
    """
    evaluation_script = test_spec.eval_script

    # BUG: Scikit-learn C extensions use OpenMP for parallelism. In constrained sandbox
    # environments, thread oversubscription causes deadlocks during test execution.
    # Pinning to 1 thread prevents this while still allowing tests to pass correctly.
    if "scikit-learn" in task_id:
        evaluation_script = evaluation_script.replace(
            "set -uxo pipefail",
            "set -uxo pipefail\nexport OMP_NUM_THREADS=1\nexport OPENBLAS_NUM_THREADS=1",
        )

    # Django-specific fix for locale generation
    if "django" in task_id:
        evaluation_script = evaluation_script.replace("locale-gen", "locale-gen en_US.UTF-8")

    # BUG: Sphinx uses tox to run pytest, but the parser needs per-test PASSED/FAILED
    if test_spec.repo == "sphinx-doc/sphinx":
        evaluation_script = evaluation_script.replace(
            "tox --current-env -epy39 -v --",
            "tox --current-env -epy39 -v -- -rA",
        )

    return evaluation_script


def create_run_command(task_id: str) -> str:
    """
    Create the command to run the evaluation script.

    Args:
        task_id: Task identifier (used for pylint-specific setup)

    Returns:
        Shell command to execute the evaluation script
    """
    run_command = "cd /testbed"

    # Pylint-specific: clear PYTHONPATH
    if "pylint" in task_id:
        run_command += " && PYTHONPATH="

    # Increase recursion limit and run evaluation script.
    # Pipe eval.sh through `tee` so its stdout is a pipe (not the interactive PTY
    # that `sandbox.command` allocates). This keeps live streaming for the
    # watchdog while writing a faithful, non-TTY copy to EVAL_OUTPUT_PATH that
    # grading reads back — otherwise TTY-sensitive reporters like sympy render a
    # carriage-return progress bar with no parseable per-test lines.
    run_command += " && python3 -c 'import sys; sys.setrecursionlimit(10000)'"
    run_command += (
        " && GIT_PAGER=cat PAGER=cat LESS='-F -X' TERM=dumb"
        f" /bin/bash /root/eval.sh 2>&1 | tee {EVAL_OUTPUT_PATH}"
    )

    return run_command
