"""Test specification and script generation utilities."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from swebench.harness import utils as harness_utils
from swebench.harness.constants import START_TEST_OUTPUT
from swebench.harness.utils import TestSpec

__all__ = ["TestSpec", "make_test_spec"]


def make_test_spec(task: dict[str, Any]) -> TestSpec:
    """Build the harness TestSpec from a dataset row (its eval_script, log_parser, eval_type, image)."""
    return cast(TestSpec, harness_utils.make_test_spec(task))  # type: ignore[reportUnknownMemberType]

# Path inside the sandbox where the raw eval output is captured for grading.
# Grading reads this file (a plain pipe, not the interactive PTY) so that
# TTY-sensitive test reporters (e.g. sympy's `bin/test`, which switches to a
# carriage-return progress display on a TTY) still emit the per-line results
# that the SWE-bench log parsers require.
EVAL_OUTPUT_PATH = "/root/eval_output.log"

# Binary assets a text patch cannot carry (rendering baselines such as expected.png) are
# staged here before the eval script runs and copied into the working tree after its
# `git apply`, exactly as the harness does, so test data never lives in the task image.
IMAGE_ASSETS_DIR = "/image_assets"

# Commands the pre-5.0 harness ran while building each Verified environment, keyed by repo
# and version. Newer harness releases dropped the spec table (the images already contain
# their effect); this copy keeps the agent sandbox set up exactly as before.
_PRE_INSTALL_PATH = Path(__file__).parent / "verified_pre_install.json"


@lru_cache(maxsize=1)
def _pre_install_table() -> dict[str, dict[str, list[str]]]:
    return cast(dict[str, dict[str, list[str]]], json.loads(_PRE_INSTALL_PATH.read_text()))


def get_pre_install_commands(repo: str, version: str) -> list[str]:
    """
    Get pre-install commands for a specific repository and version.

    Args:
        repo: Repository name (e.g., "django/django")
        version: Version string

    Returns:
        List of pre-install commands, or empty list if none specified
    """
    return list(_pre_install_table().get(repo, {}).get(version, []))


def test_patch_assets(test_spec: TestSpec) -> list[dict[str, str]]:
    """Binary assets the patch and test patch need, each with its repo `path` and source `url`."""
    assets: list[dict[str, str]] = []
    declared = cast(dict[str, Any], getattr(test_spec, "image_assets", None) or {})
    for key in ("test_patch", "patch"):
        for entry in cast(list[Any], declared.get(key) or []):
            if isinstance(entry, dict):
                fields = cast(dict[str, Any], entry)
                if fields.get("path"):
                    assets.append({"path": str(fields["path"]), "url": str(fields.get("url") or "")})
    return assets


def asset_sandbox_path(repo_path: str) -> str:
    """Where a staged asset lives in the sandbox before the eval script restores it."""
    return f"{IMAGE_ASSETS_DIR}/{repo_path.replace('/', '__')}"


def asset_restore_commands(assets: list[dict[str, str]]) -> list[str]:
    """Shell lines that copy each staged asset into place, run after the eval script's `git apply`."""
    return [f"mkdir -p $(dirname {asset['path']}) && cp {asset_sandbox_path(asset['path'])} {asset['path']}" for asset in assets]


def create_evaluation_script(test_spec: TestSpec, task_id: str, restore_commands: list[str] | None = None) -> str:
    """
    Create the evaluation script for running tests.

    Args:
        test_spec: Test specification for the task
        task_id: Task identifier (used for Django locale fix)
        restore_commands: Asset-restore lines to insert just before the test-output start marker

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

    if restore_commands:
        lines = evaluation_script.split("\n")
        for index, line in enumerate(lines):
            if START_TEST_OUTPUT in line:
                lines[index:index] = restore_commands
                break
        else:
            lines.extend(restore_commands)
        evaluation_script = "\n".join(lines)

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


def task_row_summary(task: dict[str, Any]) -> dict[str, Any]:
    """The grading-relevant fields of a dataset row, for contract hashing."""
    return {
        "eval_script": task.get("eval_script"),
        "log_parser": task.get("log_parser"),
        "eval_type": task.get("eval_type"),
        "image_assets": task.get("image_assets"),
    }
