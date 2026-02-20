"""Test specification and script generation utilities."""

from typing import Any, cast

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.test_spec.test_spec import TestSpec


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

    # Django-specific fix for locale generation
    if "django" in task_id:
        evaluation_script = evaluation_script.replace("locale-gen", "locale-gen en_US.UTF-8")

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

    # Increase recursion limit and run evaluation script
    run_command += " && python3 -c 'import sys; sys.setrecursionlimit(10000)'"
    run_command += " && /bin/bash /root/eval.sh 2>&1"

    return run_command
