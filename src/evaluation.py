"""
Isolated file for grading the test output for a given instance.
We isolate this file from utils.py as all dependencies come from the swebench package.
"""

from typing import Any

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_ONLY_REPOS,
    FAIL_TO_PASS,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
)
from swebench.harness.grading import (
    TestSpec,
    compute_fail_to_pass,
    compute_pass_to_pass,
    get_eval_tests_report,
    get_resolution_status,
)
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER


def grade_test_output(test_output: str, instance_id: str) -> dict[str, Any]:
    """
    Grade test output in memory using SWE-bench's logic.

    Returns:
        dict with resolved status, scores, and detailed test results
    """
    from src.utils import fetch_test_spec

    test_spec: TestSpec = fetch_test_spec(instance_id)

    # Check for error codes
    bad_codes = [
        APPLY_PATCH_FAIL,
        RESET_FAILED,
        TESTS_ERROR,
        TESTS_TIMEOUT,
    ]

    if any(code in test_output for code in bad_codes):
        return {
            "instance_id": instance_id,
            "patch_successfully_applied": False,
            "resolved": False,
            "resolution_status": "NO",
        }

    # Check for test output markers
    if not (START_TEST_OUTPUT in test_output and END_TEST_OUTPUT in test_output):
        return {
            "instance_id": instance_id,
            "patch_successfully_applied": False,
            "resolved": False,
            "resolution_status": "NO",
        }

    # Get log parser for this repo
    log_parser = MAP_REPO_TO_PARSER[test_spec.repo]

    # Extract content between markers
    test_content = test_output.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]

    # Parse test content
    status_map = log_parser(test_content, test_spec)

    # Fallback to full content if nothing found between markers
    if not status_map:
        status_map = log_parser(test_output, test_spec)

    # === END IN-MEMORY get_logs_eval ===

    # Build gold results reference
    eval_ref = {
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }

    # Determine eval type
    eval_type = EvalType.FAIL_ONLY if test_spec.repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL

    # Generate report
    report = get_eval_tests_report(status_map, eval_ref, eval_type=eval_type)  # type: ignore

    # Get resolution status
    resolution_status = get_resolution_status(report)

    # Calculate scores
    f2p_score = compute_fail_to_pass(report)
    p2p_score = compute_pass_to_pass(report)

    return {
        "instance_id": instance_id,
        "patch_successfully_applied": True,
        "resolved": resolution_status == ResolvedStatus.FULL.value,
        "resolution_status": resolution_status,
        "fail_to_pass": {
            "success": report[FAIL_TO_PASS]["success"],
            "failure": report[FAIL_TO_PASS]["failure"],
        },
        "pass_to_pass": {
            "success": report[PASS_TO_PASS]["success"],
            "failure": report[PASS_TO_PASS]["failure"],
        },
        "f2p_score": f2p_score,
        "p2p_score": p2p_score,
        "status_map": status_map,
    }
