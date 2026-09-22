"""
Isolated file for grading the test output for a given instance.
We isolate this file from other utilities as all dependencies come from the swebench package.
"""

import re
import unicodedata

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
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
    PARSER_REGISTRY,
    SUITE_RAN,
    compute_fail_to_pass,
    compute_pass_to_pass,
    get_eval_tests_report,
    get_resolution_status,
)
from swebench.harness.utils import TestSpec

from swebench_service.schemas import EvaluationResult


def grade_test_output(test_output: str, test_spec: TestSpec, prediction: str | None) -> EvaluationResult:
    """
    Grade test output in memory using SWE-bench's logic.

    The parser and the eval type come from the dataset row (`log_parser`, `eval_type`),
    which is how the harness grades every SWE-bench dataset since 5.0.

    Args:
        test_output: The output from running tests
        test_spec: The test specification for this task
        prediction: The patch/diff that was applied (optional)

    Returns:
        EvaluationResult with resolved status, scores, and detailed test results
    """
    # Check for error codes
    bad_codes = [
        APPLY_PATCH_FAIL,
        RESET_FAILED,
        TESTS_ERROR,
        TESTS_TIMEOUT,
    ]

    if any(code in test_output for code in bad_codes):
        return EvaluationResult(
            patch_successfully_applied=False,
            resolved=False,
            resolution_status="NO",
            prediction=prediction,
        )

    # Check for test output markers
    if not (START_TEST_OUTPUT in test_output and END_TEST_OUTPUT in test_output):
        return EvaluationResult(
            patch_successfully_applied=False,
            resolved=False,
            resolution_status="NO",
            prediction=prediction,
        )

    # Get log parser for this task
    log_parser = PARSER_REGISTRY[test_spec.log_parser]

    # Extract content between markers
    test_content = test_output.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]

    # BUG: Split concatenated test results onto separate lines. The stream_command layer
    # can emit chunks without newlines, fusing adjacent test results together.
    # Django-style: "... ok<next_test>" -> "... ok\n<next_test>"
    test_content = re.sub(r"(\.\.\. (?:ok|OK|FAIL|ERROR|skipped))(?=\S)", r"\1\n", test_content)

    # BUG: Pytest-style: "...real]PASSED lib/" -> "...real]\nPASSED lib/"
    test_content = re.sub(r"(?<=\S)((?:PASSED|FAILED|ERROR|SKIPPED|XFAIL) )", r"\n\1", test_content)

    # Parse test content
    status_map = log_parser(test_content, test_spec)

    # Fallback to full content if nothing found between markers
    if not status_map:
        status_map = log_parser(test_output, test_spec)

    # No parsed results and no sign the suite ran: the run is invalid, not a pass. Under
    # EvalType.FAIL_ONLY an absent test counts as success, so without this a suite that
    # never started would grade as resolved.
    if not status_map and not SUITE_RAN.search(test_output):
        return EvaluationResult(
            patch_successfully_applied=False,
            resolved=False,
            resolution_status="NO",
            prediction=prediction,
        )

    # BUG: Remove all unicode characters that are control characters
    status_map = {"".join(c for c in k if unicodedata.category(c)[0] != "C"): v for k, v in status_map.items()}

    # === END IN-MEMORY get_logs_eval ===

    # Build gold results reference
    eval_ref = {
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }

    eval_type = EvalType(test_spec.eval_type)

    # Generate report
    report = get_eval_tests_report(status_map, eval_ref, eval_type=eval_type)  # type: ignore

    # Get resolution status
    resolution_status = get_resolution_status(report)

    # Calculate scores
    f2p_score = compute_fail_to_pass(report)
    p2p_score = compute_pass_to_pass(report)

    return EvaluationResult(
        prediction=prediction,
        patch_successfully_applied=True,
        resolved=resolution_status == ResolvedStatus.FULL.value,
        resolution_status=resolution_status,
        fail_to_pass={
            "success": report[FAIL_TO_PASS]["success"],
            "failure": report[FAIL_TO_PASS]["failure"],
        },
        pass_to_pass={
            "success": report[PASS_TO_PASS]["success"],
            "failure": report[PASS_TO_PASS]["failure"],
        },
        f2p_score=f2p_score,
        p2p_score=p2p_score,
        status_map=status_map,
    )
