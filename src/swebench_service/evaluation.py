"""
Isolated file for grading the test output for a given instance.
We isolate this file from other utilities as all dependencies come from the swebench package.
"""

import re
import unicodedata

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

from swebench_service.schemas import EvaluationResult


def grade_test_output(test_output: str, test_spec: TestSpec, prediction: str | None) -> EvaluationResult:
    """
    Grade test output in memory using SWE-bench's logic.

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

    # Get log parser for this repo
    log_parser = MAP_REPO_TO_PARSER[test_spec.repo]

    # Extract content between markers
    test_content = test_output.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]

    # BUG: Strip ANSI escape sequences before stripping control chars; otherwise the ESC byte
    # is removed but the bracket remnants ([32m, [0m) corrupt test names and break split regexes.
    test_content = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", test_content)
    test_content = "".join(c for c in test_content if unicodedata.category(c)[0] != "C")

    # BUG: Split concatenated test results onto separate lines. The stream_command layer
    # can emit chunks without newlines, fusing adjacent test results together.
    # Django-style replacement variant: "test_A (class) ... test_B (class) ... ok"
    # test_A's result was lost (replaced by test_B's header) — split so test_B parses correctly.
    # Lookahead requires next word starts with "test" to avoid false-firing on "... oktest_B".
    test_content = re.sub(r"\.\.\. (?=test\w* \([^)\n]+\) \.\.\.)", "\n", test_content)

    # Django-style concatenation variant: "... ok<next_test>" -> "... ok\n<next_test>"
    test_content = re.sub(r"(\.\.\. (?:ok|OK|FAIL|ERROR|skipped))(?=\S)", r"\1\n", test_content)

    # BUG: Pytest-style "PASSED at start": "...real]PASSED lib/" -> "...real]\nPASSED lib/"
    test_content = re.sub(r"(?<=\S)((?:PASSED|FAILED|ERROR|SKIPPED|XFAIL) )", r"\n\1", test_content)

    # BUG: Pytest-style "PASSED at end": "test[Ba] PASSEDtest[Bi]" -> "test[Ba] PASSED\ntest[Bi]"
    test_content = re.sub(r"((?:PASSED|FAILED|ERROR|SKIPPED|XFAIL))(?=\S)", r"\1\n", test_content)

    # BUG: Pytest separator fused to test name: "test_foo=====" -> "test_foo\n====="
    test_content = re.sub(r"(?<=\S)(={5,})", r"\n\1", test_content)

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
