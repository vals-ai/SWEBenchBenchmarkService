"""
Isolated file for grading the test output for a given instance.
We isolate this file from other utilities as all dependencies come from the swebench package.
"""

import re
import unicodedata
from typing import Any, NamedTuple

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
    TestStatus,
)
from swebench.harness.grading import (
    PARSER_REGISTRY,
    SUITE_RAN,
    compute_fail_to_pass,
    compute_pass_to_pass,
    get_eval_tests_report,
    get_resolution_status,
    parse_test_exit_code,
)
from swebench.harness.utils import TestSpec

from swebench_service.schemas import EvaluationResult

# The result is one WebSocket frame, and the framework client drops a frame over 10 MiB, which
# ends the evaluation and every resume of it. The echoed patch is the only unbounded field, so
# only its head is decoded and sent; the full patch is the persisted prediction artifact, and
# `prediction_bytes` gives its real size.
MAX_ECHOED_PREDICTION_BYTES = 1024 * 1024


class EchoedPrediction(NamedTuple):
    """What the evaluation result says about the captured patch."""

    text: str | None
    size: int | None
    truncated: bool

    def fields(self) -> dict[str, Any]:
        return {"prediction": self.text, "prediction_bytes": self.size, "prediction_truncated": self.truncated}


def echo_prediction(patch: bytes) -> EchoedPrediction:
    """Decode at most MAX_ECHOED_PREDICTION_BYTES of the patch, never splitting a character."""
    if not patch:
        return EchoedPrediction(None, None, False)
    if len(patch) <= MAX_ECHOED_PREDICTION_BYTES:
        return EchoedPrediction(patch.decode("utf-8", errors="replace"), len(patch), False)
    cut = MAX_ECHOED_PREDICTION_BYTES
    while cut > 0 and patch[cut] & 0xC0 == 0x80:  # a continuation byte: back up to the split character's lead byte
        cut -= 1
    return EchoedPrediction(patch[:cut].decode("utf-8", errors="replace"), len(patch), True)


def grade_test_output(
    test_output: str, test_spec: TestSpec, prediction: EchoedPrediction | str | None
) -> EvaluationResult:
    """
    Grade test output in memory using SWE-bench's logic.

    The parser and the eval type come from the dataset row (`log_parser`, `eval_type`),
    which is how the harness grades every SWE-bench dataset since 5.0.

    Args:
        test_output: The output from running tests
        test_spec: The test specification for this task
        prediction: The captured patch, as `echo_prediction` describes it (a plain string is echoed as is)

    Returns:
        EvaluationResult with resolved status, scores, and detailed test results
    """
    if not isinstance(prediction, EchoedPrediction):
        prediction = echo_prediction(prediction.encode("utf-8") if prediction is not None else b"")
    echoed = prediction.fields()

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
            **echoed,
        )

    # Check for test output markers
    if not (START_TEST_OUTPUT in test_output and END_TEST_OUTPUT in test_output):
        return EvaluationResult(
            patch_successfully_applied=False,
            resolved=False,
            resolution_status="NO",
            **echoed,
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
            **echoed,
        )

    # A patch can print its own "PASSED" lines (from a conftest.py hook, say), so
    # cross-check the log against the test command's exit status, which the eval script
    # records after the end marker. Exiting non-zero while reporting no failure at all
    # means the log is not describing the run that actually happened.
    exit_code = parse_test_exit_code(test_output)
    reported_failure = any(status in (TestStatus.FAILED.value, TestStatus.ERROR.value) for status in status_map.values())
    if exit_code not in (None, 0) and status_map and not reported_failure:
        return EvaluationResult(
            patch_successfully_applied=False,
            resolved=False,
            resolution_status="NO",
            **echoed,
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
        **echoed,
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
