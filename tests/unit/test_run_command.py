from swebench_service.test_spec import create_run_command


def test_create_run_command_disables_interactive_git_pagers() -> None:
    """Evaluation scripts can call git commands that would otherwise page.

    Test cases:
    - Git pager environment variables are disabled before running eval.sh.
    - Pylint tasks still clear PYTHONPATH.
    """
    command = create_run_command("astropy__astropy-12907")
    pylint_command = create_run_command("pylint-dev__pylint-7080")

    assert "GIT_PAGER=cat PAGER=cat LESS='-F -X' TERM=dumb" in command
    assert "GIT_PAGER=cat PAGER=cat LESS='-F -X' TERM=dumb" in pylint_command
    assert "PYTHONPATH=" in pylint_command


def test_create_run_command_tees_eval_output_to_file() -> None:
    """Eval output must be captured to a file for grading.

    `sandbox.command` runs the eval inside an interactive PTY, on which
    TTY-sensitive reporters (e.g. sympy's `bin/test`) emit a carriage-return
    progress bar with no parseable per-test lines. Teeing eval.sh through a pipe
    makes its stdout non-TTY, restoring the per-test lines the parser needs.
    """
    from swebench_service.test_spec import EVAL_OUTPUT_PATH

    command = create_run_command("sympy__sympy-21612")

    assert f"/bin/bash /root/eval.sh 2>&1 | tee {EVAL_OUTPUT_PATH}" in command
