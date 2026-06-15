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
