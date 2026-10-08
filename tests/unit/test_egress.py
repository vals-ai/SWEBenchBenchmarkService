"""Task policies follow the setup and evaluation tools for each dataset."""

import pytest

from swebench_service.benchmark_service import SWEBenchService, _build_setup_script


@pytest.fixture(autouse=True)
def setup_dataset() -> None:
    """These retrieval tests provide task metadata without downloading datasets."""


@pytest.mark.parametrize(
    ("task_id", "repo", "version", "dataset", "setup_hosts", "evaluation_hosts"),
    [
        ("django-task", "django/django", "1.11", "default",
         {"archive.ubuntu.com", "security.ubuntu.com"}, {"pypi.org", "files.pythonhosted.org"}),
        ("matplotlib-task", "matplotlib/matplotlib", "3.7", "default",
         {"archive.ubuntu.com", "security.ubuntu.com", "www.qhull.org"}, {"pypi.org", "files.pythonhosted.org"}),
        ("astropy-task", "astropy/astropy", "5.0", "default",
         set(), {"pypi.org", "files.pythonhosted.org"}),
        ("sphinx-doc__sphinx-10323", "sphinx-doc/sphinx", "5.0", "default",
         set(), {"pypi.org", "files.pythonhosted.org"}),
        ("openlayers__openlayers-14932", "openlayers/openlayers", "7.1", "multimodal",
         set(), {"registry.npmjs.org", "registry.yarnpkg.com", "storage.googleapis.com"}),
    ],
)
async def test_retrieve_task_selects_package_access(
    task_id: str, repo: str, version: str, dataset: str,
    setup_hosts: set[str], evaluation_hosts: set[str],
) -> None:
    service = SWEBenchService()
    service.datasets = {dataset: {task_id: {"repo": repo, "version": version, "image": "swebench/test:latest"}}}
    response = await service.retrieve_task(task_id, dataset=dataset)
    policy = response.model_dump(mode="json")["egress"]
    assert set(policy["setup_task"]) == setup_hosts
    assert set(policy["evaluation"]) == evaluation_hosts
    assert setup_hosts | evaluation_hosts <= set(policy["run"])
    assert "*" not in policy["run"]


@pytest.mark.parametrize("task_id", ["sphinx-doc__sphinx-10614", "sphinx-doc__sphinx-11510"])
async def test_sphinx_graphviz_setup_has_package_access(task_id: str) -> None:
    task = {
        "repo": "sphinx-doc/sphinx",
        "version": "7.2",
        "image": f"swebench/sweb.eval.x86_64.{task_id.replace('__', '_1776_')}:latest",
    }
    service = SWEBenchService()
    service.datasets = {"default": {task_id: task}}

    assert "apt-get update && apt-get install -y graphviz" in _build_setup_script(task)
    response = await service.retrieve_task(task_id, dataset="default")
    policy = response.model_dump(mode="json")["egress"]
    apt_hosts = {"archive.ubuntu.com", "security.ubuntu.com"}
    assert set(policy["setup_task"]) == apt_hosts
    assert apt_hosts <= set(policy["run"])
    assert set(policy["evaluation"]) == {"pypi.org", "files.pythonhosted.org"}
