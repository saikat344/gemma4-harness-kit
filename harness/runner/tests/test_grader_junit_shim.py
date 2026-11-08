"""Audit F1 (2026-11-05): the grader's FixingContainerManager must get the same strict JUnit checks
from the REAL installed swegemma ``verify_task`` as the official ContainerManager.

swegemma 0.2.11 only rejects a missing JUnit XML when ``type(docker).__name__`` is a known real
sandbox class; the envfix wrapper reports ``ContainerManager`` for that reason. Only container
setup and Docker exec are faked here; ``verify_task`` and its JUnit validation run unmodified.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

verification = pytest.importorskip('swegemma.harness.verification')
from swegemma.models import Task  # noqa: E402
from swegemma.sandbox import ContainerConfig, ContainerManager  # noqa: E402

from harness.common.envfix import FixingContainerManager  # noqa: E402

REQUIRED = 'tests/test_x.py::test_fix'
_SETUP_FUNCS = ('setup_container_wheels', 'extract_snapshot', 'setup_synthetic_hardware', 'setup_git_exclude',
                'install_editable_package', 'install_test_dependencies', 'setup_workspace_test_config',
                'setup_baseline_commit')


def _xml(*, skipped: bool = False) -> str:
    inner = '<skipped message="s"/>' if skipped else ''
    return (f'<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="2" failures="0" errors="0" '
            f'skipped="{int(skipped)}"><testcase classname="tests.test_x" name="test_fix">{inner}</testcase>'
            '<testcase classname="tests.test_x" name="test_other"/></testsuite></testsuites>')


@pytest.fixture
def fake_docker(monkeypatch):
    """Fake the Docker side of ContainerManager; returns a dict to set the JUnit XML content."""
    from adk_eval_core.sandbox import ExecutionResult

    state = {'xml': ''}

    def exec_(self, container_id, command, *, timeout=None):
        if 'python3 -s -m pytest' in command:
            return ExecutionResult(status='ok', exit_code=0, stdout='.\n1 passed in 0.01s\n', stderr='')
        if command.startswith('cat /tmp/_swegemma_junit_'):
            return ExecutionResult(status='ok', exit_code=0, stdout=state['xml'], stderr='')
        return ExecutionResult(status='ok', exit_code=0, stdout='', stderr='')

    monkeypatch.setattr(ContainerManager, 'start', lambda self: 'cid-1')
    monkeypatch.setattr(ContainerManager, 'stop', lambda self, cid: None)
    monkeypatch.setattr(ContainerManager, 'exec', exec_)
    for name in _SETUP_FUNCS:
        monkeypatch.setattr(verification, name, lambda *a, **k: None)
    return state


def _verify(mgr, tmp_path):
    task = Task(instance_id='t_1', repo='org/r', base_commit='abc', problem_statement='p',
                FAIL_TO_PASS=(REQUIRED,))
    cfg = SimpleNamespace(harness=SimpleNamespace(verification_timeout_seconds=300, command_timeout_seconds=300),
                          results_dir=None)
    return asyncio.run(verification.verify_task(mgr, cfg, task, tmp_path / 't_1.tgz', agent_patch=''))


def _managers():
    return [ContainerManager(ContainerConfig(image='img')),
            FixingContainerManager(ContainerConfig(image='img'), instance_id='t_1', commands=[])]


def test_wrapper_reports_upstream_sandbox_name():
    assert type(FixingContainerManager()).__name__ == 'ContainerManager'

    class Sub(FixingContainerManager):  # e.g. the runner's RecordingContainerManager
        pass

    assert Sub.__name__ == 'ContainerManager'


@pytest.mark.parametrize('idx', [0, 1], ids=['ContainerManager', 'FixingContainerManager'])
def test_missing_junit_xml_is_rejected(fake_docker, tmp_path, idx):
    fake_docker['xml'] = ''
    res = _verify(_managers()[idx], tmp_path)
    assert res.resolved is False
    assert 'Missing JUnit XML' in (res.error or '')


@pytest.mark.parametrize('idx', [0, 1], ids=['ContainerManager', 'FixingContainerManager'])
def test_valid_junit_xml_is_accepted(fake_docker, tmp_path, idx):
    fake_docker['xml'] = _xml()
    res = _verify(_managers()[idx], tmp_path)
    assert res.resolved is True, res.error


@pytest.mark.parametrize('idx', [0, 1], ids=['ContainerManager', 'FixingContainerManager'])
def test_skipped_required_test_is_rejected(fake_docker, tmp_path, idx):
    fake_docker['xml'] = _xml(skipped=True)
    res = _verify(_managers()[idx], tmp_path)
    assert res.resolved is False
    assert 'Required test node did not pass' in (res.error or '')
