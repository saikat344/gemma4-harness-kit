"""Audit F7 (2026-11-05): a failed required env repair must surface as a setup failure, and the
manager must record which repairs actually succeeded."""

from __future__ import annotations

import pytest

pytest.importorskip('swegemma')
from adk_eval_core.sandbox import ExecutionResult  # noqa: E402
from swegemma.harness.container_setup import ContainerSetupError, install_test_dependencies  # noqa: E402
from swegemma.sandbox import ContainerConfig, ContainerManager  # noqa: E402

from harness.common.envfix import EnvFixError, FixingContainerManager  # noqa: E402

SETUP = 'python3 /tmp/setup.py --fast-path org/r'


@pytest.fixture
def fake_exec(monkeypatch):
    """ContainerManager.exec fake: exit codes by command, every call recorded."""
    codes: dict[str, int] = {}
    calls: list[str] = []

    def exec_(self, container_id, command, *, timeout=None):
        calls.append(command)
        code = codes.get(command, 0)
        return ExecutionResult(status='ok' if code == 0 else 'error', exit_code=code,
                               stdout='', stderr='boom' if code else '')

    monkeypatch.setattr(ContainerManager, 'exec', exec_)
    return codes, calls


def _mgr(cmds):
    return FixingContainerManager(ContainerConfig(image='img'), instance_id='t_1', commands=cmds)


def test_failed_repair_after_successful_setup_raises_setup_error(fake_exec):
    codes, calls = fake_exec
    codes['fix2'] = 1
    mgr = _mgr(['fix1', 'fix2', 'fix3'])
    with pytest.raises(ContainerSetupError) as ei:
        mgr.exec('cid', SETUP)
    assert isinstance(ei.value, EnvFixError) and 'fix2' in str(ei.value)
    assert calls == [SETUP, 'fix1', 'fix2']  # stops at the first failure
    assert mgr.applied_fixes == ['fix1']
    assert [(r['command'], r['ok']) for r in mgr.fix_results] == [('fix1', True), ('fix2', False)]
    assert len(mgr.fix_log) == 2 and 'exit=1' in mgr.fix_log[1]


def test_successful_repairs_return_setup_result(fake_exec):
    _, calls = fake_exec
    mgr = _mgr(['fix1', 'fix2'])
    assert mgr.exec('cid', SETUP).exit_code == 0
    assert calls == [SETUP, 'fix1', 'fix2']
    assert mgr.applied_fixes == ['fix1', 'fix2']


def test_non_setup_commands_run_no_repairs(fake_exec):
    _, calls = fake_exec
    mgr = _mgr(['fix1'])
    mgr.exec('cid', 'echo hi')
    assert calls == ['echo hi'] and mgr.fix_results == []


def test_upstream_setup_step_propagates_repair_failure(fake_exec, monkeypatch, tmp_path):
    """Through the real swegemma install_test_dependencies (the caller that runs setup.py): it only
    logs a nonzero setup.py exit, but our repair failure must reach its caller as ContainerSetupError."""
    import swegemma.harness.container_setup as cs

    codes, calls = fake_exec
    setup_py = tmp_path / 'setup.py'
    setup_py.write_text('')
    monkeypatch.setattr(cs, '_ensure_container_site_packages', lambda *a, **k: None)
    monkeypatch.setattr(cs, 'resolve_sandbox_setup_script', lambda config: setup_py)
    monkeypatch.setattr(ContainerManager, 'copy_to', lambda self, *a, **k: None)
    codes['fix1'] = 1
    mgr = _mgr(['fix1'])
    with pytest.raises(ContainerSetupError):
        install_test_dependencies(mgr, 'cid', 'org/r', fast_path=True, config=None)
    assert calls[0] == SETUP and calls[1] == 'fix1'
    assert calls[-1] == 'rm -f /tmp/setup.py'  # upstream cleanup still ran


def test_grader_records_repair_failure_as_setup_error(fake_exec, monkeypatch, tmp_path):
    import argparse
    import asyncio

    import swegemma.harness.verification as verification
    from swegemma.models import Task

    from harness.grader import cli as grader

    import harness.common.envfix as envfix

    codes, _ = fake_exec
    codes['repair-b'] = 2
    monkeypatch.setattr(envfix, 'fix_commands', lambda iid, repo='', fixes=None: ['repair-a', 'repair-b'])

    async def fake_verify(mgr, cfg, task, snap, **kw):
        mgr.exec('cid', SETUP)  # what install_test_dependencies does inside verify_task
        raise AssertionError('unreachable')

    monkeypatch.setattr(verification, 'verify_task', fake_verify)
    monkeypatch.setattr(ContainerManager, 'cleanup_all', lambda self: None)
    (tmp_path / 't_1.tgz').write_bytes(b'x')
    task = Task(instance_id='t_1', repo='org/r', base_commit='abc', problem_statement='p',
                FAIL_TO_PASS=('tests/test_x.py::test_fix',))
    args = argparse.Namespace(no_env_fixes=False, run_empty_patches=False, network='none', task_timeout=60)
    row = asyncio.run(grader._grade_one(task, 'diff --git a/x b/x\n', mode='predictions', args=args, cfg=None,
                                        image='img', snapshots=tmp_path, logs_dir=tmp_path))
    assert row['passed'] is False and row['error_class'] == 'setup_error'
    assert row['env_fixes_applied'] == ['repair-a'] and row['env_fixes_failed'] == ['repair-b']
    assert 'exit=2' in (tmp_path / 't_1.log').read_text()
