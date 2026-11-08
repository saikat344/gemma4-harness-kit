"""Resume guards: the runner refuses to mix result conditions in one run dir, the grader re-grades stale rows."""
from __future__ import annotations

import copy
from pathlib import Path

from harness.grader import cli as grader
from harness.runner import cli as runner

BASE = {
    'submission_sha256': 'abc', 'budgets': {'max_time_minutes': 5.0, 'max_tool_calls': 50, 'overrides': {}},
    'compaction': {'token_threshold': 14336}, 'context_cache': None, 'served_model': 'gemma',
    'image': {'image': 'swebench-sandbox:latest', 'id': 'sha256:1'},
    'data': {'tasks_path': '/d/tasks.jsonl', 'snapshots_dir': '/d/snapshots', 'wheels_dir': None},
    'env_fixes': True, 'timing_scale': 1.0, 'checkpoints': [], 'git': 'r1', 'concurrency': 1,
}


def _with(path: str, value):
    m = copy.deepcopy(BASE)
    *head, last = path.split('.')
    d = m
    for p in head:
        d = d[p]
    d[last] = value
    return m


def test_same_conditions_resume_cleanly():
    assert runner.condition_diff(BASE, copy.deepcopy(BASE)) == ([], [])


def test_budget_override_is_a_result_condition():
    cur = _with('budgets', {'max_time_minutes': 15.0, 'max_tool_calls': 50,
                            'overrides': {'max_time_minutes': {'from': 5.0, 'to': 15.0}}})
    assert runner.condition_diff(BASE, cur)[0] == ['budgets']


def test_env_fixes_image_and_checkpoints_are_result_conditions():
    assert runner.condition_diff(BASE, _with('env_fixes', False))[0] == ['env_fixes']
    assert runner.condition_diff(BASE, _with('image.id', 'sha256:2'))[0] == ['image.id']
    assert runner.condition_diff(BASE, _with('checkpoints', [5, 7.5]))[0] == ['checkpoints']


def test_git_and_concurrency_are_soft():
    cur = _with('git', 'r2')
    cur['concurrency'] = 2
    assert runner.condition_diff(BASE, cur) == ([], ['git', 'concurrency'])


def test_old_manifest_without_a_key_still_resumes():
    old = copy.deepcopy(BASE)
    del old['checkpoints']
    assert runner.condition_diff(old, _with('checkpoints', [5]))[0] == []


def test_conditions_sha_ignores_soft_fields():
    assert runner.conditions_sha256(BASE) == runner.conditions_sha256(_with('git', 'r9'))
    assert runner.conditions_sha256(BASE) != runner.conditions_sha256(_with('env_fixes', False))


def _key(**kw):
    args = dict(mode='predictions', image_id='sha256:1', network='none', verification_timeout=300, task_timeout=3600,
                env_fixes=True, run_empty_patches=False, setup_py=None, wheels_dir=None)
    args.update(kw)
    return grader.grade_key(**args)


def test_grade_key_tracks_conditions(tmp_path):
    assert _key() == _key()
    assert _key() != _key(network='bridge')
    assert _key() != _key(image_id='sha256:2')
    assert _key() != _key(env_fixes=False)
    assert _key() != _key(verification_timeout=600)
    assert _key() != _key(task_timeout=600)
    setup = tmp_path / 'setup.py'
    setup.write_text('print(1)\n')
    k1 = _key(setup_py=setup)
    assert k1 != _key()
    setup.write_text('print(2)\n')
    assert _key(setup_py=setup) != k1
    wheels = tmp_path / 'wheels'
    wheels.mkdir()
    (wheels / 'a-1.0-py3-none-any.whl').write_bytes(b'x')
    w1 = _key(wheels_dir=wheels)
    assert w1 != _key()  # --no-wheels differs from a wheels dir
    (wheels / 'b-1.0-py3-none-any.whl').write_bytes(b'y')
    assert _key(wheels_dir=wheels) != w1


def test_grade_key_ignores_tasks_path():
    """The tasks file location is not a grading condition; its content is tracked per task."""
    import inspect
    assert 'tasks_path' not in inspect.signature(grader.grade_key).parameters


def _rows(key, task_keys, preds_old):
    return {iid: {'instance_id': iid, 'patch_sha256': grader.patch_sha256(p), 'grade_key': key,
                  'task_sha256': task_keys.get(iid, 'x')} for iid, p in preds_old.items()}


def test_reusable_rows_drop_changed_patches_keys_and_legacy_rows():
    key = _key()
    tk = {k: f't{k}' for k in 'abcdef'}
    preds = {'a': 'diff a', 'b': 'diff b2', 'c': 'diff c', 'd': 'diff d', 'f': 'diff f'}
    done = {
        'a': {'instance_id': 'a', 'patch_sha256': grader.patch_sha256('diff a'), 'grade_key': key, 'task_sha256': 'ta'},
        'b': {'instance_id': 'b', 'patch_sha256': grader.patch_sha256('diff b1'), 'grade_key': key,
              'task_sha256': 'tb'},  # patch changed
        'c': {'instance_id': 'c', 'patch_sha256': grader.patch_sha256('diff c'), 'grade_key': 'other',
              'task_sha256': 'tc'},  # env changed
        'd': {'instance_id': 'd'},  # written before the check
        'e': {'instance_id': 'e', 'patch_sha256': grader.patch_sha256(''), 'grade_key': key,
              'task_sha256': 'te'},  # not predicted now
        'f': {'instance_id': 'f', 'patch_sha256': grader.patch_sha256('diff f'), 'grade_key': key},  # no task hash
    }
    keep, dropped = grader.reusable_rows(done, preds, key, tk)
    assert list(keep) == ['a'] and dropped == 5


def _write_tasks(path, tasks):
    import json
    path.write_text(''.join(json.dumps(t) + '\n' for t in tasks), encoding='utf-8')


def _task(iid, test_name):
    test_patch = (f'diff --git a/tests/test_{iid}.py b/tests/test_{iid}.py\n'
                  f'--- a/tests/test_{iid}.py\n+++ b/tests/test_{iid}.py\n'
                  f'@@ -0,0 +1,2 @@\n+def {test_name}():\n+    assert True\n')
    return {'instance_id': iid, 'repo': 'org/r', 'base_commit': 'abc', 'problem_statement': 'p',
            'patch': '', 'test_patch': test_patch, 'FAIL_TO_PASS': [f'tests/test_{iid}.py::{test_name}'],
            'PASS_TO_PASS': []}


def _task_keys(tasks_file, snaps):
    from swegemma.models import load_tasks
    from harness.common import paths
    return {t.instance_id: grader.task_sha256(t, paths.snapshot_path(t.instance_id, snaps))
            for t in load_tasks(tasks_file)}


def test_changed_test_patch_at_same_path_invalidates_only_that_row(tmp_path):
    """Audit F5 reproduction: rewriting test_patch in place must not reuse the old verdict."""
    tasks_file, snaps = tmp_path / 'tasks.jsonl', tmp_path / 'snaps'
    snaps.mkdir()
    _write_tasks(tasks_file, [_task('a', 'test_old'), _task('b', 'test_b')])
    key = _key()
    preds = {'a': 'diff a', 'b': 'diff b'}
    tk1 = _task_keys(tasks_file, snaps)
    done = _rows(key, tk1, preds)
    assert len(grader.reusable_rows(done, preds, key, tk1)[0]) == 2

    _write_tasks(tasks_file, [_task('a', 'test_new'), _task('b', 'test_b')])  # same path, new expected test
    tk2 = _task_keys(tasks_file, snaps)
    assert tk2['a'] != tk1['a'] and tk2['b'] == tk1['b']
    keep, dropped = grader.reusable_rows(done, preds, key, tk2)
    assert list(keep) == ['b'] and dropped == 1


def test_adding_or_editing_unrelated_task_keeps_other_rows(tmp_path):
    tasks_file, snaps = tmp_path / 'tasks.jsonl', tmp_path / 'snaps'
    snaps.mkdir()
    _write_tasks(tasks_file, [_task('a', 'test_a'), _task('b', 'test_b')])
    key = _key()
    preds = {'a': 'diff a', 'b': 'diff b'}
    done = _rows(key, _task_keys(tasks_file, snaps), preds)
    _write_tasks(tasks_file, [_task('a', 'test_a'), _task('b', 'test_b2'), _task('c', 'test_c')])
    keep, dropped = grader.reusable_rows(done, preds, key, _task_keys(tasks_file, snaps))
    assert list(keep) == ['a'] and dropped == 1


def test_snapshot_identity_is_part_of_the_task_key(tmp_path):
    import os
    from harness.common import paths
    tasks_file, snaps = tmp_path / 'tasks.jsonl', tmp_path / 'snaps'
    snaps.mkdir()
    _write_tasks(tasks_file, [_task('a', 'test_a')])
    missing = _task_keys(tasks_file, snaps)['a']
    snap = paths.snapshot_path('a', snaps)
    snap.write_bytes(b'v1')
    present = _task_keys(tasks_file, snaps)['a']
    assert present != missing
    snap.write_bytes(b'v2-longer')
    assert _task_keys(tasks_file, snaps)['a'] != present
    st = snap.stat()
    os.utime(snap, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # same size, new mtime
    assert _task_keys(tasks_file, snaps)['a'] not in (present, missing)


def test_task_content_change_is_a_result_condition_but_subset_change_is_not():
    prev = _with('data.task_sha256', {'t1': 'aaa', 't2': 'bbb'})
    assert runner.condition_diff(prev, _with('data.task_sha256', {'t1': 'aaa', 't3': 'ccc'}))[0] == []
    assert runner.condition_diff(prev, _with('data.task_sha256', {'t1': 'XXX'}))[0] == ['data.task_sha256']


def test_grade_key_includes_grader_semantics(monkeypatch):
    # A grader behaviour change (version bump) must invalidate resumed rows.
    k1 = _key()
    monkeypatch.setattr(grader, 'GRADER_SEMANTICS', grader.GRADER_SEMANTICS + 1)
    assert _key() != k1
