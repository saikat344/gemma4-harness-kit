"""--status-dir live status files (no docker, no network)."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from harness.runner import cli, status


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _writer(d, clock=None, **kw):
    kw.setdefault('tasks_total', 4)
    return status.StatusWriter(d, 'ab9-run', '/secret/dir/submissions/v1.zip', 'http://192.0.2.5:8000/v1', clock=clock or Clock(),
                               **kw)


def test_writes_json_and_index_atomically(tmp_path: Path):
    clock = Clock()
    w = _writer(tmp_path, clock, concurrency=2)
    w.write_once()
    j = json.loads((tmp_path / 'ab9-run.json').read_text())
    assert j['run'] == 'ab9-run' and j['state'] == 'running'
    assert (j['tasks_total'], j['tasks_done'], j['tasks_running'], j['current_task']) == (4, 0, [], None)
    assert j['resolved_so_far'] is None and j['eta_s'] is None and j['submission'] == 'v1'
    assert j['endpoint_kind'] == 'remote' and j['started'].endswith('Z')
    w.task_started('t1')
    w.task_started('t2')
    clock.t += 100
    w.task_finished('t1', 'done', 'submitted', empty_patch=True)
    clock.t += 100
    w.write_once()
    j = json.loads((tmp_path / 'ab9-run.json').read_text())
    assert (j['tasks_done'], j['tasks_running'], j['current_task'], j['empty_so_far']) == (1, ['t2'], 't2', 1)
    assert j['last_event'] == 'finished t1: done submitted'
    assert j['eta_s'] == int(3 * 100 / 2)  # 3 left x mean 100 s / concurrency 2
    idx = json.loads((tmp_path / 'index.json').read_text())
    assert [r['run'] for r in idx['runs']] == ['ab9-run'] and idx['runs'][0]['tasks_done'] == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ['ab9-run.json', 'index.json']  # no temp files left


def test_no_urls_keys_or_paths_leak(tmp_path: Path):
    w = _writer(tmp_path)
    w.task_started('django__django-1')
    w.write_once()
    text = (tmp_path / 'ab9-run.json').read_text()
    for needle in ('192.0.2.5', 'http', '8000', '/secret', 'submissions/', '.zip', 'key'):
        assert needle not in text, needle
    assert status.endpoint_kind('http://127.0.0.1:8000/v1') == 'local'
    assert status.endpoint_kind('https://x--gemma4-vllm.modal.run/v1') == 'modal'
    assert status.endpoint_kind('http://ssh4.vast.ai:1234/v1') == 'vast'
    assert status.safe_name('../../etc/pw d') == '.._.._etc_pw_d'.lstrip('.')


def test_resolved_from_grades_file_and_final_state(tmp_path: Path):
    grades = tmp_path / 'grades.jsonl'
    grades.write_text('{"instance_id": "a", "passed": true}\n{"instance_id": "b", "passed": false}\n')
    w = status.StatusWriter(tmp_path / 'st', 'r', 'sub', None, tasks_total=2, grades_path=grades, task_ids=['a', 'b'])
    (tmp_path / 'st').mkdir()
    w.start()
    w.task_started('a')
    w.task_finished('a', 'done', 'submitted')
    w.close('done', 'run finished')
    j = json.loads((tmp_path / 'st' / 'r.json').read_text())
    assert (j['state'], j['resolved_so_far'], j['eta_s'], j['tasks_running']) == ('done', 1, 0, [])
    assert j['endpoint_kind'] == 'local'


def test_index_lists_other_recent_runs_and_skips_old_and_broken(tmp_path: Path):
    now = 1_800_000_000.0
    (tmp_path / 'older.json').write_text(json.dumps({'run': 'older', 'state': 'done', 'updated': status.iso(now - 3600),
                                                     'tasks_done': 5, 'tasks_total': 5}))
    (tmp_path / 'ancient.json').write_text(json.dumps({'run': 'ancient', 'state': 'done',
                                                       'updated': status.iso(now - 30 * 86400)}))
    (tmp_path / 'broken.json').write_text('{nope')
    w = _writer(tmp_path, Clock(now))
    w.write_once()
    idx = json.loads((tmp_path / 'index.json').read_text())
    assert [r['run'] for r in idx['runs']] == ['ab9-run', 'older']


def test_disabled_when_no_dir_is_a_noop():
    w = _writer(None).start()
    w.task_started('x')
    w.task_finished('x', 'done')
    w.close()  # nothing to write, nothing raised


def test_unwritable_dir_logs_once_and_never_raises(tmp_path: Path, capsys):
    blocked = tmp_path / 'file'
    blocked.write_text('i am a file, not a dir')
    for d in (blocked / 'sub', '/proc/definitely/not/writable'):
        w = _writer(d).start()
        w.task_started('t1')
        w.task_finished('t1', 'done')
        w.task_started('t2')
        w.close('failed', 'x')
        assert w.write_errors >= 1
    err = capsys.readouterr().err
    assert err.count('status-dir unavailable') == 2  # once per writer, however many attempts failed


def test_readonly_dir_is_survived(tmp_path: Path, capsys):
    if os.geteuid() == 0:
        pytest.skip('root ignores permissions')
    ro = tmp_path / 'ro'
    ro.mkdir()
    ro.chmod(0o500)
    try:
        w = _writer(ro).start()
        w.task_started('t1')
        w.close()
        assert w.write_errors >= 1 and capsys.readouterr().err.count('status-dir unavailable') == 1
        assert list(ro.iterdir()) == []
    finally:
        ro.chmod(0o700)


def test_hanging_share_never_slows_the_caller(tmp_path: Path):
    w = _writer(tmp_path)
    gate = threading.Event()
    w.write_once = lambda: gate.wait(30)  # an SMB write that hangs
    w.start()
    t0 = time.time()
    for i in range(200):
        w.task_started(f't{i}')
        w.task_finished(f't{i}', 'done')
    w.close(timeout=0.2)  # bounded even though the writer is stuck
    assert time.time() - t0 < 3
    gate.set()


def test_heartbeat_rewrites_without_events(tmp_path: Path):
    w = status.StatusWriter(tmp_path, 'hb', 's', None, tasks_total=1, interval_s=0.05).start()
    deadline = time.time() + 5
    seen = set()
    while time.time() < deadline and len(seen) < 2:
        try:
            seen.add(json.loads((tmp_path / 'hb.json').read_text())['updated'])
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    w.close()
    assert len(seen) >= 1  # 'updated' has 1 s resolution; the file keeps being rewritten
    assert json.loads((tmp_path / 'hb.json').read_text())['state'] == 'done'


def test_cli_flag_and_env_default(monkeypatch, tmp_path: Path):
    monkeypatch.delenv('GEMMA4_STATUS_DIR', raising=False)
    args = cli.build_parser().parse_args(['run', '--submission', 's', '--out', 'o'])
    assert args.status_dir is None
    monkeypatch.setenv('GEMMA4_STATUS_DIR', str(tmp_path))
    assert cli.build_parser().parse_args(['run', '--submission', 's', '--out', 'o']).status_dir == str(tmp_path)
    args = cli.build_parser().parse_args(['run', '--submission', 's', '--out', 'o', '--status-dir', '/x'])
    assert args.status_dir == '/x'
