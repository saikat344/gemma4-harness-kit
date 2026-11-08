"""Live status files for a dashboard (`--status-dir DIR` / $GEMMA4_STATUS_DIR).

While a run is active, DIR/<run-name>.json holds a small progress record and DIR/index.json lists recent runs. DIR may
be a network share read by another machine (SMB has no symlinks, so every file is written to a temp name and
os.replace()d).

Safety rules (this directory is shared over the network):
  * no secrets, prompts, endpoint URLs or keys, and no file paths: the run name is the only name that appears
    (`endpoint_kind` is local|modal|vast|remote, never the URL);
  * the run must never crash or slow down because of it: writes happen on a daemon thread, every error is swallowed,
    and an unavailable directory is logged once on stderr and then ignored (the share can disappear at any time).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

INTERVAL_S = 15.0
INDEX_NAME = 'index.json'
INDEX_MAX_RUNS = 20
INDEX_MAX_AGE_S = 7 * 24 * 3600
LAST_EVENT_MAX = 120


def iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def safe_name(name: str) -> str:
    """The run name as a file stem: letters, digits, dot, dash, underscore only."""
    return re.sub(r'[^A-Za-z0-9._-]', '_', name).lstrip('.')[:100] or 'run'


def endpoint_kind(url: str | None) -> str:
    u = (url or '').lower()
    if 'modal.run' in u or 'modal.' in u:
        return 'modal'
    if 'vast' in u:
        return 'vast'
    host = re.sub(r'^[a-z]+://', '', u).split('/')[0].split(':')[0]
    if host in ('127.0.0.1', 'localhost', '::1', '[::1]', '0.0.0.0', ''):
        return 'local'
    return 'remote'


class StatusWriter:
    """Thread-safe progress tracker that mirrors itself into DIR/<run>.json. All methods are safe to call from the
    event loop: they only update memory and wake the writer thread."""

    def __init__(self, status_dir: str | os.PathLike | None, run: str, submission: str, endpoint: str | None,
                 tasks_total: int, tasks_done: int = 0, empty: int = 0, concurrency: int = 1,
                 grades_path: str | os.PathLike | None = None, task_ids: list[str] | None = None,
                 interval_s: float = INTERVAL_S, clock=time.time):
        self.enabled = bool(status_dir)
        self.dir = Path(status_dir) if status_dir else None
        self.run = safe_name(run)
        self.submission = safe_name(Path(str(submission)).stem if submission else '')
        self.kind = endpoint_kind(endpoint)
        self.host = platform.node()
        self.total = tasks_total
        self.done = tasks_done
        self.empty = empty
        self.concurrency = max(1, concurrency)
        self.grades_path = Path(grades_path) if grades_path else None
        self.task_ids = set(task_ids or [])
        self.interval = interval_s
        self.clock = clock
        self.started = clock()
        self.state = 'running'
        self.last_event = 'run started'
        self.running: list[str] = []
        self.todo_total = tasks_total - tasks_done
        self._finished_wall: list[float] = []
        self._t0: dict[str, float] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._warned = False
        self._thread: threading.Thread | None = None
        self.write_errors = 0

    # -- lifecycle ----------------------------------------------------------------------------------
    def start(self) -> StatusWriter:
        if self.enabled and self._thread is None:
            self._thread = threading.Thread(target=self._loop, name='status-writer', daemon=True)
            self._thread.start()
            self._wake.set()
        return self

    def close(self, state: str = 'done', last_event: str | None = None, timeout: float = 5.0) -> None:
        """Final write (state done|failed), then stop the thread. Never raises, waits at most `timeout` s."""
        if not self.enabled:
            return
        with self._lock:
            self.state = state
            self.running = []
            if last_event:
                self.last_event = last_event[:LAST_EVENT_MAX]
        self._stop = True
        self._wake.set()
        t = self._thread
        if t is not None:
            t.join(timeout)
        else:
            self._write_safely()

    # -- events -------------------------------------------------------------------------------------
    def task_started(self, iid: str) -> None:
        with self._lock:
            if iid not in self.running:
                self.running.append(iid)
            self._t0[iid] = self.clock()
            self.last_event = f'started {iid}'[:LAST_EVENT_MAX]
        self._wake.set()

    def task_finished(self, iid: str, status: str | None, termination: str | None = None,
                      empty_patch: bool = False) -> None:
        with self._lock:
            if iid in self.running:
                self.running.remove(iid)
            t0 = self._t0.pop(iid, None)
            if t0 is not None:
                self._finished_wall.append(max(0.0, self.clock() - t0))
            self.done += 1
            if empty_patch:
                self.empty += 1
            what = ' '.join(str(x) for x in (status, termination) if x)
            self.last_event = f'finished {iid}: {what}'[:LAST_EVENT_MAX]
        self._wake.set()

    # -- snapshot -----------------------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        now = self.clock()
        with self._lock:
            running = list(self.running)
            walls = list(self._finished_wall)
            done, empty, state, last = self.done, self.empty, self.state, self.last_event
        eta = None
        remaining = max(0, self.total - done)
        if state == 'running' and walls:
            eta = int(remaining * (sum(walls) / len(walls)) / self.concurrency)
        elif state != 'running':
            eta = 0 if state == 'done' else None
        return {
            'run': self.run,
            'submission': self.submission,
            'endpoint_kind': self.kind,
            'host': self.host,
            'started': iso(self.started),
            'updated': iso(now),
            'tasks_total': self.total,
            'tasks_done': done,
            'tasks_running': running,
            'resolved_so_far': self._resolved(),
            'empty_so_far': empty,
            'current_task': running[0] if running else None,
            'last_event': last,
            'eta_s': eta,
            'state': state,
        }

    def _resolved(self) -> int | None:
        """Count of resolved tasks if a grades file already exists in the run dir (grading is normally a later step)."""
        if not self.grades_path:
            return None
        try:
            if not self.grades_path.exists():
                return None
            from harness.runner import report
            grades = report.load_grades(self.grades_path)
            return sum(1 for k, v in grades.items() if v and (not self.task_ids or k in self.task_ids))
        except Exception:
            return None

    # -- writing ------------------------------------------------------------------------------------
    def _loop(self) -> None:
        last = 0.0
        while True:
            self._wake.wait(self.interval)
            self._wake.clear()
            stop = self._stop
            # events write at once; the heartbeat (timeout wake-up) is at most one write per interval anyway
            if stop or self.clock() - last >= 0.5:
                self._write_safely()
                last = self.clock()
            if stop:
                return

    def _write_safely(self) -> None:
        try:
            self.write_once()
        except Exception as e:  # noqa: BLE001 - the share may vanish at any moment; the run must not care
            self.write_errors += 1
            if not self._warned:
                self._warned = True
                print(f'[runner] status-dir unavailable ({type(e).__name__}: {getattr(e, "strerror", None) or e}); '
                      'continuing without live status', file=sys.stderr, flush=True)

    def write_once(self) -> None:
        """Write DIR/<run>.json then DIR/index.json. Raises on any I/O problem (callers swallow it)."""
        d = self.dir
        assert d is not None
        d.mkdir(exist_ok=True)
        snap = self.snapshot()
        _atomic_json(d / f'{self.run}.json', snap)
        _atomic_json(d / INDEX_NAME, build_index(d, snap, self.clock()))


def _atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=1)
            f.flush()
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def build_index(d: Path, mine: dict[str, Any], now: float) -> dict[str, Any]:
    """Recent runs in DIR (newest first), always including `mine`. Unreadable files are skipped."""
    rows: dict[str, dict[str, Any]] = {}
    for p in d.glob('*.json'):
        if p.name == INDEX_NAME or p.name.startswith('.'):
            continue
        try:
            j = json.loads(p.read_text(encoding='utf-8'))
            if not isinstance(j, dict) or not isinstance(j.get('run'), str):
                continue
            rows[j['run']] = j
        except Exception:
            continue
    rows[mine['run']] = mine
    keep = []
    for j in rows.values():
        try:
            age = now - dt.datetime.fromisoformat(str(j.get('updated')).replace('Z', '+00:00')).timestamp()
        except Exception:
            continue
        if age <= INDEX_MAX_AGE_S or j is mine:
            keep.append({k: j.get(k) for k in ('run', 'state', 'updated', 'tasks_done', 'tasks_total')})
    keep.sort(key=lambda r: str(r.get('updated')), reverse=True)
    return {'updated': iso(now), 'runs': keep[:INDEX_MAX_RUNS]}
