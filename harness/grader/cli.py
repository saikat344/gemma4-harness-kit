"""Scorer-equivalent grader: grade predictions (instance_id -> unified diff) with the official
``swegemma.harness.verification.verify_task`` in the ``swebench-sandbox`` Docker image.

    python -m harness.grader --predictions preds.jsonl --out runs/<name>/grades.jsonl --workers 2
    python -m harness.grader --gold  --out runs/validation-mac/gold/grades.jsonl
    python -m harness.grader --empty --out runs/validation-mac/empty/grades.jsonl

Outputs (next to --out): grades.jsonl (one row per task, appended as tasks finish; --resume
skips ids already present), summary.json, logs/<instance_id>.log, grader.log.
See docs/grader.md.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from harness.common import paths
from harness.common.tasks import load_predictions

log = logging.getLogger('harness.grader')


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog='python -m harness.grader', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--predictions', type=Path, help='.jsonl/.json/.csv/.parquet (id|instance_id, prediction|patch|model_patch)')
    src.add_argument('--gold', action='store_true', help='grade each task\'s gold `patch` (soundness check: should pass)')
    src.add_argument('--empty', action='store_true', help='grade an empty patch, running the tests anyway (soundness check: should fail)')
    p.add_argument('--tasks', type=Path, default=None, help='tasks.jsonl (default $GEMMA4_DATA_DIR/tasks.jsonl)')
    p.add_argument('--snapshots-dir', type=Path, default=None, help='default $GEMMA4_SNAPSHOTS_DIR or $GEMMA4_DATA_DIR/snapshots')
    p.add_argument('--wheels-dir', type=Path, default=None, help='competition wheels/ (default $GEMMA4_WHEELS_DIR or $GEMMA4_DATA_DIR/wheels)')
    p.add_argument('--no-wheels', action='store_true',
                   help='inject no wheels/ (task sets whose image carries its own environment; '
                        'what swegemma does when no wheels dir resolves). Run from a dir without ./wheels')
    p.add_argument('--sandbox-dir', type=Path, default=None, help='dir holding setup.py (default $GEMMA4_DATA_DIR/sandbox)')
    p.add_argument('--image', default=None, help=f'sandbox image (default $GEMMA4_SANDBOX_IMAGE or {paths.DEFAULT_IMAGE})')
    p.add_argument('--out', type=Path, required=True, help='grades.jsonl path; logs go next to it')
    p.add_argument('--workers', type=int, default=2, help='concurrent tasks (each uses a 4 GiB / 2 vCPU container)')
    p.add_argument('--task-ids', nargs='+', default=None, help='restrict to these instance ids')
    p.add_argument('--repos', nargs='+', default=None, help='restrict to these repo short names (fastapi rich ...)')
    p.add_argument('--include-missing-snapshots', action='store_true',
                   help='count tasks whose snapshot is missing as failures instead of skipping them')
    p.add_argument('--run-empty-patches', action='store_true',
                   help='run tests for empty predictions (the scorer skips Phase 2 for an empty agent patch -> unresolved)')
    p.add_argument('--verification-timeout', type=int, default=300, help='pytest timeout in seconds (scorer: 300)')
    p.add_argument('--task-timeout', type=int, default=3600, help='grader-side wall clock cap per task (s)')
    p.add_argument('--network', default='none',
                   help="container network_mode (scorer: none). A pre-created offline network that DROPs instead of "
                        "rejecting makes requests' TestTimeout tests time out as upstream expects; see docs/grader.md")
    p.add_argument('--no-env-fixes', action='store_true', help='disable harness/sandbox/task_env_fixes.json (pure official env)')
    p.add_argument('--resume', action='store_true', help='skip ids already in --out')
    p.add_argument('-v', '--verbose', action='store_true')
    return p.parse_args(argv)


# --------------------------------------------------------------------------- helpers

_SUMMARY_RE = re.compile(r'^=*\s*((?:\d+ \w+(?:, )?)+) in [\d.]+s', re.M)


def _pytest_counts(stdout: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    lines = [l for l in stdout.splitlines() if re.search(r'\d+ (passed|failed|error|errors|skipped|xfailed|xpassed|deselected|warnings?)\b', l)]
    if lines:
        for n, kind in re.findall(r'(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b', lines[-1]):
            counts['error' if kind.startswith('error') else kind] = int(n)
    return counts


def _failed_nodes(stdout: str) -> list[str]:
    out = []
    for line in stdout.splitlines():
        m = re.match(r'^(FAILED|ERROR) (\S+)', line)
        if m:
            out.append(f'{m.group(1)} {m.group(2)}')
    return out


def _classify(res: Any, stdout: str) -> str:
    if res.resolved:
        return 'none'
    err = (getattr(res, 'error_message', None) or '')
    code = res.test_exit_code
    if err.startswith('Failed to apply agent patch'):
        return 'patch_apply_failed'
    if err.startswith('Failed to apply test_patch'):
        return 'test_patch_apply_failed'
    if err.startswith('Evaluation error'):
        return 'eval_error'
    if err.startswith('Missing test specification'):
        return 'missing_test_spec'
    if code in (124, 137) or 'Command timed out' in stdout:
        return 'timeout'
    if code == 0:
        if err.startswith('Required test node'):
            return 'required_test_not_passed'
        return 'junit_invalid'
    if code == 5:
        return 'no_tests_collected'
    if code in (2, 3, 4) or 'error' in _pytest_counts(stdout) and 'passed' not in _pytest_counts(stdout):
        return 'collection_or_usage_error'
    if code == 1:
        return 'tests_failed'
    return f'exit_{code}'


def _wheels_listing_sha256(wheels: Path) -> str:
    """Hash of the wheels/ listing (file name + size per .whl)."""
    listing = '\n'.join(sorted(f'{p.name}:{p.stat().st_size}' for p in wheels.glob('*.whl')))
    return hashlib.sha256(listing.encode()).hexdigest()


def _wheels_cache_dir(wheels: Path) -> Path:
    """swegemma caches unpacked wheel tars in $TMPDIR/swegemma_sp_cache_v8 keyed only by name;
    key our temp dir by the wheels listing so a changed wheels/ never reuses a stale tar."""
    h = _wheels_listing_sha256(wheels)[:12]
    d = paths.cache_dir() / f'swegemma-tmp-{h}'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _check_image(image: str) -> str:
    import docker

    try:
        img = docker.from_env().images.get(image)
    except Exception as e:
        raise SystemExit(f'sandbox image {image!r} not found ({e}); build it with harness/sandbox/build.sh')
    arch = img.attrs.get('Architecture', '?')
    if arch != 'amd64':
        log.warning('image %s is %s, not amd64: fastapi tasks need the x86-only pydantic_core wheel', image, arch)
    return arch


# Same file as harness.common.envfix.FIXES_PATH (that module imports swegemma, so it is not imported at load time).
FIXES_PATH = paths.REPO_ROOT / 'harness' / 'sandbox' / 'task_env_fixes.json'


def patch_sha256(patch: str) -> str:
    return hashlib.sha256(patch.encode('utf-8')).hexdigest()


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Bump when a grader change alters what `passed` means for the same inputs: resumed rows from an
# older grader are then re-graded. 2: missing JUnit XML is fatal (F1), failed env repairs are setup errors (F7).
GRADER_SEMANTICS = 2


def grade_key(*, mode: str, image_id: str | None, network: str, verification_timeout: int, task_timeout: int,
              env_fixes: bool, run_empty_patches: bool, setup_py: Path | None, wheels_dir: Path | None) -> str:
    """Hash of the grading conditions shared by every task (audit F5, 2026-11-05).

    It covers what produced a verdict, not where it lives: the image id, network, pytest and outer
    per-task timeouts, the env-fix table's content, the sandbox setup.py content, and the wheels/
    listing (name + size of each .whl; ``wheels_dir=None`` = --no-wheels). Per-task inputs (test
    spec, snapshot) are in ``task_sha256``. A resumed row is reused only when its ``grade_key``,
    ``task_sha256`` and ``patch_sha256`` all match (``reusable_rows``).
    """
    fixes = FIXES_PATH.read_bytes() if env_fixes and FIXES_PATH.is_file() else b''
    blob = {'mode': mode, 'image_id': image_id, 'network': network, 'verification_timeout': verification_timeout,
            'task_timeout': task_timeout, 'env_fixes': env_fixes, 'env_fixes_sha256': hashlib.sha256(fixes).hexdigest(),
            'run_empty_patches': run_empty_patches, 'setup_py_sha256': _file_sha256(setup_py),
            'wheels_listing_sha256': _wheels_listing_sha256(wheels_dir) if wheels_dir is not None else None,
            'grader_semantics': GRADER_SEMANTICS}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]


def task_sha256(task: Any, snapshot: Path) -> str:
    """Per-task digest of the inputs that decide this task's verdict (audit F5, 2026-11-05).

    Task definition: instance id, repo (selects setup.py and env-fix entries), base commit, test_patch,
    FAIL_TO_PASS / PASS_TO_PASS, and the test files the grader targets. Snapshot identity is
    size + mtime_ns of ``<snapshots>/<id>.tgz`` rather than a content hash: snapshots run to ~100 MB
    each, so hashing them on every resume is too slow, and a re-downloaded identical file merely
    costs a re-grade. Digesting per task means adding or editing other rows of tasks.jsonl does not
    invalidate this task's cached verdict.
    """
    try:
        st = snapshot.stat()
        snap = {'size': st.st_size, 'mtime_ns': st.st_mtime_ns}
    except OSError:
        snap = None
    blob = {'instance_id': task.instance_id, 'repo': task.repo, 'base_commit': getattr(task, 'base_commit', ''),
            'test_patch': getattr(task, 'test_patch', ''), 'FAIL_TO_PASS': list(getattr(task, 'FAIL_TO_PASS', ())),
            'PASS_TO_PASS': list(getattr(task, 'PASS_TO_PASS', ())),
            'test_files': list(getattr(task, 'test_files', None) or ()), 'snapshot': snap}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]


def reusable_rows(done: dict[str, dict[str, Any]], preds: dict[str, str], key: str,
                  task_keys: dict[str, str]) -> tuple[dict[str, dict[str, Any]], int]:
    """The resumed rows still valid for these predictions and conditions, and how many were dropped.

    A row is kept only if its task is predicted now and its ``grade_key`` (global conditions),
    ``task_sha256`` (that task's spec + snapshot, from ``task_keys``) and ``patch_sha256`` all match.
    Rows missing any of these (written before the check, or for ids no longer in tasks.jsonl) are
    dropped, i.e. re-graded.
    """
    keep = {iid: r for iid, r in done.items()
            if iid in preds and iid in task_keys and r.get('grade_key') == key
            and r.get('task_sha256') == task_keys[iid] and r.get('patch_sha256') == patch_sha256(preds[iid])}
    return keep, len(done) - len(keep)


def _provenance(image: str) -> dict[str, Any]:
    """Image id and harness git revision, recorded in summary.json so a result names what produced it."""
    import subprocess

    out: dict[str, Any] = {'image_id': None, 'harness_rev': None, 'harness_dirty': None}
    try:
        import docker
        out['image_id'] = docker.from_env().images.get(image).id
    except Exception:
        pass
    repo = Path(__file__).resolve().parents[2]
    try:
        out['harness_rev'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True,
                                            timeout=10).stdout.strip() or None
        st = subprocess.run(['git', 'status', '--porcelain', '--', 'harness'], cwd=repo, capture_output=True,
                            text=True, timeout=10)
        out['harness_dirty'] = bool(st.stdout.strip()) if st.returncode == 0 else None
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- grading


async def _grade_one(task: Any, patch: str, *, mode: str, args: argparse.Namespace, cfg: Any,
                     image: str, snapshots: Path, logs_dir: Path) -> dict[str, Any]:
    from swegemma.harness.container_setup import ContainerSetupError
    from swegemma.harness.verification import _extract_test_functions_from_patch, strip_protected_files_from_patch, verify_task
    from swegemma.sandbox import ContainerConfig

    from harness.common.envfix import FixingContainerManager, fix_commands

    iid = task.instance_id
    t0 = time.perf_counter()
    snap = paths.snapshot_path(iid, snapshots)
    targets = task.test_files
    required = list(task.FAIL_TO_PASS) + list(task.PASS_TO_PASS) or _extract_test_functions_from_patch(task.test_patch)
    stripped = strip_protected_files_from_patch(patch, extra_protected_paths=targets) if patch.strip() else ''
    row: dict[str, Any] = {
        'instance_id': iid, 'repo': task.repo, 'mode': mode, 'passed': False,
        'patch_lines': len(patch.splitlines()), 'patch_lines_after_strip': len(stripped.splitlines()),
        'patch_empty': not patch.strip(), 'target_test_files': targets, 'required_tests': required,
    }
    fixes = [] if args.no_env_fixes else fix_commands(iid, task.repo)
    row['env_fixes'] = len(fixes)
    log_lines = [f'instance_id: {iid}', f'repo: {task.repo}', f'mode: {mode}', f'image: {image}',
                 f'snapshot: {snap}', f'target_test_files: {targets}', f'required_tests: {required}',
                 f'patch lines raw/after-strip: {row["patch_lines"]}/{row["patch_lines_after_strip"]}',
                 f'env_fix commands: {fixes}']
    res = None
    mgr = None
    try:
        if not snap.exists():
            row.update(error='snapshot missing', error_class='missing_snapshot', test_exit_code=-1)
        elif not patch.strip() and mode == 'predictions' and not args.run_empty_patches:
            # Scorer: an empty agent patch means agent_error and no Phase 2 -> unresolved.
            row.update(error='empty patch (scorer skips verification)', error_class='empty_patch', test_exit_code=-1)
        else:
            mgr = FixingContainerManager(ContainerConfig(image=image, network_mode=args.network), instance_id=iid, repo=task.repo, commands=fixes)
            res = await asyncio.wait_for(
                verify_task(mgr, cfg, task, snap, agent_patch=patch, start_time=t0),
                timeout=args.task_timeout)
            stdout = (res.test_output or '').split('\nSTDERR:\n', 1)[0]
            err = getattr(res, 'error_message', None)
            if not res.resolved and not err:
                tail = [l for l in stdout.splitlines() if l.strip() and l.strip() != 'STDOUT:']
                summ = [l for l in tail if re.search(r' in [\d.]+s\b', l)]
                err = ((summ or tail)[-1].strip('= ') if tail else f'pytest exit {res.test_exit_code}')[:300]
            row.update(passed=bool(res.resolved), test_exit_code=res.test_exit_code,
                       error=err, error_class=_classify(res, stdout),
                       pytest_counts=_pytest_counts(stdout), failed_tests=_failed_nodes(stdout)[:50])
    except asyncio.TimeoutError:
        row.update(error=f'grader task timeout {args.task_timeout}s', error_class='grader_timeout', test_exit_code=-1)
    except ContainerSetupError as e:
        row.update(error=f'ContainerSetupError: {e}', error_class='setup_error', test_exit_code=-1)
    except Exception as e:  # never let one task kill the run
        log.exception('grading %s crashed', iid)
        row.update(error=f'{type(e).__name__}: {e}', error_class='grader_crash', test_exit_code=-1)
    finally:
        if mgr is not None:
            # Also on a setup error, e.g. a failed env repair (EnvFixError is a ContainerSetupError).
            log_lines += mgr.fix_log
            row['env_fixes_applied'] = mgr.applied_fixes
            row['env_fixes_failed'] = [r['command'] for r in mgr.fix_results if not r['ok']]
            try:
                await asyncio.to_thread(mgr.cleanup_all)
            except Exception:
                pass
    row['duration_seconds'] = round(time.perf_counter() - t0, 1)
    log_lines += [f'passed: {row["passed"]}', f'error_class: {row.get("error_class")}', f'error: {row.get("error")}',
                  f'duration_seconds: {row["duration_seconds"]}', '=' * 30 + ' test output',
                  (res.test_output if res is not None else '')]
    (logs_dir / f'{iid}.log').write_text('\n'.join(str(x) for x in log_lines) + '\n', encoding='utf-8')
    return row


def _summary(rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    by_repo: dict[str, dict[str, int]] = defaultdict(lambda: {'total': 0, 'resolved': 0})
    for r in rows:
        k = r['repo'].rsplit('/', 1)[-1]
        by_repo[k]['total'] += 1
        by_repo[k]['resolved'] += int(r['passed'])
    resolved = sum(int(r['passed']) for r in rows)
    total = len(rows)
    return {**meta, 'total': total, 'resolved': resolved,
            'resolution_rate': round(resolved / total, 4) if total else 0.0,
            'by_repo': dict(sorted(by_repo.items())),
            'error_classes': dict(Counter(r.get('error_class') or 'none' for r in rows)),
            'failed_ids': sorted(r['instance_id'] for r in rows if not r['passed'])}


async def _run(args: argparse.Namespace) -> int:
    tasks_path = (args.tasks or paths.tasks_path()).resolve()
    snapshots = (args.snapshots_dir or paths.snapshots_dir()).resolve()
    wheels = None if args.no_wheels else (args.wheels_dir or paths.wheels_dir()).resolve()
    sandbox_dir = (args.sandbox_dir or paths.data_dir() / 'sandbox').resolve()
    image = args.image or paths.sandbox_image()
    out = args.out.resolve()
    out_dir = out.parent
    logs_dir = out_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not (sandbox_dir / 'setup.py').is_file():
        raise SystemExit(f'{sandbox_dir}/setup.py not found (competition sandbox/setup.py)')
    if wheels is not None and not any(wheels.glob('*.whl')):
        raise SystemExit(f'no wheels in {wheels} (competition wheels/ directory); --no-wheels to inject none')
    # Must be set before swegemma resolves setup.py / the wheel tar cache.
    os.environ['KAGGLE_SANDBOX_DIR'] = str(sandbox_dir)
    if wheels is not None:
        tempfile.tempdir = str(_wheels_cache_dir(wheels))

    from adk_submission import ModelRegistry
    from swegemma.config import EvalConfig
    from swegemma.models import load_tasks

    arch = _check_image(image)
    tasks = load_tasks(tasks_path)
    if args.task_ids:
        want = set(args.task_ids)
        tasks = [t for t in tasks if t.instance_id in want]
        missing_ids = want - {t.instance_id for t in tasks}
        if missing_ids:
            log.warning('ids not in %s: %s', tasks_path, sorted(missing_ids))
    if args.repos:
        tasks = [t for t in tasks if t.repo.rsplit('/', 1)[-1] in set(args.repos)]

    if args.gold:
        mode, preds = 'gold', {t.instance_id: t.patch for t in tasks}
    elif args.empty:
        mode, preds = 'empty', {t.instance_id: '' for t in tasks}
        args.run_empty_patches = True
    else:
        mode, preds = 'predictions', load_predictions(args.predictions)
        unknown = set(preds) - {t.instance_id for t in load_tasks(tasks_path)}
        if unknown:
            log.warning('%d prediction ids not in tasks.jsonl (ignored): %s', len(unknown), sorted(unknown)[:10])
        tasks = [t for t in tasks if t.instance_id in preds]

    if not args.include_missing_snapshots:
        skipped = [t.instance_id for t in tasks if not paths.snapshot_path(t.instance_id, snapshots).exists()]
        if skipped:
            log.warning('skipping %d tasks without a snapshot in %s', len(skipped), snapshots)
        tasks = [t for t in tasks if t.instance_id not in set(skipped)]
    else:
        skipped = []

    done: dict[str, dict[str, Any]] = {}
    if args.resume and out.exists():
        for line in out.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                done[r['instance_id']] = r
    elif out.exists():
        out.unlink()
    prov = _provenance(image)
    key = grade_key(mode=mode, image_id=prov['image_id'], network=args.network,
                    verification_timeout=args.verification_timeout, task_timeout=args.task_timeout,
                    env_fixes=not args.no_env_fixes, run_empty_patches=bool(args.run_empty_patches),
                    setup_py=sandbox_dir / 'setup.py', wheels_dir=wheels)
    # Every task in the file, not just this selection, so rows of tasks outside --task-ids/--repos
    # are still validated and kept when the output is rewritten.
    task_keys = {t.instance_id: task_sha256(t, paths.snapshot_path(t.instance_id, snapshots))
                 for t in load_tasks(tasks_path)}
    done, stale = reusable_rows(done, preds, key, task_keys)
    if stale:
        log.warning('--resume: re-grading %d rows whose patch, task spec/snapshot or grading conditions changed '
                    '(or that predate the check)',
                    stale)
    todo = [t for t in tasks if t.instance_id not in done]
    log.info('grading %d tasks (%d already done) mode=%s image=%s (%s) workers=%d', len(todo), len(done), mode, image, arch, args.workers)

    cfg = EvalConfig(tasks_path=tasks_path, snapshots_dir=snapshots, results_dir=out_dir, submission_dir=out_dir,
                     models=ModelRegistry(), image=image, sandbox='docker', wheels_dir=wheels,
                     verification_timeout_seconds=args.verification_timeout, display_mode='quiet')

    sem = asyncio.Semaphore(max(1, args.workers))
    rows = list(done.values())
    lock = asyncio.Lock()
    n_total = len(todo)

    async def worker(t: Any) -> None:
        async with sem:
            row = await _grade_one(t, preds.get(t.instance_id, ''), mode=mode, args=args, cfg=cfg,
                                   image=image, snapshots=snapshots, logs_dir=logs_dir)
            row.update(patch_sha256=patch_sha256(preds.get(t.instance_id, '')), grade_key=key,
                       task_sha256=task_keys[t.instance_id])
        async with lock:
            rows.append(row)
            with open(out, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
            print(f'[{len(rows) - len(done)}/{n_total}] {row["instance_id"]}: '
                  f'{"PASS" if row["passed"] else "FAIL"} ({row.get("error_class")}) {row["duration_seconds"]}s', flush=True)

    t_start = time.perf_counter()
    await asyncio.gather(*(worker(t) for t in todo))
    order = {t.instance_id: i for i, t in enumerate(tasks)}
    rows.sort(key=lambda r: order.get(r['instance_id'], 1 << 30))
    with open(out, 'w', encoding='utf-8') as f:  # rewrite in task order
        for r in rows:
            f.write(json.dumps(r) + '\n')
    meta = {'mode': mode, 'image': image, 'image_arch': arch, 'network': args.network, 'tasks_path': str(tasks_path),
            'predictions': str(args.predictions) if args.predictions else None, 'env_fixes': not args.no_env_fixes,
            'skipped_missing_snapshot': skipped, 'wall_seconds': round(time.perf_counter() - t_start, 1),
            'workers': args.workers, 'grade_key': key, **prov}
    summ = _summary(rows, meta)
    (out_dir / 'summary.json').write_text(json.dumps(summ, indent=2) + '\n', encoding='utf-8')
    print(f'RESOLVED {summ["resolved"]}/{summ["total"]} ({100 * summ["resolution_rate"]:.1f}%) mode={mode} '
          f'by_repo={ {k: f"{v["resolved"]}/{v["total"]}" for k, v in summ["by_repo"].items()} } '
          f'errors={summ["error_classes"]}', flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out.resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.FileHandler(out_dir / 'grader.log', encoding='utf-8')]
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    handlers.append(sh)
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger('harness.grader').setLevel(logging.INFO)
    sh.addFilter(lambda r: r.levelno >= logging.WARNING or r.name.startswith('harness'))
    return asyncio.run(_run(args))
