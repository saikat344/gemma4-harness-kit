"""Orchestrator: `python -m harness.runner run|report|export-sft|checkpoints ...` (run is the default subcommand).

Layout of runs/<name>/:
  run.json                  manifest: args, budgets (+ overrides), endpoint models, submission hash, versions
  submission/               frozen copy of the submission that was run
  tasks/<id>/spec.json      worker input
  tasks/<id>/result.json    per-task metrics (status done|infra_error|killed|worker_timeout)
  tasks/<id>/patch.diff     raw agent patch (after protected-file stripping)
  tasks/<id>/llm_calls.jsonl  every LLM request/response + budget state (SFT source)
  tasks/<id>/compactions.jsonl  ADK compaction events (summary text, window, budget state)
  tasks/<id>/trace_legacy.json  ADK event trace (thinking, tool calls/results, usage, compaction, nudges)
  tasks/<id>/traces/trace_<id>.json  same trace in the harness's ATIF v1.7 format
  tasks/<id>/logs/<id>.log  harness rich log;  tasks/<id>/worker.log  worker stdout/stderr
  tasks/<id>/checkpoints.jsonl + checkpoints/<t>.patch  with --checkpoints: graded patch at each time cap t
  predictions.jsonl         {instance_id, patch} (patch == "NO_PATCH" when empty) + {id, prediction} aliases
  summary.json / summary.md run summary (see report.py)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from harness.runner import checkpoints, core, report, status

TERMINAL_OK = {'done'}


def _git_rev() -> str | None:
    try:
        rev = subprocess.run(['git', '-C', str(core.REPO_ROOT), 'rev-parse', 'HEAD'], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(core.REPO_ROOT), 'status', '--porcelain'], capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return rev + ('-dirty' if dirty else '') if rev else None
    except Exception:
        return None


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for pkg in ('swegemma', 'adk-submission', 'adk-eval-core', 'google-adk', 'google-genai', 'litellm', 'docker'):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = 'missing'
    out['python'] = platform.python_version()
    return out


# run.json fields that change task results. Resuming a run dir with any of them different would put tasks run
# under the old conditions next to tasks run under the new ones, while run.json only names the new ones.
RESULT_CONDITIONS = ('submission_sha256', 'budgets', 'compaction', 'context_cache', 'served_model', 'image.id',
                     'data.tasks_path', 'data.task_sha256', 'data.snapshots_dir', 'data.wheels_dir', 'env_fixes', 'timing_scale',
                     'checkpoints')
# Recorded and reported on resume, but not refused: they can shift timing without defining the experiment.
SOFT_CONDITIONS = ('git', 'versions', 'concurrency', 'endpoint')


_MISSING = object()  # a key absent from a manifest, as opposed to one stored as null (e.g. --no-compaction)


def _get_path(d: dict[str, Any], key: str, default: Any = None) -> Any:
    for part in key.split('.'):
        if not isinstance(d, dict) or part not in d:
            return default
        d = d[part]
    return d


def condition_diff(prev: dict[str, Any], cur: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(result conditions that differ, soft conditions that differ) between two run.json manifests.

    A key absent from an older manifest is not compared, so run dirs from before this check still resume. A key
    stored as null (--no-compaction, no wheels dir) is a real setting and is compared like any other value.
    """
    def differs(k: str) -> bool:
        a, b = _get_path(prev, k, _MISSING), _get_path(cur, k, _MISSING)
        if a is _MISSING:
            return False
        if k == 'data.task_sha256' and isinstance(a, dict) and isinstance(b, dict):
            # only tasks present in both sessions: a resume may select a different subset
            return any(a[i] != b[i] for i in a.keys() & b.keys())
        return json.dumps(None if b is _MISSING else b, sort_keys=True, default=str) != json.dumps(a, sort_keys=True,
                                                                                                default=str)
    return [k for k in RESULT_CONDITIONS if differs(k)], [k for k in SOFT_CONDITIONS if differs(k)]


def conditions_sha256(manifest: dict[str, Any]) -> str:
    blob = {k: _get_path(manifest, k) for k in RESULT_CONDITIONS}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()


def _image_info(image: str) -> dict[str, Any]:
    try:
        import docker
        img = docker.from_env().images.get(image)
        return {'image': image, 'id': img.id, 'arch': img.attrs.get('Architecture'), 'created': img.attrs.get('Created')}
    except Exception as e:
        return {'image': image, 'error': f'{type(e).__name__}: {e}'}


def _remove_containers(task_dir: Path) -> None:
    path = task_dir / 'containers.txt'
    if not path.exists():
        return
    try:
        import docker
        client = docker.from_env()
    except Exception:
        return
    for cid in path.read_text().split():
        try:
            c = client.containers.get(cid)
            c.remove(force=True)
        except Exception:
            pass


def _redact_argv(argv: list[str]) -> list[str]:
    """Never persist the endpoint API key (run.json records argv)."""
    out, hide = [], False
    for a in argv:
        out.append('***' if hide else ('--api-key=***' if a.startswith('--api-key=') else a))
        hide = a == '--api-key'
    return out


async def _run_worker(spec_path: Path, task_dir: Path, hard_timeout_s: float, grace_s: float = 30.0,
                      api_key: str = 'EMPTY') -> None:
    log = open(task_dir / 'worker.log', 'ab')  # noqa: SIM115
    env = dict(os.environ)
    env['GEMMA4_ENDPOINT_API_KEY'] = api_key  # handed to the worker via env, not spec.json (which stays on disk)
    env['PYTHONPATH'] = str(core.REPO_ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    proc = await asyncio.create_subprocess_exec(
        sys.executable, '-m', 'harness.runner.worker', str(spec_path),
        cwd=str(core.REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, env=env,
        start_new_session=True,
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=hard_timeout_s)
    except (TimeoutError, asyncio.CancelledError) as exc:
        cancelled = isinstance(exc, asyncio.CancelledError)
        with _suppress():
            proc.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=grace_s)
        except TimeoutError:
            with _suppress():
                os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
        _remove_containers(task_dir)
        res = task_dir / 'result.json'
        prev = json.loads(res.read_text()) if res.exists() else {}
        if prev.get('status') != 'done':
            prev.update({'status': 'killed' if cancelled else 'worker_timeout',
                         'instance_id': json.loads(spec_path.read_text())['instance_id'],
                         'error': f'orchestrator stopped worker after {hard_timeout_s:.0f}s' if not cancelled
                         else 'run interrupted'})
            res.write_text(json.dumps(prev, indent=2))
        if cancelled:
            raise
    finally:
        log.close()
    if not (task_dir / 'result.json').exists():
        (task_dir / 'result.json').write_text(json.dumps({
            'status': 'infra_error', 'instance_id': json.loads(spec_path.read_text())['instance_id'],
            'error': f'worker exited with code {proc.returncode} without result.json (see worker.log)'}, indent=2))
        _remove_containers(task_dir)


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return exc[0] is not None and issubclass(exc[0], (ProcessLookupError, PermissionError, OSError))


def _needs_run(task_dir: Path, retry_statuses: set[str]) -> bool:
    res = task_dir / 'result.json'
    if not res.exists():
        return True
    try:
        status = json.loads(res.read_text()).get('status')
    except Exception:
        return True
    return status in retry_statuses


def _reset_task_dir(task_dir: Path) -> None:
    import shutil
    if task_dir.exists():
        attempts = task_dir.parent.parent / 'attempts'
        attempts.mkdir(exist_ok=True)
        dest = attempts / f'{task_dir.name}.{int(time.time())}'
        shutil.move(str(task_dir), str(dest))
    task_dir.mkdir(parents=True)


def cmd_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.out).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / 'run.json'
    prev_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    data_dir = Path(args.data_dir).resolve()
    tasks_path = Path(args.tasks).resolve() if args.tasks else data_dir / 'tasks.jsonl'
    data = {
        'tasks_path': str(tasks_path),
        'snapshots_dir': str(Path(args.snapshots_dir).resolve() if args.snapshots_dir else data_dir / 'snapshots'),
        'graph_dir': str(data_dir / 'graphs'),
        'embeddings_dir': str(data_dir / 'embeddings'),
        'wheels_dir': str(Path(args.wheels_dir).resolve()) if args.wheels_dir else (
            str(data_dir / 'wheels') if (data_dir / 'wheels').is_dir() else None),
    }

    agent_dir = core.prepare_submission(Path(args.submission), run_dir)
    sub_info = core.validate_submission(agent_dir)
    overrides = dict(kv.split('=', 1) for kv in (args.override or []))
    budgets = core.load_budgets(agent_dir, overrides)

    rows = core.load_task_rows(tasks_path)
    all_ids = [r['instance_id'] for r in rows]
    row_of = {r['instance_id']: r for r in rows}
    repo_of = {r['instance_id']: r['repo'] for r in rows}
    ids = core.resolve_task_ids(args.task_ids, all_ids)
    if args.limit:
        ids = ids[: args.limit]
    # Content identity of the selected tasks (audit F5): a changed test_patch at the same path is a different run.
    data['task_sha256'] = {i: hashlib.sha256(json.dumps(row_of[i], sort_keys=True, default=str).encode()).hexdigest()[:16]
                           for i in ids}
    ckpts = checkpoints.parse(args.checkpoints)
    if ckpts and budgets.max_time_minutes and max(ckpts) >= budgets.max_time_minutes:
        print(f'WARNING: checkpoints {ckpts} reach max_time_minutes={budgets.max_time_minutes}; a cap at or above '
              'the budget is not observable (the run itself stops there)')

    ep = core.probe_endpoint(args.endpoint, args.api_key)
    served = args.served_model or (ep['base_ids'][0] if ep['base_ids'] else None)
    if not served:
        print(f'ERROR: no base model served at {args.endpoint}: {ep["models"]}')
        return 2
    missing_loras = [a for a in sub_info['adapters'] if a not in ep['lora_ids']]
    if missing_loras and not args.allow_missing_loras:
        print(f'ERROR: submission adapters {missing_loras} are not served by the endpoint '
              f'(served LoRAs: {ep["lora_ids"]}). Restart vLLM with harness/serve --submission <dir>, '
              'or pass --allow-missing-loras.')
        return 2
    if ep.get('max_model_len') and ep['max_model_len'] != 32768:
        print(f'WARNING: endpoint max_model_len={ep["max_model_len"]} (scorer: 32768)')

    deviations = []
    if budgets.overrides:
        deviations.append(f'budget overrides: {budgets.overrides}')
    if args.no_compaction:
        deviations.append('compaction + context cache disabled (--no-compaction)')
    if args.image != 'swebench-sandbox:latest':
        deviations.append(f'sandbox image {args.image} (scorer: swebench-sandbox:latest, Kaggle-built)')
    if missing_loras:
        deviations.append(f'adapters not served: {missing_loras}')
    if ckpts:
        deviations.append(f'time checkpoints at {ckpts} min (observer; one git diff exec per crossing)')
    if not args.no_env_fixes:
        deviations.append('per-task test-env fixes from harness/sandbox/task_env_fixes.json (same as grader; '
                          '--no-env-fixes for the pure official env)')

    manifest = {
        'name': run_dir.name,
        'created_at': prev_manifest['created_at'] if prev_manifest else time.time(),
        'argv': _redact_argv(sys.argv),
        'git': _git_rev(),
        'host': platform.node(),
        'platform': platform.platform(),
        'versions': _versions(),
        'endpoint': args.endpoint,
        'served_model': served,
        'endpoint_models': ep,
        'submission_source': str(Path(args.submission).resolve()),
        'submission_sha256': core.dir_sha256(agent_dir),
        'submission': sub_info,
        'budgets': budgets.to_dict(),
        'compaction': None if args.no_compaction else core.COMPACTION,
        'context_cache': None if args.no_compaction else core.CONTEXT_CACHE,
        'image': _image_info(args.image),
        'data': data,
        'task_ids': ids,
        'concurrency': args.concurrency,
        'timing_scale': args.timing_scale,
        'env_fixes': not args.no_env_fixes,
        'checkpoints': ckpts,
        'deviations': deviations,
    }
    if prev_manifest and prev_manifest.get('submission_sha256') != manifest['submission_sha256']:
        print('ERROR: run dir already holds a different submission; use a new --out')
        return 2
    if prev_manifest:
        changed, soft = condition_diff(prev_manifest, manifest)
        if changed and not args.allow_condition_change:
            print('ERROR: resuming would mix experimental conditions; finished tasks ran under the old ones:')
            for k in changed:
                print(f'  - {k}: {json.dumps(_get_path(prev_manifest, k), default=str)} -> '
                      f'{json.dumps(_get_path(manifest, k), default=str)}')
            print('Use a new --out, or pass --allow-condition-change (recorded in run.json sessions).')
            return 2
        for k in soft:
            print(f'[runner] note: {k} changed since the last session (not a result condition; recorded)')
    sessions = (prev_manifest or {}).get('sessions', [])
    sessions.append({'started_at': time.time(), 'argv': _redact_argv(sys.argv), 'n_tasks': len(ids),
                     'conditions_sha256': conditions_sha256(manifest),
                     'condition_change_allowed': bool(prev_manifest and args.allow_condition_change)})
    manifest['sessions'] = sessions
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    retry = {'infra_error', 'killed', 'worker_timeout'} if args.retry_failed else {'killed'}
    if args.rerun_all:
        retry |= {'done', 'infra_error', 'worker_timeout'}
    todo = [i for i in ids if _needs_run(run_dir / 'tasks' / i, retry)]
    print(f'[runner] {len(ids)} tasks selected, {len(ids) - len(todo)} already complete, {len(todo)} to run; '
          f'concurrency={args.concurrency}; budgets={budgets.to_dict()}; served={served}')
    if deviations:
        print('[runner] deviations from scorer:', *deviations, sep='\n  - ')

    hard_timeout = args.worker_timeout or (
        (budgets.max_time_minutes * 60 if budgets.max_time_minutes else 3600) + args.setup_allowance)

    # Live status for the Board (off unless --status-dir / $GEMMA4_STATUS_DIR); never raises, never blocks the run.
    todo_set = set(todo)
    prior = [i for i in ids if i not in todo_set]
    prior_empty = 0
    for i in prior:
        try:
            if json.loads((run_dir / 'tasks' / i / 'result.json').read_text()).get('empty_patch'):
                prior_empty += 1
        except Exception:
            pass
    st = status.StatusWriter(args.status_dir, run_dir.name, args.submission, args.endpoint, tasks_total=len(ids),
                             tasks_done=len(prior), empty=prior_empty, concurrency=args.concurrency,
                             grades_path=run_dir / 'grades.jsonl', task_ids=ids).start()

    async def _orchestrate() -> None:
        sem = asyncio.Semaphore(args.concurrency)
        done_count = 0
        t_start = time.time()

        async def one(idx: int, iid: str) -> None:
            nonlocal done_count
            async with sem:
                task_dir = run_dir / 'tasks' / iid
                _reset_task_dir(task_dir)
                spec = {
                    'instance_id': iid,
                    'task_dir': str(task_dir),
                    'task_index': ids.index(iid) + 1,
                    'total_tasks': len(ids),
                    'submission_dir': str(agent_dir),
                    'declared_model': sub_info['declared_model'],
                    'served_model': served,
                    'endpoint': args.endpoint,
                    'image': args.image,
                    'budgets': budgets.to_dict(),
                    'compaction': manifest['compaction'],
                    'context_cache': manifest['context_cache'],
                    'data': data,
                    'log_level': args.log_level,
                    'env_fixes': not args.no_env_fixes,
                    'checkpoints': ckpts,
                }
                spec_path = task_dir / 'spec.json'
                spec_path.write_text(json.dumps(spec, indent=2))
                t0 = time.time()
                st.task_started(iid)
                try:
                    await _run_worker(spec_path, task_dir, hard_timeout, api_key=args.api_key)
                    res = json.loads((task_dir / 'result.json').read_text())
                except BaseException:
                    st.task_finished(iid, 'interrupted')
                    raise
                st.task_finished(iid, res.get('status'), res.get('termination'),
                                 empty_patch=bool(res.get('empty_patch')))
                done_count += 1
                print(f'[runner] {done_count}/{len(todo)} {iid} ({repo_of[iid]}): {res.get("status")} '
                      f'{res.get("termination", res.get("error", ""))} patch={res.get("patch_chars", 0)}c '
                      f'tools={res.get("tool_calls_total", "-")} agent={res.get("timings", {}).get("agent_s", "-")}s '
                      f'wall={time.time() - t0:.0f}s elapsed={time.time() - t_start:.0f}s', flush=True)

        await asyncio.gather(*(one(i, iid) for i, iid in enumerate(todo)))

    t0 = time.time()
    final_state, final_event = 'done', 'run finished'
    try:
        asyncio.run(_orchestrate())
    except KeyboardInterrupt:
        print('[runner] interrupted; re-run the same command to resume')
        final_state, final_event = 'failed', 'interrupted; re-run to resume'
    except BaseException:
        st.close('failed', 'runner crashed')
        raise
    st.close(final_state, final_event)
    sessions[-1]['ended_at'] = time.time()
    sessions[-1]['wall_s'] = round(time.time() - t0, 1)
    manifest['sessions'] = sessions
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    summary = report.build(run_dir, timing_scale=args.timing_scale)
    print(report.render_md(summary))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    summary = report.build(Path(args.run_dir), timing_scale=args.timing_scale, grades_path=args.grades)
    print(report.render_md(summary))
    return 0


def cmd_export_sft(args: argparse.Namespace) -> int:
    from harness.runner import export_sft
    n = export_sft.export(Path(args.run_dir), Path(args.out), grades_path=args.grades,
                          only_resolved=args.only_resolved, include_summarizer=args.include_summarizer)
    print(f'wrote {n} samples to {args.out}')
    return 0


def cmd_checkpoints(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / 'at'
    stats = checkpoints.export(run_dir, out_dir, checkpoints.parse(args.at) or None)
    if not stats:
        print(f'no checkpoints in {run_dir} (run with --checkpoints)')
        return 1
    for s in stats:
        print(f"t={checkpoints.fmt(s['t'])} min: {s['with_patch']}/{s['n']} with a patch {s['counts']} -> {s['path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='python -m harness.runner', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd')

    r = sub.add_parser('run', help='run a submission against tasks (default)')
    r.add_argument('--submission', required=True, help='submission dir or .zip (agent.yaml at root)')
    r.add_argument('--endpoint', default=os.environ.get('GEMMA4_ENDPOINT', 'http://127.0.0.1:8000/v1'))
    r.add_argument('--api-key', default=os.environ.get('LOCAL_API_KEY', 'EMPTY'))
    r.add_argument('--served-model', help='base model id on the endpoint (default: discovered from /v1/models)')
    r.add_argument('--out', required=True, help='run dir, e.g. runs/<name>; re-run the same command to resume')
    r.add_argument('--data-dir', default=os.environ.get('GEMMA4_DATA_DIR') or str(core.DEFAULT_DATA_DIR),
                   help='competition data root (tasks.jsonl, snapshots/, graphs/, embeddings/, wheels/, sandbox/setup.py); '
                        'default $GEMMA4_DATA_DIR (same env vars as harness/common/paths.py) or data/kaggle')
    r.add_argument('--tasks', help='tasks.jsonl (default <data-dir>/tasks.jsonl)')
    r.add_argument('--snapshots-dir', default=os.environ.get('GEMMA4_SNAPSHOTS_DIR'),
                   help='default $GEMMA4_SNAPSHOTS_DIR or <data-dir>/snapshots')
    r.add_argument('--wheels-dir', default=os.environ.get('GEMMA4_WHEELS_DIR'),
                   help='default $GEMMA4_WHEELS_DIR or <data-dir>/wheels if present')
    r.add_argument('--task-ids', help='subset name (harness/subsets/<name>.txt), ids file, or comma list')
    r.add_argument('--limit', type=int, help='only the first N selected tasks')
    r.add_argument('--concurrency', type=int, default=1, help='parallel task processes (one sandbox each)')
    r.add_argument('--image', default=os.environ.get('GEMMA4_SANDBOX_IMAGE') or 'swebench-sandbox:latest')
    r.add_argument('--override', action='append', metavar='KEY=VALUE',
                   help='DEVIATION: override an eval_config budget (recorded in run.json)')
    r.add_argument('--no-compaction', action='store_true',
                   help='DEVIATION: disable compaction + context cache (what `swegemma eval` does)')
    r.add_argument('--no-env-fixes', action='store_true',
                   help='do not apply harness/sandbox/task_env_fixes.json in the agent sandbox')
    r.add_argument('--retry-failed', action='store_true', help='re-run tasks with infra_error/worker_timeout')
    r.add_argument('--rerun-all', action='store_true', help='re-run every selected task (old dirs -> attempts/)')
    r.add_argument('--allow-missing-loras', action='store_true')
    r.add_argument('--allow-condition-change', action='store_true',
                   help='resume even though budgets/env fixes/image/model/... differ from the run dir (mixes conditions)')
    r.add_argument('--worker-timeout', type=float, help='hard kill per task, seconds (default budget + allowance)')
    r.add_argument('--setup-allowance', type=float, default=900.0,
                   help='seconds added to max_time_minutes for sandbox setup/teardown in the hard kill')
    r.add_argument('--timing-scale', type=float, default=1.0,
                   help='multiply local agent seconds by this to estimate Kaggle 4xL4 time (see docs/runner.md)')
    r.add_argument('--status-dir', default=os.environ.get('GEMMA4_STATUS_DIR') or None,
                   help='write live progress to DIR/<run>.json (+ index.json) every ~15 s for the Board; '
                        'default $GEMMA4_STATUS_DIR, off if unset. Never fails the run if DIR is unavailable')
    r.add_argument('--checkpoints', metavar='T1,T2,...',
                   help='minutes of agent time (fractions ok), e.g. 5,7.5,10: record the patch the scorer would grade '
                        'had max_time_minutes been T (tasks/<id>/checkpoints.jsonl); export with `checkpoints`')
    r.add_argument('--log-level', default='INFO')
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser('report', help='(re)build summary.json/summary.md for a run dir')
    rp.add_argument('run_dir')
    rp.add_argument('--timing-scale', type=float, default=None)
    rp.add_argument('--grades', help='grader output (jsonl/csv/json with instance_id|id + resolved) to merge')
    rp.set_defaults(fn=cmd_report)

    ex = sub.add_parser('export-sft', help='convert llm_calls.jsonl into chat SFT samples')
    ex.add_argument('run_dir')
    ex.add_argument('--out', required=True)
    ex.add_argument('--grades', help='grader output to filter/label by resolved')
    ex.add_argument('--only-resolved', action='store_true',
                    help='keep resolved tasks only; grades match by instance id, not by patch, so use one grades file per run')
    ex.add_argument('--include-summarizer', action='store_true')
    ex.set_defaults(fn=cmd_export_sft)

    ck = sub.add_parser('checkpoints', help='write predictions_at_<t>.jsonl (grader input) from a --checkpoints run')
    ck.add_argument('run_dir')
    ck.add_argument('--out-dir', help='default <run_dir>/at')
    ck.add_argument('--at', help='only these checkpoints (minutes, comma list); default all in run.json')
    ck.set_defaults(fn=cmd_checkpoints)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ('run', 'report', 'export-sft', 'checkpoints', '-h', '--help'):
        argv = ['run', *argv]
    args = build_parser().parse_args(argv)
    if not getattr(args, 'fn', None):
        build_parser().print_help()
        return 1
    return args.fn(args)
