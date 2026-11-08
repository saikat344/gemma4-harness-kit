"""Time checkpoints: what would have been graded had `max_time_minutes` been t (observer only).

Run one long-budget pass (e.g. `--override max_time_minutes=15 --checkpoints 5,7.5,10`) and read off the patch the
scorer would have graded at each shorter cap. The scorer grades the LAST `submit_patch` snapshot
(`ctx.submitted_patch`, even an empty one) if the agent submitted; otherwise the stripped working-tree diff
(`git add -N . && git diff --binary _swegemma_baseline`); otherwise nothing. The scorer checks its time budget
after each ADK event (`context.elapsed_seconds > timeout_sec`), so we check at the same points we can see: before
each LLM call (a cap crossed during tool execution) and when its response arrives (a cap crossed while decoding,
before that turn's tools run). Each crossing costs ONE docker exec (git diff on a throwaway copy of the index, so
the agent's own index/`git status` is untouched), never one per call.

Per task: `checkpoints.jsonl` (one line per t) and `checkpoints/<t>.patch` (the effective patch; empty file = none).
A task that ends before t gets its final graded patch for t (`reason: "ended_before"`): with a cap of t it would
have ended the same way. `python -m harness.runner checkpoints runs/<run>` turns these into
`predictions_at_<t>.jsonl` files in the grader's input format.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any

# Same diff as swegemma submit_patch / the fallback in agent_runner, but `git add -N` goes to a temp copy of the
# index (GIT_INDEX_FILE), so taking a checkpoint does not change what the agent later sees in `git status`.
WORKTREE_DIFF_CMD = (
    'cd /workspace && _ci=$(mktemp) && cp "$(git rev-parse --git-path index)" "$_ci" && '
    'export GIT_INDEX_FILE="$_ci" && git add -N . && '
    '(git diff --binary _swegemma_baseline 2>/dev/null || git diff --binary HEAD); '
    '_rc=$?; rm -f "$_ci"; exit $_rc'
)
EXEC_TIMEOUT_S = 30


def parse(spec: str | None) -> list[float]:
    """'5,7.5,10' -> [5.0, 7.5, 10.0] (minutes, sorted, de-duplicated; fractions allowed)."""
    if not spec:
        return []
    vals = sorted({float(s) for s in spec.split(',') if s.strip()})
    if any(v < 0 for v in vals):
        raise ValueError(f'checkpoints must be >= 0 minutes: {spec}')
    return vals


def fmt(t: float) -> str:
    """File-name form of a checkpoint: 5 -> '5', 7.5 -> '7.5'."""
    return f'{t:g}'


def sha(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def effective(submitted: bool, submitted_patch: str | None, worktree: str | None) -> tuple[str, str]:
    """(source, patch) the scorer would grade: a submission wins (even an empty one, as in agent_runner), else a
    non-empty working-tree diff, else nothing. worktree None = capture failed."""
    if submitted:
        return 'submitted', submitted_patch or ''
    if worktree is None:
        return 'error', ''
    if worktree.strip():
        return 'worktree', worktree
    return 'none', ''


class Checkpointer:
    """Captures submitted@t / worktree@t the first time agent elapsed passes t. Never raises."""

    def __init__(self, task_dir: Path, context: Any, minutes: list[float], test_files: list[str] | None = None,
                 exec_timeout: int = EXEC_TIMEOUT_S) -> None:
        self.context = context
        self.pending = sorted(minutes)
        self.test_files = list(test_files or ())
        self.exec_timeout = exec_timeout
        self.jsonl = task_dir / 'checkpoints.jsonl'
        self.patch_dir = task_dir / 'checkpoints'
        self.err_log = task_dir / 'checkpoints_errors.log'
        self.patch_dir.mkdir(parents=True, exist_ok=True)

    def _log_error(self, where: str) -> None:
        try:
            with open(self.err_log, 'a', encoding='utf-8') as f:
                f.write(f'[{time.strftime("%H:%M:%S")}] {where}\n{traceback.format_exc()}\n')
        except Exception:
            pass

    def _worktree_diff(self) -> str | None:
        ctx = self.context
        res = ctx.docker.exec(ctx.container_id, WORKTREE_DIFF_CMD, timeout=self.exec_timeout)
        if getattr(res, 'exit_code', 1) != 0:
            with open(self.err_log, 'a', encoding='utf-8') as f:
                f.write(f'[{time.strftime("%H:%M:%S")}] git diff exit={getattr(res, "exit_code", None)} '
                        f'status={getattr(res, "status", None)} stderr={(getattr(res, "stderr", "") or "")[:500]}\n')
            return None
        diff = res.stdout or ''
        if diff.strip():
            from swegemma.harness.verification import strip_protected_files_from_patch
            diff = strip_protected_files_from_patch(diff, extra_protected_paths=self.test_files)
        return diff

    def _write(self, ts: list[float], rec: dict[str, Any], patch: str) -> None:
        with open(self.jsonl, 'a', encoding='utf-8') as f:
            for t in ts:
                (self.patch_dir / f'{fmt(t)}.patch').write_text(patch, encoding='utf-8')
                f.write(json.dumps({'t': t, **rec}) + '\n')

    def poll(self, where: str, call_index: int | None = None) -> None:
        """Capture every pending checkpoint whose time has passed (one shared snapshot)."""
        try:
            ctx = self.context
            if not self.pending or getattr(ctx, 'agent_start_time', None) is None:
                return
            elapsed = float(ctx.agent_elapsed_seconds or 0.0)
            due = [t for t in self.pending if elapsed > t * 60.0]
            if not due:
                return
            self.pending = [t for t in self.pending if t not in due]
            submitted = bool(getattr(ctx, 'patch_submitted', False))
            sub_patch = getattr(ctx, 'submitted_patch', None) if submitted else None
            t0 = time.perf_counter()
            try:
                worktree = self._worktree_diff()
            except Exception:
                self._log_error(f'worktree diff at {where} call {call_index}')
                worktree = None
            source, patch = effective(submitted, sub_patch, worktree)
            self._write(due, {
                'elapsed_actual': round(elapsed, 3),
                'reason': 'crossed',
                'where': where,
                'call_index': call_index,
                'has_submitted': submitted,
                'submitted_sha': sha(sub_patch) if submitted else None,
                'worktree_sha': sha(worktree),
                'effective_source': source,
                'effective_sha': sha(patch) if patch.strip() else None,
                'effective_chars': len(patch.strip()),
                'capture_s': round(time.perf_counter() - t0, 3),
            }, patch)
        except Exception:
            self._log_error(f'poll at {where} call {call_index}')

    def finish(self, final_patch: str, submitted: bool) -> None:
        """Task ended: every checkpoint not yet reached gets the final graded patch (reason ended_before)."""
        try:
            if not self.pending:
                return
            ctx = self.context
            due, self.pending = self.pending, []
            final_patch = final_patch or ''
            source = 'submitted' if submitted else ('worktree' if final_patch.strip() else 'none')
            self._write(due, {
                'elapsed_actual': round(float(getattr(ctx, 'agent_elapsed_seconds', 0.0) or 0.0), 3),
                'reason': 'ended_before',
                'where': 'end',
                'call_index': None,
                'has_submitted': submitted,
                'submitted_sha': sha(getattr(ctx, 'submitted_patch', None) or '') if submitted else None,
                'worktree_sha': None if submitted else sha(final_patch),  # sandbox is gone; final fallback diff
                'effective_source': source,
                'effective_sha': sha(final_patch) if final_patch.strip() else None,
                'effective_chars': len(final_patch.strip()),
                'capture_s': 0.0,
            }, final_patch)
        except Exception:
            self._log_error('finish')


# --------------------------------------------------------------------------------------------------
# Export: one grader-input predictions file per checkpoint
# --------------------------------------------------------------------------------------------------


def _records(task_dir: Path) -> dict[float, dict[str, Any]]:
    out: dict[float, dict[str, Any]] = {}
    p = task_dir / 'checkpoints.jsonl'
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            try:
                rec = json.loads(line)
                out[float(rec['t'])] = rec
            except Exception:
                continue
    return out


def export(run_dir: Path, out_dir: Path, minutes: list[float] | None = None) -> list[dict[str, Any]]:
    """Write out_dir/predictions_at_<t>.jsonl (same rows as predictions.jsonl) for each checkpoint.

    A task gets its checkpoint patch whenever a record for t exists, whatever its final status: a task the
    orchestrator killed late still had a valid state at an earlier crossing. No record (not reached and the
    worker never finished, or not started) -> NO_PATCH."""
    from harness.runner.report import NO_PATCH

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'run.json').read_text())
    ids: list[str] = manifest['task_ids']
    recs = {iid: _records(run_dir / 'tasks' / iid) for iid in ids}
    ts = sorted(minutes or manifest.get('checkpoints') or {t for r in recs.values() for t in r})
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for t in ts:
        path = out_dir / f'predictions_at_{fmt(t)}.jsonl'
        counts: dict[str, int] = {}
        with open(path, 'w', encoding='utf-8') as f:
            for iid in ids:
                rec = recs[iid].get(t)
                patch = ''
                if rec is not None:
                    pf = run_dir / 'tasks' / iid / 'checkpoints' / f'{fmt(t)}.patch'
                    patch = pf.read_text(encoding='utf-8') if pf.exists() else ''
                    key = f"{rec.get('reason')}:{rec.get('effective_source')}"
                else:
                    key = 'missing'
                counts[key] = counts.get(key, 0) + 1
                pred = patch if patch.strip() else NO_PATCH
                f.write(json.dumps({'instance_id': iid, 'patch': pred, 'id': iid, 'prediction': pred,
                                    'status': 'checkpoint' if rec is not None else 'missing',
                                    'checkpoint_t': t,
                                    'checkpoint_reason': rec.get('reason') if rec else None,
                                    'checkpoint_source': rec.get('effective_source') if rec else None}) + '\n')
        stats.append({'t': t, 'path': str(path), 'n': len(ids),
                      'with_patch': sum(1 for iid in ids if (recs[iid].get(t) or {}).get('effective_chars')),
                      'counts': dict(sorted(counts.items()))})
    return stats
