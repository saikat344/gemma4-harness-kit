"""Aggregate per-task results into predictions.jsonl, summary.json and summary.md.

Timing comparison with Kaggle 4xL4 (inferred from public Kaggle discussions): about 900 s per 1M tokens processed,
so ~5.5 min (330 s) per task for ~120 hidden tasks in 12 h (incl. ~1 h start-up). Two estimates:
  kaggle_tokens_s = total_tokens(task) * 900 / 1e6     (tokens = sum over LLM calls of prompt+completion)
  scaled_agent_s  = local agent_s * --timing-scale      (calibrate the scale once from a run on Kaggle)
Both are agent time only. Kaggle's 12 h also pays per-task sandbox setup, so each projection also has an
`..._incl_setup` variant: + mean measured setup_s of this run (unscaled; local setup is the proxy), or ASSUMED_SETUP_S
when the run recorded none.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

KAGGLE_S_PER_MTOK = 900.0
KAGGLE_TASK_BUDGET_S = 330.0
# per-task sandbox setup when a run has no measured setup_s (inferred: ~60 s/task, code audit 2 finding N4)
ASSUMED_SETUP_S = 60.0
NO_PATCH = 'NO_PATCH'


def _env_fixes_label(v: Any) -> str:
    if v is None:
        return 'unknown (run predates the flag)'
    return 'on (as grader; the scorer applies none)' if v else 'off (official env as shipped, like the scorer)'


def _stats(xs: list[float]) -> dict[str, Any]:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {'n': 0}
    xs_sorted = sorted(xs)
    return {
        'n': len(xs),
        'mean': round(statistics.fmean(xs), 2),
        'median': round(statistics.median(xs), 2),
        'p90': round(xs_sorted[min(len(xs) - 1, int(0.9 * len(xs)))], 2),
        'max': round(max(xs), 2),
        'sum': round(sum(xs), 2),
    }


def load_grades(path: str | Path | None) -> dict[str, bool]:
    """Accept jsonl / json list / csv with (instance_id|id) and resolved|passed (harness.grader writes passed)."""
    if not path:
        return {}
    p = Path(path)
    rows: list[dict[str, Any]] = []
    if p.suffix == '.csv':
        with open(p, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    else:
        text = p.read_text(encoding='utf-8').strip()
        if text.startswith('['):
            rows = json.loads(text)
        else:
            try:
                rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
            except json.JSONDecodeError:  # pretty-printed single object, e.g. a summary with results
                obj = json.loads(text)
                rows = obj.get('results') or obj.get('task_results') or []
            if len(rows) == 1 and not (rows[0].get('instance_id') or rows[0].get('id')):
                rows = rows[0].get('results') or rows[0].get('task_results') or []
    out = {}
    for r in rows:
        iid = r.get('instance_id') or r.get('id')
        val = r.get('resolved', r.get('passed'))
        if iid is None or val is None:
            continue
        out[str(iid)] = str(val).lower() in ('true', '1', 'yes') if not isinstance(val, bool) else val
    return out


def write_predictions(run_dir: Path, ids: list[str], results: dict[str, dict]) -> Path:
    """One row per selected task, in task order. Fields for both grader conventions:
    instance_id + patch (SWE-bench style) and id + prediction (NB1 / Kaggle style)."""
    path = run_dir / 'predictions.jsonl'
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        for iid in ids:
            patch_file = run_dir / 'tasks' / iid / 'patch.diff'
            res = results.get(iid, {})
            patch = patch_file.read_text(encoding='utf-8') if (res.get('status') == 'done' and patch_file.exists()) else ''
            pred = patch if patch.strip() else NO_PATCH
            f.write(json.dumps({'instance_id': iid, 'patch': pred, 'id': iid, 'prediction': pred,
                                'status': res.get('status', 'missing')}) + '\n')
    tmp.replace(path)
    return path


def build(run_dir: Path, timing_scale: float | None = None, grades_path: str | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'run.json').read_text())
    ids: list[str] = manifest['task_ids']
    if timing_scale is None:
        timing_scale = float(manifest.get('timing_scale') or 1.0)
    results: dict[str, dict] = {}
    for iid in ids:
        rp = run_dir / 'tasks' / iid / 'result.json'
        if rp.exists():
            try:
                results[iid] = json.loads(rp.read_text())
            except Exception:
                results[iid] = {'status': 'unreadable'}
    write_predictions(run_dir, ids, results)
    grades = load_grades(grades_path) if grades_path else {}
    if not grades and (run_dir / 'grades.jsonl').exists():
        grades = load_grades(run_dir / 'grades.jsonl')

    done = {i: r for i, r in results.items() if r.get('status') == 'done'}
    status_counts = Counter(r.get('status', 'missing') for r in results.values())
    status_counts['not_started'] = len(ids) - len(results)

    tool_breakdown: Counter = Counter()
    tool_errors: dict[str, Counter] = defaultdict(Counter)
    termination = Counter()
    finish = Counter()
    per_task = []
    for iid in ids:
        r = results.get(iid, {})
        row = {'instance_id': iid, 'status': r.get('status', 'not_started'), 'repo': r.get('repo')}
        if r.get('status') == 'done':
            tool_breakdown.update(r.get('tool_call_breakdown', {}))
            for tool, errs in (r.get('tool_errors') or {}).items():
                tool_errors[tool].update(errs)
            termination[r.get('termination')] += 1
            finish.update(r.get('finish_reasons', {}))
            tok = r.get('total_tokens') or 0
            agent_s = (r.get('timings') or {}).get('agent_s')
            row.update({
                'termination': r.get('termination'),
                'patch_chars': r.get('patch_chars'),
                'tool_calls': r.get('tool_calls_total'),
                'tool_errors': r.get('tool_errors_total'),
                'llm_calls': r.get('llm_calls_agent'),
                'compactions': r.get('compactions'),
                'nudges': r.get('nudges'),
                'total_tokens': tok,
                'max_prompt_tokens': r.get('max_prompt_tokens'),
                'setup_s': (r.get('timings') or {}).get('setup_s'),
                'agent_s': agent_s,
                'scaled_agent_s': round(agent_s * timing_scale, 1) if agent_s is not None else None,
                'kaggle_tokens_s': round(tok * KAGGLE_S_PER_MTOK / 1e6, 1),
            })
        else:
            row['error'] = r.get('error')
        if iid in grades:
            row['resolved'] = grades[iid]
        per_task.append(row)

    d = list(done.values())
    n = len(d) or 1
    empty = sum(1 for r in d if r.get('empty_patch'))
    # malformed / failed edits only; BudgetExceeded is a budget event, not an edit error
    edit_errs = Counter({k: v for k, v in tool_errors.get('edit_file', Counter()).items() if k != 'BudgetExceeded'})
    kag = [row['kaggle_tokens_s'] for row in per_task if row.get('kaggle_tokens_s') is not None]
    scaled = [row['scaled_agent_s'] for row in per_task if row.get('scaled_agent_s') is not None]
    setups = [float(row['setup_s']) for row in per_task if row.get('setup_s') is not None]
    setup_mean = statistics.fmean(setups) if setups else ASSUMED_SETUP_S
    setup_basis = (f'measured: mean setup_s {setup_mean:.1f}s over {len(setups)} tasks (local sandbox, unscaled)' if setups
                   else f'assumed {ASSUMED_SETUP_S:.0f}s/task (no setup_s recorded in this run)')

    def hours_120(xs: list[float], extra_s: float = 0.0) -> float | None:
        return round((statistics.fmean(xs) + extra_s) * 120 / 3600, 2) if xs else None
    sessions = manifest.get('sessions', [])
    summary: dict[str, Any] = {
        'run': manifest.get('name'),
        'served_model': manifest.get('served_model'),
        'submission_sha256': manifest.get('submission_sha256'),
        'budgets': manifest.get('budgets'),
        'env_fixes': manifest.get('env_fixes'),
        'deviations': manifest.get('deviations'),
        'n_selected': len(ids),
        'status_counts': dict(status_counts),
        'infra_failures': {i: results[i].get('error') for i in results if results[i].get('status') != 'done'},
        'wall_time_s': round(sum(s.get('wall_s') or 0 for s in sessions), 1),
        'concurrency': manifest.get('concurrency'),
        'empty_patch_rate': round(empty / n, 3) if d else None,
        'empty_patches': empty,
        'submitted_rate': round(sum(1 for r in d if r.get('patch_submitted')) / n, 3) if d else None,
        'fallback_diff_patches': sum(1 for r in d if r.get('patch_source') == 'fallback_diff'),
        'termination': dict(termination),
        'agent_exceptions_after_submit': sum(1 for r in d if r.get('agent_exception_after_submit')),
        'agent_minutes_per_task': _stats([(r.get('timings') or {}).get('agent_s', 0) / 60 for r in d]),
        'setup_seconds_per_task': _stats([(r.get('timings') or {}).get('setup_s') for r in d]),
        'total_seconds_per_task': _stats([(r.get('timings') or {}).get('total_s') for r in d]),
        'tool_calls_per_task': _stats([r.get('tool_calls_total', 0) for r in d]),
        'llm_calls_per_task': _stats([r.get('llm_calls_agent', 0) for r in d]),
        'tokens_per_task': _stats([r.get('total_tokens', 0) for r in d]),
        'max_prompt_tokens': _stats([r.get('max_prompt_tokens', 0) for r in d]),
        'completion_tok_per_llm_s': _stats([r.get('completion_tok_per_llm_s') for r in d]),
        'tool_call_breakdown': dict(tool_breakdown),
        'tool_errors': {k: dict(v) for k, v in tool_errors.items()},
        'tool_errors_total': sum(sum(v.values()) for v in tool_errors.values()),
        'edit_file_errors': sum(edit_errs.values()),
        'edit_file_errors_by_type': dict(edit_errs),
        'edit_file_error_rate': round(sum(edit_errs.values()) / max(1, tool_breakdown.get('edit_file', 0)), 3),
        'command_timeouts': sum(v.get('TimeoutExceeded', 0) for v in tool_errors.values()),
        'budget_exceeded_tool_responses': sum(v.get('BudgetExceeded', 0) for v in tool_errors.values()),
        'agent_time_budget_hits': termination.get('time_budget', 0),
        'unparsed_tool_call_text': sum(r.get('unparsed_tool_call_text', 0) for r in d),
        'nudges_total': sum(r.get('nudges', 0) for r in d),
        'compactions_total': sum(r.get('compactions', 0) for r in d),
        'llm_call_errors': sum(r.get('llm_call_errors', 0) for r in d),
        'finish_reasons': dict(finish),
        'kaggle_projection': {
            'method': f'{KAGGLE_S_PER_MTOK:.0f} s per 1M tokens (prompt+completion summed over calls)',
            'per_task_s': _stats(kag),
            'tasks_over_330s': sum(1 for x in kag if x > KAGGLE_TASK_BUDGET_S),
            'projected_hours_for_120_tasks': hours_120(kag),
            'projected_hours_for_120_tasks_incl_setup': hours_120(kag, setup_mean),
            'setup_s_per_task_used': round(setup_mean, 1),
            'setup_basis': setup_basis,
            'timing_scale': timing_scale,
            'scaled_agent_s': _stats(scaled),
            'scaled_hours_for_120_tasks': hours_120(scaled),
            'scaled_hours_for_120_tasks_incl_setup': hours_120(scaled, setup_mean),
        },
        'per_task': per_task,
    }
    if grades:
        graded = [row for row in per_task if 'resolved' in row]
        by_repo: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in graded:
            by_repo[row.get('repo') or '?'][0] += int(bool(row['resolved']))
            by_repo[row.get('repo') or '?'][1] += 1
        summary['resolved'] = sum(1 for row in graded if row['resolved'])
        summary['graded'] = len(graded)
        summary['resolved_by_repo'] = {k: f'{a}/{b}' for k, (a, b) in by_repo.items()}
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / 'summary.md').write_text(render_md(summary))
    return summary


def render_md(s: dict[str, Any]) -> str:
    def st(key: str, unit: str = '') -> str:
        v = s.get(key) or {}
        return f"mean {v.get('mean', '-')}{unit}, median {v.get('median', '-')}{unit}, p90 {v.get('p90', '-')}{unit}" if v.get('n') else '-'

    kp = s['kaggle_projection']
    lines = [
        f"# Run {s['run']}",
        '',
        f"- model: `{s['served_model']}`; budgets: {s['budgets']}",
        f"- env fixes: {_env_fixes_label(s.get('env_fixes'))}",
        f"- deviations: {s['deviations'] or 'none'}",
        f"- tasks: {s['n_selected']} selected; status {s['status_counts']}; wall {s['wall_time_s']}s at concurrency {s['concurrency']}",
    ]
    if 'resolved' in s:
        lines.append(f"- **resolved {s['resolved']}/{s['graded']}** by repo {s['resolved_by_repo']}")
    lines += [
        f"- empty-patch rate {s['empty_patch_rate']} ({s['empty_patches']}); submitted rate {s['submitted_rate']}; fallback-diff patches {s['fallback_diff_patches']}",
        f"- termination: {s['termination']}; agent exceptions after submit: {s.get('agent_exceptions_after_submit', 0)}",
        f"- agent minutes/task: {st('agent_minutes_per_task')}; setup s/task: {st('setup_seconds_per_task')}",
        f"- tool calls/task: {st('tool_calls_per_task')}; LLM calls/task: {st('llm_calls_per_task')}",
        f"- tokens/task: {st('tokens_per_task')}; max prompt tokens: {st('max_prompt_tokens')}; decode tok/s: {st('completion_tok_per_llm_s')}",
        f"- tool errors: {s['tool_errors_total']} {s['tool_errors']}; edit_file error rate {s['edit_file_error_rate']}",
        f"- command timeouts {s['command_timeouts']}; BudgetExceeded responses {s['budget_exceeded_tool_responses']}; time-budget hits {s['agent_time_budget_hits']}",
        f"- nudges {s['nudges_total']}; compactions {s['compactions_total']}; unparsed tool-call text {s['unparsed_tool_call_text']}; LLM call errors {s['llm_call_errors']}",
        f"- Kaggle 4xL4 projection ({kp['method']}): per task {kp['per_task_s'].get('mean', '-')}s mean, "
        f"{kp['tasks_over_330s']} over 330s, {kp['projected_hours_for_120_tasks']} h per 120 tasks "
        f"({kp.get('projected_hours_for_120_tasks_incl_setup', '-')} h incl. setup); "
        f"scaled local time x{kp['timing_scale']}: {kp['scaled_agent_s'].get('mean', '-')}s mean, {kp['scaled_hours_for_120_tasks']} h per 120 "
        f"({kp.get('scaled_hours_for_120_tasks_incl_setup', '-')} h incl. setup); setup {kp.get('setup_basis', '-')}",
    ]
    if s['infra_failures']:
        lines.append(f"- infra failures: {s['infra_failures']}")
    lines += ['', '| task | status | term | patch | tools | errs | llm | comp | tokens | agent_s | kaggle_s | resolved |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in s['per_task']:
        lines.append(f"| {r['instance_id']} | {r['status']} | {r.get('termination', '')} | {r.get('patch_chars', '')} | "
                     f"{r.get('tool_calls', '')} | {r.get('tool_errors', '')} | {r.get('llm_calls', '')} | {r.get('compactions', '')} | "
                     f"{r.get('total_tokens', '')} | {r.get('agent_s', '')} | {r.get('kaggle_tokens_s', '')} | {r.get('resolved', '')} |")
    return '\n'.join(lines) + '\n'
