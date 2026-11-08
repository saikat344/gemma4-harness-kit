"""End-to-end runner plumbing against the mock OpenAI server and a real Docker sandbox (no GPU).

Uses swebench-sandbox:latest if present, else the throwaway gemma4-runner-test:latest
(docs/runner.md "Tests"). Snapshots: httpx_3672 (1.5 MB), requests_6592 (36 MB).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.runner.mock_openai import parse_patch

from .conftest import DATA, FIXTURE_SUB, SAMPLE_SUB, require_snapshots, run_runner


def _result(run: Path, iid: str) -> dict:
    return json.loads((run / 'tasks' / iid / 'result.json').read_text())


def _gold(iid: str) -> str:
    for line in (DATA / 'tasks.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['instance_id'] == iid:
            return row['patch']
    raise KeyError(iid)


def test_oracle_two_tasks_concurrent_resume_and_sft(tmp_path, image, mock_server):
    require_snapshots('httpx_3672', 'requests_6592')
    endpoint, state = mock_server('oracle')
    run = tmp_path / 'run'
    args = ['--submission', str(FIXTURE_SUB), '--endpoint', endpoint, '--out', str(run),
            '--task-ids', 'httpx_3672,requests_6592', '--concurrency', '2', '--image', image]
    cp = run_runner(*args)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    preds = [json.loads(ln) for ln in (run / 'predictions.jsonl').read_text().splitlines()]
    assert [p['instance_id'] for p in preds] == ['httpx_3672', 'requests_6592']
    for p in preds:
        iid = p['instance_id']
        r = _result(run, iid)
        assert r['status'] == 'done' and r['termination'] == 'submitted', r
        assert p['patch'] == p['prediction'] != 'NO_PATCH'
        changed = {ln.split(' b/')[-1] for ln in p['patch'].splitlines() if ln.startswith('diff --git')}
        assert changed == {f['path'] for f in parse_patch(_gold(iid))}
        assert r['tool_errors']['edit_file'] == {'MissingArguments': 1, 'FileEditError': 1}
        td = run / 'tasks' / iid
        for f in ('trace_legacy.json', f'traces/trace_{iid}.json', f'logs/{iid}.log', 'llm_calls.jsonl', 'patch.diff'):
            assert (td / f).exists(), f
        calls = [json.loads(ln) for ln in (td / 'llm_calls.jsonl').read_text().splitlines()]
        assert calls and all('budget_before' in c and c['response'] for c in calls)
        assert calls[0]['request']['params']['extra_body']['chat_template_kwargs'] == {'enable_thinking': True}
        assert any(e['type'] == 'thinking' for e in json.loads((td / 'trace_legacy.json').read_text())['entries'])
        assert r['timings']['agent_s'] > 0 and r['timings']['setup_s'] > 0
        assert not (td / 'containers.txt').read_text().strip() == ''
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['status_counts']['done'] == 2 and summary['empty_patch_rate'] == 0.0
    assert summary['edit_file_errors'] == 4 and summary['kaggle_projection']['per_task_s']['n'] == 2

    # resume: nothing re-runs
    cp2 = run_runner(*args)
    assert cp2.returncode == 0 and '2 already complete, 0 to run' in cp2.stdout, cp2.stdout

    # report with timing scale + grades merge
    (run / 'grades.jsonl').write_text(json.dumps({'instance_id': 'httpx_3672', 'passed': True}) + '\n')
    cp3 = run_runner('report', str(run), '--timing-scale', '0.5')
    assert cp3.returncode == 0 and 'resolved 1/1' in cp3.stdout, cp3.stdout

    # SFT export
    out = tmp_path / 'sft.jsonl'
    cp4 = run_runner('export-sft', str(run), '--out', str(out))
    assert cp4.returncode == 0, cp4.stderr
    samples = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(samples) == 2  # one maximal conversation per task (no compaction)
    assert samples[0]['messages'][-1]['role'] == 'assistant' and samples[0]['tools']


def test_sample_submission_routes_loras_and_subagent_budget(tmp_path, image, mock_server):
    require_snapshots('httpx_3672')
    endpoint, state = mock_server('oracle', loras=['main_lora', 'tool_lora'])
    run = tmp_path / 'run'
    cp = run_runner('--submission', str(SAMPLE_SUB), '--endpoint', endpoint, '--out', str(run),
                    '--task-ids', 'httpx_3672', '--image', image)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert {e['model'] for e in state.log} == {'openai/main_lora'} or 'main_lora' in {e['model'] for e in state.log}
    r = _result(run, 'httpx_3672')
    # sample eval_config: max_tool_calls 10 -> BudgetExceeded responses, but submit_patch is free
    assert r['tool_errors']['edit_file'].get('BudgetExceeded', 0) > 0
    assert r['patch_submitted']


def test_missing_lora_is_refused(tmp_path, mock_server):
    if not SAMPLE_SUB.exists():
        pytest.skip('sample_submission missing (competition download)')
    endpoint, _ = mock_server('oracle')
    cp = run_runner('--submission', str(SAMPLE_SUB), '--endpoint', endpoint, '--out', str(tmp_path / 'r'),
                    '--task-ids', 'httpx_3672')
    assert cp.returncode == 2 and 'not served' in cp.stdout


def test_empty_agent_gives_no_patch_after_nudges(tmp_path, image, mock_server):
    require_snapshots('httpx_3672')
    endpoint, _ = mock_server('empty')
    run = tmp_path / 'run'
    status_dir = tmp_path / 'status'
    status_dir.mkdir()
    cp = run_runner('--submission', str(FIXTURE_SUB), '--endpoint', endpoint, '--out', str(run),
                    '--task-ids', 'httpx_3672', '--image', image, '--status-dir', str(status_dir))
    assert cp.returncode == 0, cp.stdout + cp.stderr
    r = _result(run, 'httpx_3672')
    assert r['empty_patch'] and r['nudges'] == 3 and r['termination'] == 'no_submit_no_diff', r
    st = json.loads((status_dir / 'run.json').read_text())  # --status-dir: final live-status file (run dir is 'run')
    assert (st['state'], st['tasks_done'], st['tasks_total'], st['empty_so_far'], st['tasks_running']) == ('done', 1, 1, 1, [])
    assert 'http' not in json.dumps(st) and json.loads((status_dir / 'index.json').read_text())['runs'][0]['run'] == 'run'
    assert json.loads((run / 'predictions.jsonl').read_text())['patch'] == 'NO_PATCH'


def test_raw_tool_call_text_counts_and_nudges(tmp_path, image, mock_server):
    require_snapshots('httpx_3672')
    endpoint, _ = mock_server('rawtool')
    run = tmp_path / 'run'
    cp = run_runner('--submission', str(FIXTURE_SUB), '--endpoint', endpoint, '--out', str(run),
                    '--task-ids', 'httpx_3672', '--image', image)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    r = _result(run, 'httpx_3672')
    assert r['unparsed_tool_call_text'] >= 1 and r['finish_reasons'].get('length', 0) >= 1, r


def test_compaction_is_triggered_and_recorded(tmp_path, image, mock_server):
    require_snapshots('httpx_3672')
    endpoint, _ = mock_server('oracle', inflate=20000)  # reported prompt tokens > 14336 threshold
    run = tmp_path / 'run'
    cp = run_runner('--submission', str(FIXTURE_SUB), '--endpoint', endpoint, '--out', str(run),
                    '--task-ids', 'httpx_3672', '--image', image)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    r = _result(run, 'httpx_3672')
    assert r['compactions'] >= 1 and r['llm_calls_summarizer'] >= 1, r
    assert r['compactions_in_trace'] == 0  # the official trace never sees token-threshold compaction
    comp = [json.loads(ln) for ln in (run / 'tasks' / 'httpx_3672' / 'compactions.jsonl').read_text().splitlines()]
    assert comp[0]['summary'].startswith('Summary') and comp[0]['events_in_window'] > 0
    # plumbing check: the mock's fixed summary does not quote the issue, so the first post-compaction request
    # no longer contains it verbatim (with a real model this depends on the summary: boundary B2)
    calls = [json.loads(ln) for ln in (run / 'tasks' / 'httpx_3672' / 'llm_calls.jsonl').read_text().splitlines()]
    after = [c for c in calls if c['kind'] == 'agent' and c['call_index'] > comp[0]['after_llm_call_index']]
    assert after and 'Problem Statement' not in json.dumps(after[0]['request']['messages'])
    assert r['patch_submitted']


def test_worker_hard_timeout_kills_and_cleans_containers(tmp_path, image, mock_server):
    require_snapshots('httpx_3672')
    endpoint, _ = mock_server('oracle', delay=5.0)
    run = tmp_path / 'run'
    cp = run_runner('--submission', str(FIXTURE_SUB), '--endpoint', endpoint, '--out', str(run),
                    '--task-ids', 'httpx_3672', '--image', image, '--worker-timeout', '12')
    assert cp.returncode == 0, cp.stdout + cp.stderr
    r = _result(run, 'httpx_3672')
    assert r['status'] in ('worker_timeout', 'killed'), r
    import docker
    client = docker.from_env()
    for cid in (run / 'tasks' / 'httpx_3672' / 'containers.txt').read_text().split():
        try:
            c = client.containers.get(cid)
            assert c.status not in ('running',), cid
        except docker.errors.NotFound:
            pass
    assert json.loads((run / 'predictions.jsonl').read_text())['patch'] == 'NO_PATCH'


def test_time_budget_override(tmp_path, image, mock_server):
    require_snapshots('httpx_3672')
    endpoint, _ = mock_server('oracle', delay=2.0)
    run = tmp_path / 'run'
    cp = run_runner('--submission', str(FIXTURE_SUB), '--endpoint', endpoint, '--out', str(run),
                    '--task-ids', 'httpx_3672', '--image', image, '--override', 'max_time_minutes=0.1')
    assert cp.returncode == 0, cp.stdout + cp.stderr
    r = _result(run, 'httpx_3672')
    assert r['termination'] == 'time_budget' and r['timings']['agent_s'] < 15, r
    manifest = json.loads((run / 'run.json').read_text())
    assert any('budget overrides' in d for d in manifest['deviations'])


def _tool_outputs(run: Path, iid: str) -> list[str]:
    entries = json.loads((run / 'tasks' / iid / 'trace_legacy.json').read_text())['entries']
    return [e['result'] for e in entries if e.get('type') == 'tool_response']


def _between(text: str, tag: str) -> str:
    # the tool result is JSON-encoded inside the trace entry, so newlines may be literal "\n"
    text = text.replace('\\n', '\n')
    return text.split(f'{tag}_BEGIN', 1)[1].split(f'{tag}_END', 1)[0].strip()


@pytest.mark.parametrize('env_fixes', [True, False], ids=['fixes_on', 'fixes_off'])
def test_env_fixes_in_agent_sandbox_fastapi(tmp_path, image, mock_server, env_fixes):
    """With fixes (default) the agent's very first tool call already sees the fixed starlette (< 0.49, what the
    fastapi snapshot requires) and a clean `git status` / empty `git diff HEAD`, so fixes don't leak into the patch.
    Without fixes it sees the injected newest starlette. Needs the fastapi_9425 snapshot (214 MB;
    harness/grader/fetch_snapshots.sh) and the real swebench-sandbox image (the throwaway test image has no
    starlette wheels)."""
    from packaging.version import Version

    if image != 'swebench-sandbox:latest':
        pytest.skip('needs swebench-sandbox:latest (competition wheels incl. starlette)')
    require_snapshots('fastapi_9425')
    iid = 'fastapi_9425'
    endpoint, _ = mock_server('probe')
    run = tmp_path / 'run'
    extra = [] if env_fixes else ['--no-env-fixes']
    cp = run_runner('--submission', str(FIXTURE_SUB), '--endpoint', endpoint, '--out', str(run),
                    '--task-ids', iid, '--image', image, *extra)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    r = _result(run, iid)
    assert r['status'] == 'done', r

    manifest = json.loads((run / 'run.json').read_text())
    assert manifest['env_fixes'] is env_fixes
    assert any('test-env fixes' in d for d in manifest['deviations']) is env_fixes
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['env_fixes'] is env_fixes
    assert f'- env fixes: {"on" if env_fixes else "off"}' in (run / 'summary.md').read_text()

    first = _tool_outputs(run, iid)[0]  # the agent's first tool call, i.e. before it could do anything itself
    flat = first.replace('\\n', '\n')
    dist = flat.split('STARLETTE_DIST=', 1)[1].split()[0].strip('"\\,')
    imp = flat.split('STARLETTE_IMPORT=', 1)[1].split()[0].strip('"\\,')
    assert dist == imp
    # clean tree at the agent's first command: nothing from setup or the fixes shows in git
    assert _between(first, 'GIT_STATUS') == ''
    assert _between(first, 'HEAD_DIFF') == ''
    # nothing leaks into the patch either
    assert (run / 'tasks' / iid / 'patch.diff').read_text().strip() == ''
    log = run / 'tasks' / iid / 'env_fixes.log'
    if env_fixes:
        assert Version(dist) < Version('0.49'), first  # fastapi snapshot: starlette<0.49
        assert 'resolve_from_wheels' in log.read_text() and 'exit=0' in log.read_text()
    else:
        assert Version(dist) >= Version('1.0'), first  # swegemma injects the newest wheel (1.6.0)
        assert not log.exists()
