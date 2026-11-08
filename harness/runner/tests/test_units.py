"""Fast tests: no docker, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.runner import core, export_sft, mock_openai, report, worker

from .conftest import DATA, FIXTURE_SUB, SAMPLE_SUB


def _rows():
    if not (DATA / 'tasks.jsonl').exists():
        pytest.skip('tasks.jsonl missing')
    return core.load_task_rows(DATA / 'tasks.jsonl')


def test_parse_gold_patches_cover_every_file():
    for r in _rows():
        files = mock_openai.parse_patch(r['patch'])
        assert files, r['instance_id']
        for f in files:
            assert f['deleted'] or f['hunks'], (r['instance_id'], f['path'])
            for old, new in f['hunks']:
                assert old != new


def test_budgets_follow_nb1(tmp_path: Path):
    (tmp_path / 'eval_config.yaml').write_text('evaluation:\n  max_tool_calls: 28\n  max_llm_calls: 80\n')
    b = core.load_budgets(tmp_path)
    assert (b.max_tool_calls, b.max_turns, b.max_time_minutes, b.timeout_seconds) == (28, 80, 60.0, 300)
    (tmp_path / 'eval_config.yaml').write_text('timeout_seconds: 240\nmax_time_minutes: 8\n')  # root dict form
    b = core.load_budgets(tmp_path, {'max_turns': '12'})
    assert (b.timeout_seconds, b.max_time_minutes, b.max_tool_calls, b.max_turns) == (240, 8.0, 100, 12)
    assert b.overrides == {'max_turns': {'from': None, 'to': 12}}
    (tmp_path / 'eval_config.yaml').unlink()
    assert core.load_budgets(tmp_path).source.startswith('defaults')


def test_sample_submission_validates():
    if not SAMPLE_SUB.exists():
        pytest.skip('sample_submission missing')
    info = core.validate_submission(SAMPLE_SUB)
    assert info['declared_model'] == 'gemma-4-31b-it-qat-w4a16-ct'
    assert set(info['adapters']) == {'main_lora', 'tool_lora'}
    assert core.validate_submission(FIXTURE_SUB)['adapters'] == {}


def test_subsets():
    ids = [r['instance_id'] for r in _rows()]
    dev = core.resolve_task_ids('dev40', ids)
    assert len(dev) == 40 and len(set(dev)) == 40
    repos = {r['instance_id']: r['repo'] for r in _rows()}
    assert {repos[i] for i in dev} == set(repos.values())
    assert core.resolve_task_ids('all129', ids) == ids
    with pytest.raises(ValueError):
        core.resolve_task_ids('nope_1', ids)


def test_prepare_submission_zip(tmp_path: Path):
    import shutil
    z = shutil.make_archive(str(tmp_path / 'sub'), 'zip', root_dir=FIXTURE_SUB)
    run = tmp_path / 'run'
    run.mkdir()
    dest = core.prepare_submission(Path(z), run)
    assert (dest / 'agent.yaml').is_file() and (dest / 'eval_config.yaml').is_file()


def test_classify_termination():
    assert worker.classify_termination('Agent exceeded session timeout (5 min)', False, 'x', 0) == 'time_budget'
    assert worker.classify_termination(None, True, 'x', 0) == 'submitted'
    assert worker.classify_termination('Agent completed execution without calling submit_patch.', False, '', 3) == 'no_submit_no_diff'
    assert worker.classify_termination(None, False, 'diff', 3) == 'nudges_exhausted'
    # audit-2 N7: a submit followed by a sandbox execution error stays 'submitted', the exception is flagged
    sbx = 'AgentException: Sandbox execution error: container gone'
    assert worker.classify_termination(sbx, True, 'diff', 0) == 'submitted'
    assert worker.agent_exception_after_submit(sbx, True) is True
    assert worker.classify_termination(sbx, False, '', 0) == 'agent_exception'
    assert worker.agent_exception_after_submit(sbx, False) is False
    assert worker.agent_exception_after_submit(None, True) is False


def _report_run(tmp_path: Path, setups: list[float | None]) -> Path:
    run = tmp_path / 'run'
    ids = [f't{i}' for i in range(len(setups))]
    (run / 'tasks').mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({'name': 'r', 'task_ids': ids, 'timing_scale': 1.0}))
    for iid, setup in zip(ids, setups):
        d = run / 'tasks' / iid
        d.mkdir()
        d.joinpath('result.json').write_text(json.dumps({
            'status': 'done', 'total_tokens': 200_000, 'patch_submitted': True, 'termination': 'submitted',
            'agent_exception_after_submit': iid == 't0',
            'timings': {'agent_s': 300.0, 'setup_s': setup, 'total_s': 300.0 + (setup or 0)}}))
    return run


def test_projection_includes_setup(tmp_path: Path):
    """audit-2 N4: the 120-task projection is printed with and without per-task sandbox setup."""
    s = report.build(_report_run(tmp_path, [60.0, 120.0]))
    kp = s['kaggle_projection']
    assert kp['projected_hours_for_120_tasks'] == 6.0            # 180 s/task (200k tokens x 900 s/Mtok)
    assert kp['projected_hours_for_120_tasks_incl_setup'] == 9.0  # + 90 s measured setup
    assert kp['scaled_hours_for_120_tasks'] == 10.0 and kp['scaled_hours_for_120_tasks_incl_setup'] == 13.0
    assert kp['setup_s_per_task_used'] == 90.0 and kp['setup_basis'].startswith('measured')
    assert s['agent_exceptions_after_submit'] == 1
    md = (tmp_path / 'run' / 'summary.md').read_text()
    assert '9.0 h incl. setup' in md and '13.0 h incl. setup' in md and 'measured' in md
    assert json.loads((tmp_path / 'run' / 'summary.json').read_text())['kaggle_projection'] == kp


def test_projection_assumes_setup_when_unmeasured(tmp_path: Path):
    kp = report.build(_report_run(tmp_path, [None]))['kaggle_projection']
    assert kp['setup_s_per_task_used'] == report.ASSUMED_SETUP_S and kp['setup_basis'].startswith('assumed')
    assert kp['projected_hours_for_120_tasks_incl_setup'] == round((180 + report.ASSUMED_SETUP_S) * 120 / 3600, 2)


def test_tool_error_classification():
    entries = [
        {'type': 'tool_call', 'tool': 'edit_file'},
        {'type': 'tool_response', 'tool': 'edit_file', 'result': json.dumps(
            {'error': 'Invoking `edit_file()` failed as the following mandatory input parameters are not present:\nnew_string'})},
        {'type': 'tool_response', 'tool': 'edit_file', 'result': json.dumps(
            {'status': 'error', 'error_type': 'FileEditError', 'error_message': 'old_string not found'})},
        {'type': 'tool_response', 'tool': 'run_command', 'result': json.dumps({'status': 'ok', 'stdout': ''})},
        {'type': 'text', 'content': 'oops <|tool_call>call:read_file{'},
        {'type': 'continuation_nudge'}, {'type': 'compaction'},
    ]
    m = worker.trace_metrics(entries)
    assert m['tool_errors'] == {'edit_file': {'MissingArguments': 1, 'FileEditError': 1}}
    assert (m['unparsed_tool_call_text'], m['nudges'], m['compactions']) == (1, 1, 1)


def test_mock_summarizer_and_lora_listing():
    st = mock_openai.MockState(None, 'oracle', 'base', ['main_lora'], 0)
    resp = mock_openai.respond(st, {'messages': [{'role': 'user', 'content': mock_openai.SUMMARIZER_MARKER + ' ...'}]})
    assert resp['choices'][0]['message']['content'].startswith('Summary')




def test_export_sft_dedupes_prefix_calls(tmp_path: Path):
    run = tmp_path / 'run'
    (run / 'tasks' / 't1').mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({'name': 'r', 'task_ids': ['t1']}))
    u = {'role': 'user', 'content': 'fix it'}
    a1 = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'c1', 'type': 'function',
                                                               'function': {'name': 'read_file', 'arguments': '{}'}}]}
    t1 = {'role': 'tool', 'tool_call_id': 'c1', 'content': '{}'}

    def call(i, msgs, msg):
        return {'call_index': i, 'kind': 'agent', 'request': {'messages': msgs, 'tools': [], 'params': {}},
                'response': {'choices': [{'message': msg, 'finish_reason': 'stop'}]}}
    calls = [call(1, [u], a1), call(2, [u, a1, t1], {'role': 'assistant', 'content': 'done'})]
    (run / 'tasks' / 't1' / 'llm_calls.jsonl').write_text('\n'.join(json.dumps(c) for c in calls) + '\n')
    out = tmp_path / 'sft.jsonl'
    assert export_sft.export(run, out) == 1
    sample = json.loads(out.read_text())
    assert [m['role'] for m in sample['messages']] == ['user', 'assistant', 'tool', 'assistant']


def test_load_grades_accepts_grader_rows(tmp_path: Path):
    p = tmp_path / 'grades.jsonl'
    p.write_text(json.dumps({'instance_id': 'a', 'passed': True}) + '\n' + json.dumps({'id': 'b', 'resolved': 'false'}) + '\n')
    assert report.load_grades(p) == {'a': True, 'b': False}


def test_health_check_passes_against_mock(mock_server):
    import argparse

    from harness.serve import health_check
    endpoint, _ = mock_server('oracle', loras=['main_lora'])
    c = health_check.run(argparse.Namespace(endpoint=endpoint, api_key='EMPTY', model=None, long=True, bench=2,
                                            allow_short=False))
    assert c.ok, c.results
    assert {r['check'] for r in c.results} >= {'tool_call', 'reasoning_parsed', 'no_think', 'tool_turn', 'lora:main_lora'}


def test_serve_command_matches_scorer_flags():
    if not SAMPLE_SUB.exists():
        pytest.skip('sample_submission missing')
    from harness.serve import serve
    ap_args = ['--model-dir', '/m/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2', '--vllm-python', 'py',
               '--submission', str(SAMPLE_SUB), '--print']
    import argparse
    ns = argparse.Namespace(model_dir=ap_args[1], alias=None, vllm_python='py', profile='3090', submission=str(SAMPLE_SUB),
                            gpu_memory_utilization=None, max_model_len=32768, host='127.0.0.1', port=8000, extra=None)
    cmd, env, devs = serve.build(ns)
    s = ' '.join(cmd)
    for flag in ('--max-model-len 32768', '--dtype bfloat16', '--enable-auto-tool-choice', '--tool-call-parser gemma4',
                 '--reasoning-parser gemma4', '--no-scheduler-reserve-full-isl', '--enable-lora', '--max-loras 2',
                 '--max-lora-rank 8', '"reasoning_start_str": "<|channel>"'):
        assert flag in s, flag
    assert '--tensor-parallel-size' not in s and '--served-model-name' not in s
    assert env['VLLM_WORKER_MULTIPROC_METHOD'] == 'spawn' and any('max-num-seqs' in d for d in devs)
    ns.model_dir = '/m/gemma-4-12b-it-qat-w4a16-ct/2'
    ns.profile, ns.submission = 'small', None
    cmd12 = ' '.join(serve.build(ns)[0])
    assert 'Gemma4ForCausalLM' in cmd12 and '--limit-mm-per-prompt' not in cmd12 and '--enable-lora' not in cmd12


def test_serve_env_scorer_wins_over_ambient():
    """audit-2 S2: NB1 env overrides pre-exported values by default and every difference is listed."""
    from harness.serve import serve
    ambient = {'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128', 'TRANSFORMERS_OFFLINE': '1',
               'OPENAI_API_KEY': 'not-a-real-key-value', 'HOME': '/h'}
    env, devs = serve.nb1_env(ambient)
    assert {k: env[k] for k in serve.NB1_SERVER_ENV} == serve.NB1_SERVER_ENV and env['HOME'] == '/h'
    assert len(devs) == 2 and all(d.startswith('env ') and 'overridden' in d for d in devs)   # equal value: no note
    assert not any('not-a-real-key-value' in d for d in devs) and any('OPENAI_API_KEY' in d for d in devs)
    env, devs = serve.nb1_env(ambient, keep_ambient=True)
    assert env['PYTORCH_CUDA_ALLOC_CONF'] == 'max_split_size_mb:128' and env['OPENAI_API_KEY'] == 'not-a-real-key-value'
    assert len(devs) == 2 and all('kept' in d for d in devs) and not any('not-a-real-key-value' in d for d in devs)
    assert serve.nb1_env({}) == (dict(serve.NB1_SERVER_ENV), [])


def test_serve_build_lists_ambient_env_deviation(monkeypatch):
    import argparse

    from harness.serve import serve
    monkeypatch.setenv('VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS', '0')
    ns = argparse.Namespace(model_dir='/m/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2', alias=None,
                            vllm_python='py', profile='3090', submission=None, gpu_memory_utilization=None,
                            max_model_len=32768, host='127.0.0.1', port=8000, extra=None)
    _cmd, env, devs = serve.build(ns)
    assert env['VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS'] == '1'
    assert any('VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS' in d and 'overridden' in d for d in devs)
    ns.keep_ambient_env = True
    _cmd, env, devs = serve.build(ns)
    assert env['VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS'] == '0'
    assert any('VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS' in d and 'kept' in d for d in devs)
