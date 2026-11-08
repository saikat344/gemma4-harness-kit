"""P-E5 serializer replay (harness/analysis/pe5.py): window rebuild + validity check, the vendored 2.2.0
formatter, fact extraction, sampling, and prepare -> run (mock server) -> report. No GPU, no Docker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.analysis import adk220_format, pe5

REPO = Path(__file__).resolve().parents[3]
REAL = REPO / 'runs' / 'pe5' / 'source'
AGENT = 'swe_test_agent'
SEP = '\\n'  # 1.36.1 joins entries with a literal backslash-n
LONG = 'x' * 2500


def _turn(k: int) -> tuple[dict, dict]:
    """Agent turn k: thought + (on even k) visible text + one tool call; and its tool result."""
    cid = f'call-{k}'
    if k == 3:
        name, args, result = 'edit_file', {'filepath': '/workspace/pkg/core.py', 'old_string': 'a', 'new_string': 'b'}, \
            {'status': 'ok', 'filepath': 'pkg/core.py', 'diff': '-a\n+b'}
    elif k == 4:
        name, args, result = 'run_command', {'command': 'pytest -q'}, \
            {'status': 'ok', 'stdout': ('mentions OSError mid-line\r\nE   KeyError: 1\r\n'
                                        'pkg.errors.ParseError: bad\n    raise LookupError(x)\n'
                                        'FAILED tests/test_core.py::test_empty - KeyError: 1\n' + LONG),
             'error_type': 'FileEditError'}
    else:
        name, args, result = 'read_file', {'filepath': f'pkg/mod{k}.py'}, {'status': 'ok', 'content': f'class Bad{k}Error(Exception):\n    pass\nraise Bad{k}Error()\nValueError: no'}
    asst = {'role': 'assistant', 'content': f'Visible note {k}.' if k % 2 == 0 else None,
            'reasoning_content': f'Thinking about step {k}.',
            'tool_calls': [{'type': 'function', 'id': cid, 'function': {'name': name, 'arguments': json.dumps(args)}}]}
    tool = {'role': 'tool', 'tool_call_id': cid, 'content': json.dumps(result)}
    return asst, tool


def make_task(run_dir: Path, iid: str, problem: str, n: int = 8, tamper: bool = False) -> Path:
    """n agent calls, then the summarizer (call n+1), then one more agent call that carries the retained tail."""
    d = run_dir / 'tasks' / iid
    d.mkdir(parents=True, exist_ok=True)
    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': f'Problem Statement:\n{problem}'}
    turns = [_turn(k) for k in range(1, n + 1)]
    calls = []
    for k in range(1, n + 1):
        msgs = [system, user] + [m for a, t in turns[:k - 1] for m in (a, t)]
        calls.append({'call_index': k, 'kind': 'agent', 'request': {'model': 'm', 'messages': msgs, 'tools': [{}],
                      'params': {'temperature': 1.0}}, 'response': {'choices': [{'message': turns[k - 1][0]}]},
                      'usage': {'prompt_tokens': 15000}})
    # expected window (independent of pe5): split moves from 2n-4 to 2n-5, i.e. user + turns 1..n-3
    entries = [f'user: {user["content"]}']
    for k in range(1, n - 2):
        a = turns[k - 1][0]
        entries.append(f'{AGENT}: {a["reasoning_content"]}')
        if a['content']:
            entries.append(f'{AGENT}: {a["content"]}')
    hist = SEP.join(entries)
    if tamper:
        hist += SEP + f'{AGENT}: an extra line'
    prompt = pe5.template_136().format(conversation_history=hist)
    calls.append({'call_index': n + 1, 'kind': 'summarizer',
                  'request': {'model': 'm', 'messages': [{'role': 'user', 'content': prompt}], 'tools': None,
                              'params': {'api_base': 'http://secret', 'response_format': None,
                                         'extra_body': {'chat_template_kwargs': {'enable_thinking': True},
                                                        'thinking_token_budget': 4096}}},
                  'response': {'choices': [{'message': {'content': 'Logged summary: pkg/core.py was edited.',
                                                        'reasoning_content': 'hmm'}}]},
                  'usage': {'prompt_tokens': 900, 'completion_tokens': 50}, 'latency_s': 3.0})
    tail = [m for a, t in turns[n - 3:] for m in (a, t)]
    calls.append({'call_index': n + 2, 'kind': 'agent', 'request': {'model': 'm', 'messages': [
        system, {'role': 'assistant', 'content': 'Logged summary'}] + tail, 'tools': [{}], 'params': {}},
        'response': {'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]}})
    (d / 'llm_calls.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in calls))
    (d / 'compactions.jsonl').write_text(json.dumps({'after_llm_call_index': n + 1, 'events_in_window': 2 * n - 5,
                                                     'session_events_total': 2 * n + 2, 'summary': 'x'}) + '\n')
    (d / 'result.json').write_text('{}')
    return d


def test_vendored_220_formatter_matches_upstream_behaviour():
    from google.adk.events.event import Event
    from google.adk.events.event_actions import EventActions, EventCompaction
    from google.genai import types

    fc = types.Part.from_function_call(name='read_file', args={'filepath': 'a.py'})
    fr = types.Part(function_response=types.FunctionResponse(name='read_file', response={'content': 'y' * 2100}))
    evs = [
        Event(author='user', content=types.Content(role='user', parts=[types.Part.from_text(text='fix it')])),
        Event(author='ag', content=types.Content(role='model', parts=[types.Part(text='think', thought=True),
                                                                      types.Part.from_text(text='say'), fc])),
        Event(author='ag', content=types.Content(role='user', parts=[fr])),
        Event(author='user', actions=EventActions(compaction=EventCompaction(
            start_timestamp=1, end_timestamp=2, compacted_content=types.Content(role='model', parts=[]))),
              content=types.Content(role='model', parts=[types.Part(text='old thought', thought=True),
                                                         types.Part.from_text(text='old summary')])),
    ]
    out = adk220_format.format_events_for_prompt(evs).split('\n')
    resp = str({'content': 'y' * 2100})
    assert out == ['user: fix it', 'ag (thought): think', 'ag: say', "ag called tool: read_file({'filepath': 'a.py'})",
                   f'Tool response from read_file: {resp[:2000]}... [truncated {len(resp) - 2000} chars]',
                   'user: old summary']
    # the 1.36.1 formatter on the same events keeps text (incl. thoughts) only
    assert pe5.format_136(evs) == SEP.join(['user: fix it', 'ag: think', 'ag: say', 'user: old thought',
                                            'user: old summary'])


def test_reconstruct_reproduces_logged_history(tmp_path):
    d = make_task(tmp_path / 'runA', 'pkg_1', 'Fix `parse_value` in pkg/core.py raising KeyError.', n=8)
    r = pe5.reconstruct(d)
    assert r['valid'] and r['exact_match'], r['reason']
    assert r['agent_name'] == AGENT
    assert r['n_events'] == 2 * 8 + 1 and r['n_window'] == 2 * 8 - 5  # split moved so call 6 keeps its response
    assert r['tail_source'] == 'next_request' and r['last_agent_call_index'] == 8
    assert r['params'] == {'extra_body': {'chat_template_kwargs': {'enable_thinking': True}, 'thinking_token_budget': 4096}}
    y = pe5.format_220(r['_window'])
    assert f"{AGENT} called tool: edit_file(" in y and 'Tool response from run_command:' in y
    assert '... [truncated' in y and 'FAILED tests/test_core.py::test_empty' in y
    assert 'Tool response' not in r['_hist_T']


def test_reconstruct_reports_mismatch(tmp_path):
    d = make_task(tmp_path / 'runA', 'pkg_1', 'Problem', n=8, tamper=True)
    r = pe5.reconstruct(d)
    assert not r['valid'] and 'history mismatch' in r['reason']


def test_reconstruct_without_next_call_uses_placeholder(tmp_path):
    d = make_task(tmp_path / 'runA', 'pkg_1', 'Problem', n=8)
    lines = (d / 'llm_calls.jsonl').read_text().splitlines()[:-1]  # the run died right after compaction
    (d / 'llm_calls.jsonl').write_text('\n'.join(lines) + '\n')
    r = pe5.reconstruct(d)
    assert r['valid'] and r['tail_source'] == 'placeholder'


def test_issue_facts_priority_and_filters():
    ps = ('Calling `Foo.bar()` with `None` fails; see https://x.org/a/b.py and `get_type_hints`.\n'
          'typing.get_type_hints raises TypeError in pkg/mod/util.py via HTTPParser. `ok`\n'
          '```\nIgnored `inside` fence\n```')
    # backticks first (`None` is a builtin, `ok` too short, fenced spans ignored), then dotted/CamelCase by position
    # (`Foo.bar` and `typing.get_type_hints` overlap chosen facts; x.org/a/b.py was a URL), then paths
    assert pe5.issue_facts(ps, 'org/pkg') == ['Foo.bar()', 'get_type_hints', 'TypeError', 'HTTPParser',
                                              'pkg/mod/util.py']


def test_tool_facts_categories(tmp_path):
    d = make_task(tmp_path / 'runA', 'pkg_1', 'P', n=9)
    r = pe5.reconstruct(d)
    facts, cats = pe5.tool_facts(r['_window'])
    assert cats['last_edit_target'] == ['pkg/core.py']
    assert cats['failing_tests'] == ['tests/test_core.py::test_empty']
    # only exceptions raised in run_command output: `E   ` prefix, line start (module-qualified), after `raise`;
    # not a mid-line mention (OSError), not error_type (FileEditError), not read_file source (Bad1Error, ValueError)
    assert cats['exceptions'] == ['KeyError', 'ParseError', 'LookupError']
    assert facts == ['pkg/core.py', 'tests/test_core.py::test_empty', 'KeyError', 'pkg/mod1.py', 'ParseError']


def test_stratified_sample_is_deterministic_and_covers_repos():
    by_task = {f'{r}_{i}': ['A', 'B'] if i % 2 else ['A'] for r, n in (('a', 15), ('b', 12), ('c', 2), ('d', 1))
               for i in range(n)}
    repos = {t: t.split('_')[0] for t in by_task}
    s1 = pe5.stratified_sample(by_task, repos, 20, 20261103)
    assert s1 == pe5.stratified_sample(by_task, repos, 20, 20261103)
    assert len(s1) == 20 and len({t for t, _ in s1}) == 20
    from collections import Counter
    c = Counter(repos[t] for t, _ in s1)
    assert set(c) == {'a', 'b', 'c', 'd'} and c['a'] >= c['b'] >= c['c'] >= c['d']
    small = {'a_1': ['A'], 'b_1': ['B']}
    assert pe5.stratified_sample(small, {'a_1': 'a', 'b_1': 'b'}, 20) == [('a_1', 'A'), ('b_1', 'B')]


def test_prepare_run_report_against_mock(tmp_path, mock_server, monkeypatch):
    runs = []
    for run in ('runA', 'runB'):
        rd = tmp_path / run
        for i, n in enumerate((8, 9, 10)):
            make_task(rd, f'pkg_{i}', f'Fix `parse_value{i}` in pkg/core.py.', n=n)
        runs.append(str(rd))
    tasks = tmp_path / 'tasks.jsonl'
    tasks.write_text(''.join(json.dumps({'instance_id': f'pkg_{i}', 'repo': 'org/pkg',
                                         'problem_statement': f'Fix `parse_value{i}` in pkg/core.py.'}) + '\n'
                             for i in range(3)))
    out = tmp_path / 'pe5'
    assert pe5.main(['prepare', '--runs', *runs, '--tasks', str(tasks), '--out', str(out)]) == 0
    prep = json.loads((out / 'prepare.json').read_text())
    assert prep['validity_all'] == {'valid': 6, 'total': 6, 'invalid': {}}
    assert prep['validity_sample']['total'] == 3  # fewer than 20 distinct tasks -> all of them
    facts = pe5.read_jsonl(out / 'facts.jsonl')
    assert [f['issue_facts'] for f in facts] == [[f'parse_value{i}', 'pkg/core.py'] for i in range(3)]
    reqs = pe5.read_jsonl(out / 'requests.jsonl')
    assert len(reqs) == 6 and {q['arm'] for q in reqs} == {'T', 'Y'}
    assert all('api_base' not in q['params'] for q in reqs)
    y = next(q for q in reqs if q['arm'] == 'Y')
    assert y['est_prompt_tokens'] > 900 and y['history_chars'] > next(q for q in reqs if q['arm'] == 'T')['history_chars']

    endpoint, state = mock_server('oracle')
    monkeypatch.setenv('VLLM_API_KEY', 'test-key')
    assert pe5.main(['run', '--endpoint', endpoint, '--out', str(out), '--samples', '2', '--limit', '5']) == 0
    assert state.requests == 5
    assert pe5.main(['run', '--endpoint', endpoint.removesuffix('/v1'), '--out', str(out), '--samples', '2']) == 0
    assert state.requests == 12
    assert pe5.main(['run', '--endpoint', endpoint, '--out', str(out), '--samples', '2']) == 0  # resumable: nothing left
    assert state.requests == 12
    assert all(e['extra'].get('chat_template_kwargs') == {'enable_thinking': True}
               and e['extra'].get('thinking_token_budget') == 4096 and not e['has_tools'] for e in state.log)
    calls = pe5.read_jsonl(out / 'calls.jsonl')
    assert len(calls) == 12 and all(c['content'].startswith('Summary:') for c in calls)
    assert 'test-key' not in (out / 'calls.jsonl').read_text()

    doc = tmp_path / 'pe5-results.md'
    assert pe5.main(['report', '--out', str(out), '--doc', str(doc)]) == 0
    res = json.loads((out / 'results.json').read_text())
    t, yy = res['arms']['T'], res['arms']['Y']
    assert t['calls'] == yy['calls'] == 6
    assert t['issue']['available']['pct'] == 100.0
    # tool facts reach the summarizer only through the typed serializer
    assert t['tool']['available']['x'] < yy['tool']['available']['x'] == yy['tool']['available']['n']
    assert res['arms']['T_logged_ab1']['tool']['retained']['x'] == 3  # 'pkg/core.py' in each logged summary
    cov = res['exploratory_char_coverage']
    assert 'EXPLORATORY' in cov['label']
    # T only covers result lines that also occur in text (here `pkg/core.py`, named in the issue); Y is truncated
    assert cov['T']['tool_results']['x'] < cov['Y']['tool_results']['x'] < cov['Y']['tool_results']['n']
    assert cov['T']['edit_strings']['n'] == 0  # the edit in the fixture has 1-char strings (< 8, not counted)
    md = doc.read_text()
    assert 'GENERATED' in md and '| Y (2.2.0 typed) | 6 |' in md and 'Validity check' in md


@pytest.mark.skipif(not (REAL / 'baseline-v0-A').exists(), reason='real ab1 logs not copied to runs/pe5/source')
def test_real_ab1_first_compactions_all_validate():
    tds = sorted(p.parent for p in REAL.glob('*/tasks/*/compactions.jsonl'))
    assert tds
    bad = {str(td): r['reason'] for td in tds if not (r := pe5.reconstruct(td))['valid']}
    assert not bad, bad
