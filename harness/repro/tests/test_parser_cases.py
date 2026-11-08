"""Tests for harness/repro/parser_cases.py (no vllm needed; real-parser tests use extracted sources if present)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from harness.repro import parser_cases as pc

SRC = Path(__file__).resolve().parents[3] / 'data' / 'vllm-src' / '0.19.1'
needs_src = pytest.mark.skipif(not (SRC / 'vllm' / 'tool_parsers' / 'gemma4_tool_parser.py').exists(),
                               reason='extracted vllm sources not in data/vllm-src/0.19.1')


def _ns(**kw) -> argparse.Namespace:
    base = dict(fake=False, vllm_src=None, tokenizer=None, chunk=3)
    base.update(kw)
    return argparse.Namespace(**base)


def _fake_env() -> pc.Env:
    return pc.load_env(_ns(fake=True))


def test_cases_are_unique_and_cover_every_tool_and_family():
    cases = pc.build_cases()
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))
    for fam in ('doc3dq', 'doc3sq', 'quotes_mixed', 'escapes', 'unicode', 'jsonlike', 'markers', 'long'):
        for tool in ('edit_file', 'write_file', 'run_command'):
            assert any(i.startswith(f'{fam}.') and i.endswith(f'.{tool}') for i in ids), (fam, tool)
    groups = {c.group for c in cases}
    assert {'multi', 'thinking', 'terminator', 'delimiter', 'collision', 'long'} <= groups
    assert any(len(c.calls) >= 3 for c in cases) and any(c.thinking for c in cases)


def test_render_call_is_gemma_native_format():
    raw = pc.render_call('edit_file', {'filepath': 'a.py', 'allow_multiple': True, 'n': 3, 'old_string': 'x"""'})
    assert raw == ('<|tool_call>call:edit_file{filepath:<|"|>a.py<|"|>,allow_multiple:true,n:3,'
                   'old_string:<|"|>x"""<|"|>}<tool_call|>')


def test_fake_parser_roundtrips_canonical_cases():
    env = _fake_env()
    rows = [pc.run_case(env, c, streaming=False) for c in pc.build_cases() if c.family == 'canonical'
            and not c.id.startswith('long.')]
    bad = [(r['id'], r['status'], r.get('detail', '')[:200]) for r in rows if r['status'] != 'OK']
    # the fake reader is simpler than vLLM's: it is allowed to fail only on nested/JSON-ish cases it ignores
    assert not bad, bad


def test_fake_parser_reproduces_r1_signature_and_adk_rejection():
    env = _fake_env()
    case = next(c for c in pc.build_cases() if c.id == 'glitch.H1.extra_delim_after_trailing_quote')
    row = pc.run_case(env, case, streaming=False)
    assert row['status'] == 'MANGLED'
    assert '"<|\\"|>,filepath"' in row['detail']
    assert row['adk'] == ['REJECT: mandatory input parameters are not present: filepath']
    assert pc.classify(row) == 'as expected'


def test_judge_distinguishes_ok_mangled_not_parsed_and_exception():
    case = pc.Case('t', 'canonical', 'g', [('run_command', {'command': 'ls'})])
    ok = pc.Parsed(None, None, [('run_command', {'command': 'ls'})], True, [], [])
    assert pc.judge(case, ok)[0] == 'OK'
    bad = pc.Parsed(None, None, [('run_command', {'command': 'ls '})], True, [], [])
    status, detail = pc.judge(case, bad)
    assert status == 'MANGLED' and '-   "command": "ls"' in detail and '+   "command": "ls "' in detail
    none = pc.Parsed(None, '<|tool_call>call:run_command{', [], False, [], [])
    assert pc.judge(case, none)[0] == 'NOT_PARSED'
    exc = pc.Parsed(None, None, [], False, [], ['Error extracting tool calls [ValueError: boom]'])
    assert pc.judge(case, exc)[0] == 'EXCEPTION'


def test_flawed_parsers_are_reported_not_crashed():
    class Unescaping(pc.FakeToolParser):
        def extract_tool_calls(self, model_output, request):
            return super().extract_tool_calls(model_output.replace('\\', '/'), request)

    class Raising(pc.FakeToolParser):
        def extract_tool_calls(self, model_output, request):
            raise RuntimeError('parser exploded')

    case = next(c for c in pc.build_cases() if c.id == 'escapes.backslash_path.edit_file')
    env = _fake_env()
    env.tool_cls = Unescaping
    assert pc.run_case(env, case, streaming=False)['status'] == 'MANGLED'
    env.tool_cls = Raising
    row = pc.run_case(env, case, streaming=False)
    assert row['status'] == 'EXCEPTION' and 'parser exploded' in row['detail']
    assert pc.classify(row) == 'REGRESSION'


def test_streaming_simulation_matches_for_fake():
    env = _fake_env()
    case = next(c for c in pc.build_cases() if c.id == 'multi.edit_write_run')
    row = pc.run_case(env, case, streaming=True)
    assert row['status'] == 'OK' and row['stream'] == 'OK'


def test_adk_verdicts():
    assert pc.adk_verdict('edit_file', {'filepath': 'a', 'old_string': 'b', 'new_string': 'c'}) == 'ok'
    assert pc.adk_verdict('edit_file', {'<|"|>,filepath': 'a', 'old_string': 'b', 'new_string': 'c'}).startswith(
        'REJECT: mandatory input parameters are not present: filepath')
    assert 'silently drops' in pc.adk_verdict('run_command', {'command': 'ls', 'cwd': '/'})
    assert pc.adk_verdict('edit_files', {}).startswith("REJECT: unknown tool")


def test_cli_without_vllm_fails_clearly(capsys):
    rc = pc.main(['--filter', 'multi.same_call_twice'])
    err = capsys.readouterr().err
    assert rc == 2 and 'cannot import vllm' in err and '--vllm-src' in err


def test_cli_fake_writes_reports(tmp_path, capsys):
    rc = pc.main(['--fake', '--filter', 'multi.', '--filter', 'glitch.H1', '--json', str(tmp_path / 'r.json'),
                  '--md', str(tmp_path / 'r.md')])
    assert rc == 0
    data = json.loads((tmp_path / 'r.json').read_text())
    assert data['needs_windows_run'] is True and data['rows']
    assert 'NEEDS WINDOWS RUN' in (tmp_path / 'r.md').read_text()
    capsys.readouterr()


@needs_src
def test_real_parser_source_canonical_roundtrip_and_r1_signature():
    env = pc.load_env(_ns(vllm_src=str(SRC)))
    assert env.info['source'].startswith('extracted-source')
    keep = [c for c in pc.build_cases() if not c.id.startswith('long.')]
    rows = [pc.run_case(env, c, streaming=True) for c in keep]
    regress = [r['id'] for r in rows if pc.classify(r) == 'REGRESSION']
    assert not regress, regress
    h1 = next(r for r in rows if r['id'] == 'glitch.H1.extra_delim_after_trailing_quote')
    assert h1['status'] == 'MANGLED' and '"<|\\"|>,filepath"' in h1['detail']
    assert h1['adk'][0].startswith('REJECT')
    # server-side facts the doc relies on
    assert pc.run_case(env, next(c for c in keep if c.id == 'glitch.H3.turn_end_instead_of_tool_call_end'),
                       streaming=False)['status'] == 'NOT_PARSED'


def test_template_history_roundtrip_check():
    env = _fake_env()
    case = next(c for c in pc.build_cases() if c.id == 'doc3dq.ends_with_quotes.edit_file')
    assert pc.template_check(env, case) == ('OK', '')

    class BadTemplate(type(env.tokenizer)):
        def apply_chat_template(self, messages, **kw):  # a template that JSON-escapes quotes
            return super().apply_chat_template(messages, **kw).replace('"', '\\"')

    env.tokenizer = BadTemplate()
    status, detail = pc.template_check(env, case)
    assert status == 'MANGLED' and 'rendered:' in detail
    row = pc.run_case(env, case, streaming=False, template=True)
    assert row['template'] == 'MANGLED'
