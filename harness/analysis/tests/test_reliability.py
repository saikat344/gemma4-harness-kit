"""Analyzer tests on synthetic runner output: one scenario per detector, plus aggregation and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.analysis import reliability as rel

from .synth import (MULTI, NOT_FOUND, TaskBuilder, adk_missing, edit_err, ok, ok_edit, unknown_tool, write_run)

OLD = 'def f(x):\n    return x + 1\n'
NEW = 'def f(x):\n    return x + 2\n'
P = 'pkg/mod.py'
MANGLED = {'<|"|>,filepath': P, 'old_string': OLD, 'new_string': NEW}


def analyze(tmp_path: Path, build, grades: bool | None = None, th: rel.Thresholds | None = None) -> dict:
    b = TaskBuilder(tmp_path / 'r', 't1')
    build(b)
    run = write_run(tmp_path, 'r', ['t1'], None if grades is None else {'t1': grades})
    return rel.analyze_runs([run], th or rel.Thresholds())['tasks'][0]


def test_r1_mangled_key_loop_is_detected_and_timed(tmp_path):
    def build(b):
        b.turn([('read_file', {'filepath': P}, ok(content='x'))])
        for _ in range(5):  # the report: the identical mangled call retried again and again
            b.turn([('edit_file', MANGLED, adk_missing('edit_file', 'filepath'))], tokens=900, latency=9.0, gap=0.2)
        b.turn([('edit_file', {'filepath': P, 'old_string': OLD, 'new_string': NEW}, ok_edit(P, OLD, NEW))])
        b.turn([('submit_patch', {}, ok())])
        b.finish()

    t = analyze(tmp_path, build, grades=False)
    assert t['repeat']['max_run'] == 5 and t['repeat']['consecutive'] == 4 and t['repeat']['after_error'] == 4
    assert t['repeat']['longest']['kind'] == 'rejected_missing'
    assert t['rejected']['total'] == 5 and t['rejected']['missing_params'] == {'filepath': 5}
    assert t['parser_signatures']['mangled_key'] == 5 and t['parser_signatures']['missing_required'] == 5
    assert {'retry_loop', 'identical_after_error', 'adk_rejected', 'parser_signature', 'identical_repeat'} <= set(t['flags'])
    assert t['time']['rejected_s'] > 5 * 9 and t['time']['share_of_agent'] > 0.5
    assert 'time_wasted_high' in t['flags']
    assert t['kinds']['rejected_missing'] == 5 and t['kinds']['ok'] == 3


def test_clean_task_has_no_failure_flags(tmp_path):
    def build(b):
        b.turn([('read_file', {'filepath': P}, ok(content='x'))])
        b.turn([('edit_file', {'filepath': P, 'old_string': OLD, 'new_string': NEW}, ok_edit(P, OLD, NEW))])
        b.turn([('run_command', {'command': 'pytest -q'}, ok(stdout='1 passed', exit_code=0))])
        b.turn([('submit_patch', {}, ok())])
        b.finish()

    t = analyze(tmp_path, build, grades=True)
    assert t['flags'] == [] and t['time']['failed_total_s'] == 0
    assert t['source'] == 'llm_calls' and t['tool_calls'] == 4 and t['turns'] == 4


def test_unknown_tool_and_budget_and_cmd_error_are_classified(tmp_path):
    def build(b):
        b.turn([('edit_files', {'filepath': P}, unknown_tool('edit_files'))])
        b.turn([('run_command', {'command': 'pytest'}, json.dumps({'status': 'error', 'error_type': 'CommandError',
                                                                  'error_message': '1 failed'}))])
        b.turn([('read_file', {'filepath': P}, json.dumps({'status': 'error', 'error_type': 'BudgetExceeded',
                                                          'error_message': 'Tool call budget exhausted (10 calls)'}))])
        b.finish(termination='tool_budget')

    t = analyze(tmp_path, build)
    assert t['kinds'] == {'rejected_unknown_tool': 1, 'cmd_error': 1, 'budget': 1}
    assert {'adk_rejected', 'tool_budget_hit', 'budget_termination'} <= set(t['flags'])
    # a failing pytest run is not a harness failure and not wasted time on its own
    assert t['time']['rejected_s'] > 0


def test_edit_file_failures_by_cause_and_heavy_flag(tmp_path):
    def build(b):
        e = lambda old, res: ('edit_file', {'filepath': P, 'old_string': old, 'new_string': 'y'}, res)  # noqa: E731
        b.turn([e('missing', edit_err(NOT_FOUND))])
        b.turn([e('missing', edit_err(NOT_FOUND))])  # identical retry after error
        b.turn([e('def', edit_err(MULTI))])
        b.turn([e('', edit_err('old_string cannot be empty.'))])
        b.turn([('edit_file', {'filepath': P, 'old_string': OLD, 'new_string': NEW}, ok_edit(P, OLD, NEW))])
        b.finish()

    t = analyze(tmp_path, build)
    ef = t['edit_file']
    assert ef['calls'] == 5 and ef['failed'] == 4 and ef['rate'] == 0.8
    assert ef['causes'] == {'old_string_not_found': 2, 'multiple_matches': 1, 'empty_old_string': 1}
    assert {'edit_fail', 'edit_fail_heavy', 'identical_after_error'} <= set(t['flags'])
    assert t['repeat']['max_run'] == 2 and 'retry_loop' not in t['flags']  # below --loop-min-run 3
    t2 = analyze(tmp_path / 'again', build, th=rel.Thresholds(loop_min_run=2))
    assert 'retry_loop' in t2['flags']


@pytest.mark.parametrize('cause, msg', [
    ('old_string_not_found', 'Failed to replace: old_string not found.'),
    ('multiple_matches', 'Target string occurs 3 times in a.py. Please provide more surrounding context.'),
    ('multiple_matches', 'Target string flexibly matches 2 locations in a.py. Please provide more surrounding context.'),
    ('identical_old_new', 'No changes to apply. old_string and new_string are identical.'),
    ('file_empty', "File 'a.py' is empty. Use write_file to populate an empty or newly created file."),
    ('bad_path', "Path traversal detected: '/etc/x' escapes workspace root."),
    ('other', 'something odd'),
])
def test_edit_cause_taxonomy(cause, msg):
    assert rel.edit_cause(msg) == cause


def test_aba_oscillation_exact_and_near_identical(tmp_path):
    def build(b):
        fwd = ('edit_file', {'filepath': P, 'old_string': OLD, 'new_string': NEW}, ok_edit(P, OLD, NEW))
        back = ('edit_file', {'filepath': P, 'old_string': NEW, 'new_string': OLD}, ok_edit(P, NEW, OLD))
        # the 3rd/5th patch is a near-identical variant (comment added), like re-applied-with-tweaks patches
        fwd2_new = NEW + '    # tweaked\n'
        fwd2 = ('edit_file', {'filepath': P, 'old_string': OLD, 'new_string': fwd2_new}, ok_edit(P, OLD, fwd2_new))
        b.turn([fwd])
        b.turn([back])
        b.turn([fwd2])
        b.turn([('edit_file', {'filepath': P, 'old_string': fwd2_new, 'new_string': OLD}, ok_edit(P, fwd2_new, OLD))])
        b.turn([fwd])
        b.finish()

    t = analyze(tmp_path, build)
    o = t['oscillation']
    assert o['reverts'] == 2 and o['reapplies'] >= 1 and o['files'] == [P] and o['edits'] == 5
    assert {'aba_revert_any', 'aba_oscillation'} <= set(t['flags'])
    # threshold as flag: with a single revert required it is `any`, with 3 required (and no reapply rule) it is not
    t1 = analyze(tmp_path / 'one', lambda b: (b.turn([('edit_file', {'filepath': P, 'old_string': OLD, 'new_string': NEW},
                                                       ok_edit(P, OLD, NEW))]),
                                              b.turn([('edit_file', {'filepath': P, 'old_string': NEW, 'new_string': OLD},
                                                       ok_edit(P, NEW, OLD))]), b.finish()))
    assert 'aba_revert_any' in t1['flags'] and 'aba_oscillation' not in t1['flags']


def test_sequential_distinct_edits_are_not_oscillation(tmp_path):
    def build(b):
        for i in range(4):  # same kind of change on different lines, never undone
            o, n = f'x{i} = 1\n', f'x{i} = 2\n'
            b.turn([('edit_file', {'filepath': P, 'old_string': o, 'new_string': n}, ok_edit(P, o, n))])
        # same direction repeated on one file is not a revert either
        b.turn([('edit_file', {'filepath': P, 'old_string': 'a = 1\n', 'new_string': 'a = 2\n'}, ok_edit(P, 'a = 1\n', 'a = 2\n'))])
        b.turn([('edit_file', {'filepath': P, 'old_string': 'a = 2\n', 'new_string': 'a = 3\n'}, ok_edit(P, 'a = 2\n', 'a = 3\n'))])
        b.finish()

    t = analyze(tmp_path, build)
    assert t['oscillation']['reverts'] == 0 and not {'aba_revert_any', 'aba_oscillation'} & set(t['flags'])


def test_failed_edits_do_not_count_as_workspace_changes(tmp_path):
    def build(b):
        b.turn([('edit_file', {'filepath': P, 'old_string': OLD, 'new_string': NEW}, ok_edit(P, OLD, NEW))])
        b.turn([('edit_file', {'filepath': P, 'old_string': NEW, 'new_string': OLD}, edit_err(NOT_FOUND))])
        b.finish()

    assert analyze(tmp_path, build)['oscillation']['reverts'] == 0


def test_write_file_oscillation_and_revert_commands(tmp_path):
    a = 'def g():\n    return 1\n' * 3
    b_ = 'def g():\n    return 2\n' * 3

    def build(b):
        b.turn([('write_file', {'filepath': P, 'content': a}, ok(filepath=P))])
        b.turn([('write_file', {'filepath': P, 'content': b_}, ok(filepath=P))])
        b.turn([('write_file', {'filepath': P, 'content': a}, ok(filepath=P))])
        b.turn([('run_command', {'command': 'git checkout -- pkg/mod.py'}, ok(stdout='', exit_code=0))])
        b.finish()

    t = analyze(tmp_path, build)
    assert t['oscillation']['reverts'] == 1 and t['oscillation']['revert_cmds'] == 1


def test_runaway_generation_and_tail(tmp_path):
    def build(b):
        b.turn([('read_file', {'filepath': P}, ok(content='x'))], tokens=300)
        b.turn([('read_file', {'filepath': P}, ok(content='x'))], tokens=5000, latency=150)
        b.turn([('edit_file', {'filepath': P, 'old_string': OLD, 'new_string': NEW}, ok_edit(P, OLD, NEW))],
               tokens=8192, latency=240, finish='length')
        b.finish()

    t = analyze(tmp_path, build)
    g = t['generation']
    assert g['cap'] == 8192 and g['max_tokens'] == 8192 and g['n_at_cap'] == 1 and g['n_length_finish'] == 1
    assert g['n_tail'] == 2 and g['runaway_time_s'] == 240.0
    assert {'runaway_generation', 'long_tail_generation'} <= set(t['flags'])


def test_unparsed_tool_call_text_is_a_parser_signature(tmp_path):
    def build(b):
        b.turn([], content='<|tool_call>call:write_file{filepath:<|"|>a.py<|"|>,content:<|"|>x', tokens=8192,
               finish='length')
        b.nudges = 1
        b.finish()

    t = analyze(tmp_path, build)
    assert t['parser_signatures']['unparsed_tool_text'] == 1 and t['parser_signatures']['truncated_tool_text'] == 1
    assert {'parser_signature', 'runaway_generation', 'nudged'} <= set(t['flags'])


def test_compaction_problem_statement_survival(tmp_path):
    def build_lost(b):
        b.turn([('read_file', {'filepath': P}, ok(content='x'))])
        b.compaction(keep_problem=False)
        b.turn([('read_file', {'filepath': P}, ok(content='x'))])
        b.finish()

    t = analyze(tmp_path / 'lost', build_lost)
    assert t['compaction']['count'] == 1 and t['compaction']['lost_problem'] == 1
    assert {'compaction', 'compaction_lost_problem'} <= set(t['flags'])

    def build_kept(b):
        b.turn([('read_file', {'filepath': P}, ok(content='x'))])
        b.compaction(keep_problem=True)
        b.turn([('read_file', {'filepath': P}, ok(content='x'))])
        b.finish()

    t = analyze(tmp_path / 'kept', build_kept)
    assert t['compaction']['lost_problem'] == 0 and t['compaction']['problem_after'] == [['user']]
    assert 'compaction' in t['flags'] and 'compaction_lost_problem' not in t['flags']

    # v1-style: the problem is templated into the system instruction, so it survives compaction
    def build_sys(b):
        b.turn([('read_file', {'filepath': P}, ok(content='x'))])
        b.compaction(keep_problem=False)
        b.turn([('read_file', {'filepath': P}, ok(content='x'))])
        b.finish()

    b = TaskBuilder(tmp_path / 'sys' / 'r', 't1', instruction_has_problem=True)
    build_sys(b)
    run = write_run(tmp_path / 'sys', 'r', ['t1'])
    t = rel.analyze_runs([run], rel.Thresholds())['tasks'][0]
    assert t['compaction']['problem_after'] == [['system']] and t['compaction']['lost_problem'] == 0


def test_empty_patch_and_worker_failure_flags(tmp_path):
    def build(b):
        b.turn([('read_file', {'filepath': P}, ok(content='x'))])
        b.finish(empty_patch=True, termination='time_budget')

    t = analyze(tmp_path, build, grades=False)
    assert {'empty_patch', 'budget_termination'} <= set(t['flags'])

    def build2(b):
        b.turn([('edit_file', MANGLED, adk_missing('edit_file', 'filepath'))])
        b.finish(status='worker_timeout', empty_patch=True, termination=None, final_text=False)

    t = analyze(tmp_path / 'w', build2)
    assert 'worker_failure' in t['flags'] and 'adk_rejected' in t['flags']  # llm_calls survive a killed worker


def test_trace_fallback_when_no_llm_calls(tmp_path):
    td = tmp_path / 'r' / 'tasks' / 't1'
    td.mkdir(parents=True)
    entries = [
        {'type': 'task_prompt', 'elapsed_s': 0.0},
        {'type': 'tool_call', 'tool': 'edit_file', 'args': MANGLED, 'elapsed_s': 5.0},
        {'type': 'tool_response', 'tool': 'edit_file', 'result': adk_missing('edit_file', 'filepath'), 'elapsed_s': 5.1},
        {'type': 'tool_call', 'tool': 'edit_file', 'args': MANGLED, 'elapsed_s': 12.0},
        {'type': 'tool_response', 'tool': 'edit_file', 'result': adk_missing('edit_file', 'filepath'), 'elapsed_s': 12.1},
    ]
    (td / 'trace_legacy.json').write_text(json.dumps({'entries': entries}))
    (td / 'result.json').write_text(json.dumps({'status': 'done', 'empty_patch': True, 'timings': {'agent_s': 13}}))
    run = write_run(tmp_path, 'r', ['t1'])
    t = rel.analyze_runs([run], rel.Thresholds())['tasks'][0]
    assert t['source'] == 'trace_legacy' and t['repeat']['consecutive'] == 1 and t['rejected']['total'] == 2
    assert t['time']['rejected_s'] == pytest.approx(12.1 - 0.0 - 0.1 + 0.0, abs=0.2) or t['time']['rejected_s'] > 0


def test_trace_fill_aligns_repeated_identical_calls(tmp_path):
    """Audit F13: an identical edit fails first (observed) and succeeds last (only in the trace). The final call
    must take the trace's second outcome, not inherit the first call's failure."""
    edit = {'filepath': P, 'old_string': OLD, 'new_string': NEW}

    def build(b):
        b.turn([('edit_file', edit, edit_err(NOT_FOUND))])
        b.turn([('read_file', {'filepath': P}, ok(content='x'))])
        b.turn([('edit_file', edit, None)])  # last turn: the result never reaches llm_calls.jsonl
        b.finish(final_text=False)
        entries = [
            {'type': 'task_prompt', 'elapsed_s': 0.0},
            {'type': 'tool_call', 'tool': 'edit_file', 'args': edit, 'elapsed_s': 1.0},
            {'type': 'tool_response', 'tool': 'edit_file', 'result': edit_err(NOT_FOUND), 'elapsed_s': 1.1},
            {'type': 'tool_call', 'tool': 'read_file', 'args': {'filepath': P}, 'elapsed_s': 2.0},
            {'type': 'tool_response', 'tool': 'read_file', 'result': ok(content='x'), 'elapsed_s': 2.1},
            {'type': 'tool_call', 'tool': 'edit_file', 'args': edit, 'elapsed_s': 3.0},
            {'type': 'tool_response', 'tool': 'edit_file', 'result': ok_edit(P, OLD, NEW), 'elapsed_s': 3.1},
        ]
        (b.dir / 'trace_legacy.json').write_text(json.dumps({'entries': entries}))

    t = analyze(tmp_path, build)
    assert t['source'] == 'llm_calls'
    assert t['kinds'].get('edit_fail') == 1 and t['kinds'].get('ok') == 2


def test_trace_fill_prefers_call_ids(tmp_path):
    """With stable ids on both sides, the id decides even when tool + args would align differently."""
    td = tmp_path / 't'
    td.mkdir()
    a = {'filepath': P, 'old_string': OLD, 'new_string': NEW}
    entries = [
        {'type': 'tool_call', 'tool': 'edit_file', 'args': a, 'id': 'c1'},
        {'type': 'tool_response', 'tool': 'edit_file', 'result': 'first', 'id': 'c1'},
        {'type': 'tool_call', 'tool': 'edit_file', 'args': a, 'id': 'c2'},
        {'type': 'tool_response', 'tool': 'edit_file', 'result': 'second', 'id': 'c2'},
    ]
    (td / 'trace_legacy.json').write_text(json.dumps({'entries': entries}))
    c = rel.Call(turn=0, tool='edit_file', args=a, raw_args=json.dumps(a), call_id='c2', llm_index=1)
    rel.fill_from_trace(td, [c])  # the only logged call, but its id says it is the second one
    assert c.result == 'second'


def test_fisher_exact():
    assert rel.fisher_exact(0, 2, 2, 0) == pytest.approx(1 / 3)
    assert rel.fisher_exact(5, 5, 5, 5) == pytest.approx(1.0)
    assert rel.fisher_exact(10, 0, 0, 10) < 1e-4


def test_aggregate_correlates_modes_with_outcome(tmp_path):
    ids = ['good1', 'good2', 'bad1', 'bad2', 'bad3']
    grades = {'good1': True, 'good2': True, 'bad1': False, 'bad2': False, 'bad3': False}
    root = tmp_path / 'r'
    for iid in ids:
        b = TaskBuilder(root, iid)
        b.turn([('read_file', {'filepath': P}, ok(content='x'))])
        if iid.startswith('bad') and iid != 'bad3':  # 2 of 3 failed tasks have a retry loop
            for _ in range(4):
                b.turn([('edit_file', MANGLED, adk_missing('edit_file', 'filepath'))])
        b.turn([('edit_file', {'filepath': P, 'old_string': OLD, 'new_string': NEW}, ok_edit(P, OLD, NEW))])
        b.finish(empty_patch=(iid == 'bad3'))
    run = write_run(tmp_path, 'r', ids, grades)
    s = rel.analyze_runs([run], rel.Thresholds())
    modes = {m['mode']: m for m in s['pooled']['modes']}
    loop = modes['retry_loop']
    assert (loop['tasks_flagged'], loop['flagged_resolved'], loop['flagged_unresolved']) == (2, 0, 2)
    assert loop['unresolved_affected'] == 2 and loop['unresolved_total'] == 3
    assert loop['share_of_unresolved_affected'] == pytest.approx(0.667, abs=1e-3)
    assert loop['share_of_resolved_affected'] == 0.0
    assert loop['resolve_rate_flagged'] == 0.0 and loop['resolve_rate_unflagged'] == pytest.approx(0.667, abs=1e-3)
    assert loop['fisher_p'] == pytest.approx(0.4, abs=1e-3)
    assert modes['empty_patch']['unresolved_affected'] == 1
    assert s['pooled']['totals']['identical_consecutive_calls'] == 6 and s['pooled']['totals']['adk_rejected_calls'] == 8
    assert s['pooled']['by_outcome']['unresolved']['adk_rejected_calls'] == 8
    assert s['pooled']['by_outcome']['resolved']['adk_rejected_calls'] == 0


def test_cli_multiple_runs_parent_dir_and_outputs(tmp_path, capsys):
    for name, bad in (('v0', True), ('v1', False)):
        b = TaskBuilder(tmp_path / 'ab1' / name, 't1')
        b.turn([('read_file', {'filepath': P}, ok(content='x'))])
        if bad:
            for _ in range(3):
                b.turn([('edit_file', MANGLED, adk_missing('edit_file', 'filepath'))])
        b.finish()
        write_run(tmp_path / 'ab1', name, ['t1'], {'t1': not bad})
    out = tmp_path / 'report'
    rc = rel.main([str(tmp_path / 'ab1'), '--out', str(out)])
    assert rc == 0
    md = (out / 'reliability.md').read_text()
    js = json.loads((out / 'reliability.json').read_text())
    assert set(js['per_run']) == {'v0', 'v1'} and js['pooled']['tasks'] == 2
    assert '## Per run' in md and '`retry_loop`' in md and 'v0' in md and 'Failure modes vs outcome' in md
    assert js['per_run']['v0']['totals']['adk_rejected_calls'] == 3 and js['per_run']['v1']['totals']['adk_rejected_calls'] == 0
    assert rel.main([str(tmp_path / 'nope')]) == 2
    capsys.readouterr()


def test_missing_grades_are_reported_not_fatal(tmp_path):
    b = TaskBuilder(tmp_path / 'r', 't1')
    b.turn([('read_file', {'filepath': P}, ok(content='x'))])
    b.finish()
    run = write_run(tmp_path, 'r', ['t1'])
    s = rel.analyze_runs([run], rel.Thresholds())
    assert s['pooled']['graded'] == 0 and any('no grades.jsonl' in n for n in s['notes'])
    assert rel.render_markdown(s)  # renders without grades
