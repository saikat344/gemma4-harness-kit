"""End-to-end: real runner + Docker sandbox against the mock server's scripted failure modes, then the analyzer.

Needs Docker and a sandbox image (swebench-sandbox:latest or gemma4-runner-test:latest) plus the httpx_3672
snapshot, like harness/runner/tests/test_e2e_mock.py; skipped otherwise (~1 min)."""

from __future__ import annotations

import json

from harness.analysis import reliability as rel

from .conftest import FIXTURE_SUB, require_snapshots, run_runner


def _analyze(run, th=None, graded: bool | None = None):
    if graded is not None:
        (run / 'grades.jsonl').write_text(json.dumps({'instance_id': 'httpx_3672', 'passed': graded}) + '\n')
    return rel.analyze_runs([run], th or rel.Thresholds())


def test_reliability_scenario_trips_every_detector(tmp_path, image, mock_server):
    require_snapshots('httpx_3672')
    endpoint, _ = mock_server('reliability')
    run = tmp_path / 'run'
    cp = run_runner('--submission', str(FIXTURE_SUB), '--endpoint', endpoint, '--out', str(run),
                    '--task-ids', 'httpx_3672', '--image', image)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    summary = _analyze(run, graded=False)
    t = summary['tasks'][0]
    assert t['source'] == 'llm_calls' and t['status'] == 'done'
    # three identical mangled-key edit_file calls, each rejected by ADK with MissingArguments(filepath)
    assert t['repeat']['max_run'] == 3 and t['repeat']['after_error'] >= 2
    assert t['rejected']['by_kind'] == {'rejected_missing': 3, 'rejected_unknown_tool': 1}
    assert t['rejected']['missing_params'] == {'filepath': 3}
    assert t['parser_signatures']['mangled_key'] == 3
    # a non-matching edit repeated once -> old_string_not_found x2 (identical retry)
    assert t['edit_file']['causes'] == {'old_string_not_found': 2}
    # A-B-A on the real file: apply, revert, re-apply, revert, apply
    assert t['oscillation']['reverts'] == 2 and t['oscillation']['reapplies'] >= 1
    # the 8192-token turn
    assert t['generation']['n_at_cap'] == 1
    flags = set(t['flags'])
    assert {'retry_loop', 'identical_after_error', 'adk_rejected', 'parser_signature', 'edit_fail', 'aba_oscillation',
            'runaway_generation'} <= flags, flags
    assert t['time']['rejected_s'] >= 0 and t['time']['failed_total_s'] > 0
    # the agent still finished: the patch is the gold hunk and was submitted
    assert t['patch_chars'] > 0 and 'empty_patch' not in flags
    md = rel.render_markdown(summary)
    assert '`retry_loop`' in md and 'httpx_3672' in md


def test_loop_scenario_is_a_pure_retry_loop(tmp_path, image, mock_server):
    require_snapshots('httpx_3672')
    endpoint, _ = mock_server('loop')
    run = tmp_path / 'run'
    cp = run_runner('--submission', str(FIXTURE_SUB), '--endpoint', endpoint, '--out', str(run),
                    '--task-ids', 'httpx_3672', '--image', image, '--override', 'max_turns=12')
    assert cp.returncode == 0, cp.stdout + cp.stderr
    t = _analyze(run, graded=False)['tasks'][0]
    assert t['repeat']['max_run'] >= 10 and t['rejected']['total'] >= 10
    # nothing was ever edited successfully, so there is nothing to submit
    assert 'retry_loop' in t['flags'] and 'empty_patch' in t['flags']
    assert t['time']['rejected_s'] >= t['time']['failed_total_s'] * 0.99
