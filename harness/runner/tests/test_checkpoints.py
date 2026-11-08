"""Time checkpoints (--checkpoints, harness/runner/checkpoints.py): unit tests plus one mock run in Docker."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.runner import checkpoints
from harness.runner.mock_openai import LATE_EDIT_FILE

from .conftest import FIXTURE_SUB, require_snapshots, run_runner

# --------------------------------------------------------------------------------------------------
# unit
# --------------------------------------------------------------------------------------------------


def test_parse_and_fmt():
    assert checkpoints.parse('10, 5,7.5,5') == [5.0, 7.5, 10.0]
    assert checkpoints.parse(None) == [] and checkpoints.parse('') == []
    assert [checkpoints.fmt(t) for t in (5.0, 7.5, 0.25)] == ['5', '7.5', '0.25']
    with pytest.raises(ValueError):
        checkpoints.parse('-1')


def test_effective_rules():
    # a submission wins over the worktree, even an empty submission (agent_runner skips the fallback then)
    assert checkpoints.effective(True, 'SUB', 'WT') == ('submitted', 'SUB')
    assert checkpoints.effective(True, '', 'WT') == ('submitted', '')
    assert checkpoints.effective(False, None, 'WT') == ('worktree', 'WT')
    assert checkpoints.effective(False, None, '  \n') == ('none', '')
    assert checkpoints.effective(False, None, None) == ('error', '')


class _FakeCtx:
    def __init__(self, diff: str = '', exit_code: int = 0, boom: bool = False):
        self.agent_start_time = 1.0
        self.agent_elapsed_seconds = 0.0
        self.patch_submitted = False
        self.submitted_patch = None
        self.container_id = 'c1'
        self.execs = 0
        outer = self

        class _Docker:
            def exec(self, cid, cmd, timeout=None):
                outer.execs += 1
                assert 'GIT_INDEX_FILE' in cmd and timeout == checkpoints.EXEC_TIMEOUT_S
                if boom:
                    raise RuntimeError('docker gone')
                return SimpleNamespace(exit_code=exit_code, stdout=outer.diff, stderr='err', status='ok')

        self.docker = _Docker()
        self.diff = diff


DIFF_SRC = 'diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1 +1 @@\n-a\n+b\n'
DIFF_TEST = 'diff --git a/tests/test_a.py b/tests/test_a.py\n--- a/tests/test_a.py\n+++ b/tests/test_a.py\n@@ -1 +1 @@\n-a\n+b\n'


def _recs(task_dir: Path) -> dict[float, dict]:
    return {r['t']: r for r in map(json.loads, (task_dir / 'checkpoints.jsonl').read_text().splitlines())}


def test_checkpointer_crossings_share_one_exec_and_strip(tmp_path):
    ctx = _FakeCtx(diff=DIFF_SRC + DIFF_TEST)
    cp = checkpoints.Checkpointer(tmp_path, ctx, [0.5, 1, 2, 3])
    ctx.agent_elapsed_seconds = 10
    cp.poll('before_call', 1)
    assert ctx.execs == 0 and not (tmp_path / 'checkpoints.jsonl').exists()
    ctx.agent_elapsed_seconds = 65  # passes 0.5 and 1 min at once -> one exec, two records
    cp.poll('after_response', 2)
    assert ctx.execs == 1
    ctx.patch_submitted, ctx.submitted_patch = True, 'SUBMITTED\n'
    ctx.agent_elapsed_seconds = 125
    cp.poll('before_call', 3)
    cp.poll('before_call', 4)  # nothing pending is due -> no exec
    assert ctx.execs == 2
    cp.finish('FINAL\n', True)
    recs = _recs(tmp_path)
    assert set(recs) == {0.5, 1.0, 2.0, 3.0}
    assert recs[0.5]['effective_source'] == recs[1.0]['effective_source'] == 'worktree'
    assert (tmp_path / 'checkpoints' / '1.patch').read_text() == DIFF_SRC  # test file stripped
    assert recs[1.0]['elapsed_actual'] == 65 and recs[1.0]['where'] == 'after_response'
    assert recs[2.0]['effective_source'] == 'submitted' and recs[2.0]['worktree_sha'] == checkpoints.sha(DIFF_SRC)
    assert (tmp_path / 'checkpoints' / '2.patch').read_text() == 'SUBMITTED\n'
    assert recs[3.0]['reason'] == 'ended_before' and (tmp_path / 'checkpoints' / '3.patch').read_text() == 'FINAL\n'


@pytest.mark.parametrize('kind', ['exit', 'raise'])
def test_checkpointer_never_raises_on_exec_failure(tmp_path, kind):
    ctx = _FakeCtx(diff=DIFF_SRC, exit_code=1 if kind == 'exit' else 0, boom=kind == 'raise')
    cp = checkpoints.Checkpointer(tmp_path, ctx, [0.1])
    ctx.agent_elapsed_seconds = 30
    cp.poll('before_call', 1)
    rec = _recs(tmp_path)[0.1]
    assert rec['effective_source'] == 'error' and rec['worktree_sha'] is None
    assert (tmp_path / 'checkpoints_errors.log').read_text().strip()
    # a broken context is logged, not raised
    cp2 = checkpoints.Checkpointer(tmp_path / 'x', SimpleNamespace(agent_start_time=1, agent_elapsed_seconds='bad'), [0])
    cp2.poll('before_call', 1)
    cp2.finish('', False)


def test_export_formats_and_missing(tmp_path):
    run = tmp_path / 'run'
    (run / 'tasks' / 'a').mkdir(parents=True)
    (run / 'tasks' / 'b').mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({'task_ids': ['a', 'b'], 'checkpoints': [5.0, 7.5]}))
    cp = checkpoints.Checkpointer(run / 'tasks' / 'a', _FakeCtx(diff=DIFF_SRC), [5.0, 7.5])
    cp.context.agent_elapsed_seconds = 301
    cp.poll('before_call', 1)
    cp.finish('', False)  # 7.5 ended_before with no patch
    stats = checkpoints.export(run, run / 'at')
    assert [s['t'] for s in stats] == [5.0, 7.5] and stats[0]['with_patch'] == 1
    rows5 = [json.loads(ln) for ln in (run / 'at' / 'predictions_at_5.jsonl').read_text().splitlines()]
    assert rows5[0]['patch'] == rows5[0]['prediction'] == DIFF_SRC and rows5[0]['id'] == 'a'
    assert rows5[1]['patch'] == 'NO_PATCH' and rows5[1]['status'] == 'missing'
    rows75 = [json.loads(ln) for ln in (run / 'at' / 'predictions_at_7.5.jsonl').read_text().splitlines()]
    assert rows75[0]['patch'] == 'NO_PATCH' and rows75[0]['checkpoint_reason'] == 'ended_before'


# --------------------------------------------------------------------------------------------------
# end to end: mock agent edits, submits, then edits again after the submission
# --------------------------------------------------------------------------------------------------


def test_checkpoints_e2e_submission_wins_over_later_worktree(tmp_path, image, mock_server):
    iid = 'requests_6629'
    require_snapshots(iid)
    endpoint, _ = mock_server('late_edit', delay=2.0)
    run = tmp_path / 'run'
    # every 0.5 s of agent time up to 40 s, so each ~2 s LLM call window holds several checkpoints; 999 is never reached
    secs = [i / 2 for i in range(1, 81)]
    ts = [s / 60 for s in secs] + [999.0]
    cp = run_runner('--submission', str(FIXTURE_SUB), '--endpoint', endpoint, '--out', str(run), '--task-ids', iid,
                    '--image', image, '--checkpoints', ','.join(repr(t) for t in ts))
    assert cp.returncode == 0, cp.stdout + cp.stderr
    td = run / 'tasks' / iid
    res = json.loads((td / 'result.json').read_text())
    assert res['status'] == 'done' and res['termination'] == 'submitted', res
    final = (td / 'patch.diff').read_text()
    assert final.strip() and LATE_EDIT_FILE not in final  # the submission, not the later worktree, is graded

    manifest = json.loads((run / 'run.json').read_text())
    assert manifest['checkpoints'] == sorted(ts)
    assert not (td / 'checkpoints_errors.log').exists()
    recs = [json.loads(ln) for ln in (td / 'checkpoints.jsonl').read_text().splitlines()]
    assert sorted(r['t'] for r in recs) == sorted(ts)
    for r in recs:
        patch = (td / 'checkpoints' / f"{checkpoints.fmt(r['t'])}.patch").read_text()
        assert r['reason'] == 'ended_before' or r['elapsed_actual'] > r['t'] * 60
        if r['has_submitted']:
            assert r['effective_source'] == 'submitted' and patch == final, r
        elif r['effective_source'] == 'worktree':
            assert patch.strip() and checkpoints.sha(patch) == r['worktree_sha']
        else:
            assert r['effective_source'] == 'none' and patch == '', r
    crossed = [r for r in recs if r['reason'] == 'crossed']
    sources = {r['effective_source'] for r in crossed}
    assert {'none', 'worktree', 'submitted'} <= sources, crossed  # before edits, edited-not-submitted, submitted
    # after the post-submit write_file the worktree differs, yet the submission is the effective patch
    late = [r for r in crossed if r['has_submitted'] and r['worktree_sha'] != r['submitted_sha']]
    assert late, crossed
    assert all(LATE_EDIT_FILE not in (td / 'checkpoints' / f"{checkpoints.fmt(r['t'])}.patch").read_text()
               for r in late)
    # edited-but-not-submitted worktree equals the final submission here (same edits, then submit)
    assert any(r['effective_source'] == 'worktree' and r['worktree_sha'] == checkpoints.sha(final) for r in crossed)
    end = next(r for r in recs if r['t'] == 999.0)
    assert end['reason'] == 'ended_before' and end['effective_source'] == 'submitted'
    # one exec per crossing event, not per checkpoint and not per call
    events = {(r['where'], r['call_index']) for r in crossed}
    calls = len((td / 'llm_calls.jsonl').read_text().splitlines())
    assert len(events) <= 2 * calls

    cp2 = run_runner('checkpoints', str(run), '--out-dir', str(run / 'at'))
    assert cp2.returncode == 0, cp2.stdout + cp2.stderr
    pred = json.loads((run / 'predictions.jsonl').read_text())
    at_end = json.loads((run / 'at' / 'predictions_at_999.jsonl').read_text())
    assert at_end['patch'] == pred['patch'] and at_end['instance_id'] == iid
    assert set(pred) <= set(at_end)
    first = json.loads((run / 'at' / f'predictions_at_{checkpoints.fmt(ts[0])}.jsonl').read_text())
    assert first['patch'] == 'NO_PATCH'
