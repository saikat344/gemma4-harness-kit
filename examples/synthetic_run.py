"""Write a small synthetic run directory (no model, no Docker) for the reliability analyzer.

    python examples/synthetic_run.py runs/synthetic
    python -m harness.analysis.reliability runs/synthetic/demo

Four task-runs, one per behaviour the analyzer detects: a clean fix, a retry loop on a parser-mangled edit_file
(ADK rejects it for a missing parameter), an A-B-A edit oscillation, and a run with one compaction. Built with the
analyzer's own test fixtures (harness/analysis/tests/synth.py); no real trajectories ship with this kit.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.analysis.tests.synth import (TaskBuilder, adk_missing, ok, ok_edit,  # noqa: E402
                                          write_run)

P = 'pkg/mod.py'
OLD = 'def f(x):\n    return x + 1\n'
NEW = 'def f(x):\n    return x + 2\n'


def main(root: Path) -> Path:
    run = root / 'demo'
    b = TaskBuilder(run, 'demo_clean')
    b.turn([('read_file', {'filepath': P}, ok(content=OLD))])
    b.turn([('edit_file', {'filepath': P, 'old_string': OLD, 'new_string': NEW}, ok_edit(P, OLD, NEW))])
    b.turn([('run_command', {'command': 'pytest -q'}, ok(stdout='1 passed', exit_code=0))])
    b.turn([('submit_patch', {}, ok())])
    b.finish()

    b = TaskBuilder(run, 'demo_mangled_loop')
    mangled = {'<|"|>,filepath': P, 'old_string': OLD, 'new_string': NEW}
    b.turn([('read_file', {'filepath': P}, ok(content=OLD))])
    for _ in range(5):
        b.turn([('edit_file', mangled, adk_missing('edit_file', 'filepath'))], tokens=900, latency=9.0, gap=0.2)
    b.turn([('edit_file', {'filepath': P, 'old_string': OLD, 'new_string': NEW}, ok_edit(P, OLD, NEW))])
    b.turn([('submit_patch', {}, ok())])
    b.finish()

    b = TaskBuilder(run, 'demo_aba')
    b.turn([('read_file', {'filepath': P}, ok(content=OLD))])
    for old, new in [(OLD, NEW), (NEW, OLD), (OLD, NEW), (NEW, OLD)]:
        b.turn([('edit_file', {'filepath': P, 'old_string': old, 'new_string': new}, ok_edit(P, old, new))])
    b.finish(empty_patch=True, termination='time_budget', final_text=False)

    b = TaskBuilder(run, 'demo_compaction')
    for i in range(3):
        b.turn([('read_file', {'filepath': f'pkg/m{i}.py'}, ok(content='x = 1\n' * 50))])
    b.compaction()
    b.turn([('edit_file', {'filepath': P, 'old_string': OLD, 'new_string': NEW}, ok_edit(P, OLD, NEW))])
    b.turn([('submit_patch', {}, ok())])
    b.finish()

    return write_run(root, 'demo', ['demo_clean', 'demo_mangled_loop', 'demo_aba', 'demo_compaction'],
                     {'demo_clean': True, 'demo_mangled_loop': True, 'demo_aba': False, 'demo_compaction': True})


if __name__ == '__main__':
    out = main(Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/synthetic'))
    print(f'wrote {out}; now: python -m harness.analysis.reliability {out}')
