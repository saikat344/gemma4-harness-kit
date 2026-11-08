<!-- source: runs/audit/ab1-reliability/per_arm.md (harness/analysis/reliability.py over the four ab1 passes, 2026-11-02/03; A100-80GB with the scorer's serving flags, Docker sandbox, repairs on); refreshed at e2efac2; aggregates only; one row label reworded to the paper's B2 wording -->
| metric (per 100 task-runs) | baseline-v0 (74 runs) | v1 (74 runs) |
|---|---|---|
| resolved (count) | 24 (32.4%) | 17 (23.0%) |
| tool calls | 1968 | 1512 |
| repeat of an earlier failed call | 44.6 | 37.8 |
| identical consecutive calls | 16.2 | 20.3 |
| ADK-rejected calls | 13.5 | 39.2 |
| mangled-key calls | 43.2 | 43.2 |
| raw tool-call text | 1.4 | 4.1 |
| edit_file failures | 21.6 | 6.8 |
| A-B-A reverts | 5.4 | 2.7 |
| A-B-A re-applies | 1.4 | 0.0 |
| compactions | 83.8 | 64.9 |
| compactions after which the issue opening is not quoted verbatim | 82.4 | 0.0 |
| empty patches | 16.2 | 43.2 |
| task-runs with >=1 compaction | 68.9 (51) | 55.4 (41) |
| task-runs with an A-B-A revert | 5.4 (4) | 2.7 (2) |
| task-runs with >=1 ADK rejection | 4.1 (3) | 12.2 (9) |
| task-runs with a repeat-after-error | 6.8 (5) | 8.1 (6) |
