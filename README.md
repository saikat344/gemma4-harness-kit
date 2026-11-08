# gemma4-harness-kit

An evaluation kit, built on the official stack, for the Kaggle **Gemma 4 Developer Agent** benchmark. It wraps, rather than
reimplements, the official runner and verifier (`swegemma` 0.2.11, `adk-submission` 0.2.13, `adk-eval-core` 0.1.0,
google-adk 1.36.1) and makes lost evidence attributable to the transformation that lost it.

Paper: *The Harness Is Part of the Benchmark* (Kaggle Gemma 4 Developer Agent paper track) — link added at
publication.

What is in it:

| Part | Path | Doc |
|---|---|---|
| Grader: official `verify_task` in a sandbox built from the competition's public Dockerfile, with switchable, documented environment repairs | `harness/grader/`, `harness/sandbox/`, `harness/common/` | [docs/grader.md](docs/grader.md) |
| Runner: executes a submission the way the official notebook does, one process per task, with per-call and compaction telemetry | `harness/runner/` | [docs/runner.md](docs/runner.md), [docs/formats.md](docs/formats.md) |
| Serving profiles: the scorer's exact vLLM argv plus single-GPU profiles, every deviation printed | `harness/serve/` | [docs/runner.md](docs/runner.md) §3 |
| Harness contracts with CPU fixtures (B1 serializer, templating, context limit) | `harness/contracts/` | [docs/harness-contracts.md](docs/harness-contracts.md) |
| Executable ADK spec: 35 tests (28 compaction, 7 agent hand-off) running real google-adk code with a scripted model, plus a version matrix runner | `harness/adk_spec/` | [docs/harness-contracts.md](docs/harness-contracts.md) §7 |
| Reliability probes: vLLM gemma4 parser suite, trajectory analyzer, P-E5 serializer replay | `harness/repro/`, `harness/analysis/` | [docs/reliability.md](docs/reliability.md), [docs/pe5.md](docs/pe5.md) |
| Task subsets and the unsound list | `harness/subsets/` | [docs/grader.md](docs/grader.md) |
| Audit results (verdicts and aggregates only) | `data/audit/` | `PROVENANCE.json` or a `provenance` field where present; `pe5/` is described in docs/pe5.md |

Not included: competition data, the harness wheels, model weights ([docs/data.md](docs/data.md) has the download
commands), agent configurations, and trajectories (`llm_calls.jsonl` holds full prompts and task text). The authors'
cloud launchers (Modal, Vast) are not shipped; `harness/serve/serve.py --model-dir <dir> --print` gives the argv to run anywhere.

## Quickstart (CPU, no Docker, no GPU)

Needs Python 3.12, [uv](https://docs.astral.sh/uv/) and the three Kaggle wheels in `data/wheelhouse/`
([docs/data.md](docs/data.md), "Harness wheels").

```bash
uv sync --frozen --project harness/runner                                       # 1. pinned env (google-adk 1.36.1)
uv run --project harness/runner python -m pytest harness/contracts -q -s        # 2. B1 + contracts, < 10 s
uv run --project harness/runner python -m harness.subsets.compare_unsound \
    --gold data/audit/gold-all/grades.jsonl --empty data/audit/empty-all/grades.jsonl   # 3. unsound list
uv run --project harness/runner python examples/synthetic_run.py runs/synthetic  # 4. a synthetic run log
uv run --project harness/runner python -m harness.analysis.reliability runs/synthetic/demo   # 5. analyzer report
```

The grader needs Docker and the competition data; see [docs/grader.md](docs/grader.md) "Setup".

## Claims → command → expected output

Run from the repository root after the setup in the linked doc. `uvr` = `uv run --project harness/runner`;
`grader` = `harness/grader/.venv/bin/python -m harness.grader` (after `harness/grader/setup_venv.sh`).

| Paper claim | Command | Expected |
|---|---|---|
| B1: the 1.36.1 compaction serializer passes only text parts (§4.1) | `uvr python -m pytest harness/contracts -q -s` | all pass in < 10 s; prints `google-adk 1.36.1` and the serializer output `'user: ISSUE_MARKER fix bug\\nagent: THOUGHT_MARKER\\nagent: TEXT_MARKER'`. On google-adk ≥ 2.2.0 the B1 test fails by design (upstream commit bdb5582 adds tool fields); the vendored 2.2.0 serializer test shows that output |
| Compaction defects reproduce in real ADK code (§4.1): text-only serializer, seed carries the previous summarizer's thinking, one-call trigger lag | `uvr python -m pytest -c harness/adk_spec/pytest.ini harness/adk_spec` | `35 passed` in a few seconds on google-adk 1.36.1 (CPU, no Docker) |
| Upgrading changes, not removes, the defects (§4.1): bdb5582 in 2.2.0, re-injection as the agent's own turn with the summarizer's thinking from 2.7.0 | per-version venvs, then `python3 harness/adk_spec/run_matrix.py <venvs...>` ([docs/harness-contracts.md](docs/harness-contracts.md) §7) | the table in `data/audit/adk-spec/matrix.md`: `s03`–`s06`, `s12` fail from 2.2.0; `r01`, `r02`, `o01` fail on 2.13.0; `l01` fails on main only; `s08` passes everywhere |
| A hand-off to a second agent is another serializer (§4.1): `transfer_to_agent` turns the other agent's events into role=user text without its thinking; an `AgentTool` sub-agent sees only its instruction and the request | `uvr python -m pytest -c harness/adk_spec/pytest.ini harness/adk_spec -k h0` | `7 passed` on 1.36.1; in the matrix only `h07` (wording) fails, on 2.13.0 and main |
| Templating and context-limit contracts (Table 3) | `uvr python -m pytest harness/contracts -q -s` (as for B1) | `test_templating_*`, `test_context_trigger_ignores_tool_results_after_the_last_usage` pass |
| Compactions are invisible in the official trace (§4.2) | `uvr python -m pytest harness/runner/tests/test_e2e_mock.py -q -k compaction`; CPU-only: `uvr python -m pytest -c harness/adk_spec/pytest.ini harness/adk_spec -k "v01 or v02 or v03"` | 1 passed (needs Docker, the sandbox image and the `httpx_3672` snapshot); CPU-only: 3 passed |
| Submission contract: last `submit_patch` wins over later edits (Table 3) | `uvr python -m pytest harness/runner/tests/test_checkpoints.py -q -k submission_wins` | 1 passed (Docker) |
| Stripping of test/config paths uses the official function (Table 3) | `uvr python -m pytest harness/runner/tests/test_checkpoints.py -q -k strip` | 1 passed (CPU) |
| Tool budget is soft; no submission and no diff → no patch (Table 3) | `uvr python -m pytest harness/runner/tests/test_e2e_mock.py -q -k "subagent_budget or empty_agent"` | 2 passed (Docker) |
| Gold audit, 129 tasks (§5) | `grader --gold --out runs/audit/gold/grades.jsonl --workers 6` | last line starts `RESOLVED 114/129 (88.4%) mode=gold`; by repo fastapi 62/67, httpx 1/1, requests 5/13, rich 46/48; about 10 min on 6 native x86_64 workers. Reference: `data/audit/gold-all/` (re-run from a clean copy of this kit on another x86_64 host, 2026-11-08: the same verdict on all 129 tasks, gold and empty) |
| Empty audit, 129 tasks (§5) | `grader --empty --out runs/audit/empty/grades.jsonl --workers 6` | `RESOLVED 2/129 (1.6%) mode=empty` (fastapi_14077, fastapi_14851); about 15 min; 3 rich tasks hit the 300 s timeout. Reference: `data/audit/empty-all/` |
| 17 unsound tasks (Table 4) | `uvr python -m harness.subsets.compare_unsound --gold runs/audit/gold/grades.jsonl --empty runs/audit/empty/grades.jsonl` | `unsound from grades: 17; listed in unsound.txt: 17; diff: none` |
| Repairs alone: gold 6/20 → 16/20 (Table 2) | `harness/sandbox/build.sh --official`, then `grader --gold --task-ids $(grep -v '^#' harness/grader/validation_subset.txt) --out runs/v/gold/grades.jsonl` and the same with `--no-env-fixes --image swebench-sandbox:official --out runs/v/gold-official/grades.jsonl` | `RESOLVED 16/20 (80.0%)` vs `RESOLVED 6/20 (30.0%)`; with `--empty`: 1/20 vs 0/20. Reference: `data/audit/subset20/` |
| Stage `official` is the public Dockerfile verbatim except pins (§3) | `python3 harness/sandbox/compose_dockerfile.py --public data/kaggle/docker/Dockerfile.public --diff` | exactly two changed lines: the `FROM` (digest + stage name) and the `pip install` line (versions) |
| Community overlap (§5) | `uvr python -m harness.subsets.compare_unsound --community task_soundness.csv` (CSV from the community notebook, [docs/data.md](docs/data.md)) | `ours 17 vs community 15, overlap 12, symmetric difference 8`. Reference: `data/audit/gluzdov-overlap.md` |
| Reliability incidence (Table 6, P-E2) | `uvr python examples/synthetic_run.py runs/synthetic && uvr python -m harness.analysis.reliability runs/synthetic/demo` (no real trajectories ship) | on the synthetic run: 4 tasks, 3 resolved, 22 tool calls, 5 ADK-rejected and 5 mangled-key calls, 1 A-B-A revert, 1 compaction. The paper's numbers: `data/audit/ab1-reliability/per_arm.md` |
| Table 5 per pass; compaction cost (§4.1) | read `data/audit/ab1-reliability/per_pass.json` (aggregated on the home PC from the ab1 run dirs; no trajectories ship) | per pass (baseline-v0-A/B, v1-A/B): resolved 11/13/9/8; time-cap endings 24/21/28/30; ContextWindowExceeded deaths 3/2/0/0; `write_file` errors 0/0/26/14 (all on /tmp paths); empty patches 8/4/11/21; compactions 32/30/30/18 (110 over 148 task-runs = 0.74). Summarizer call latency: n 110, mean 29.5 s, median 28.4 s |
| B3 in practice (§6) | read `data/audit/ab1-reliability/patch_source.json` | patch_source × resolved × termination per pass; pooled over the four ab1 passes plus the 3090 baseline pass: 51 resolved, 20 graded from the fallback diff, 22 still running at the cap |
| Timing proxy (§6) | read `data/audit/serving/health.json` (`harness.serve.health_check` summaries) | 3090 (`--profile 3090`): 34.6 tok/s single stream, 28,263-token prompt in 40.7 s, `max_model_len_32768` PASS. Rented 4×L4 (TP=4, scorer flags): read `data/audit/serving/4xl4-decode.json`: 39.70 / 36.22 / 34.03 / 32.31 tok/s at 576 / 9,319 / 18,898 / 28,472 prompt tokens |
| Refused edits are one slip (Table 6 note) | read `data/audit/ab1-reliability/backtick.json` (scanned from the ab1 baseline-v0 logs; no trajectories ship) | `edit_file_rejected` 10 of 134, `edit_file_rejected_backtick_slip` 10, 6 of them byte-identical retries; `write_file_rejected` 0 of 127 |
| Time is output tokens (§7) | read `data/audit/ab1-reliability/timeshare.json` (ab1 baseline-v0, 74 task-runs) | `llm_share_of_agent_time` 0.99, `thinking_pct_of_output_tokens` 70.2 |
| P-E3 parser suite | `<vllm-venv>/bin/python -m harness.repro.parser_cases --tokenizer <model dir>` (vLLM 0.19.1 venv and the tokenizer from the downloaded weights; [docs/reliability.md](docs/reliability.md) §1) | canonical 118/118 round-trip, glitch 15/15 fail as predicted. CPU self-test without vLLM: `uvr python -m pytest harness/repro/tests -q` |
| P-E5 serializer replay | `uvr python -m harness.analysis.pe5 prepare ...` on your own runner logs (method: [docs/pe5.md](docs/pe5.md)); the validity check runs on CPU | the validity count; the paper's results: `data/audit/pe5/` |
| Shipped-environment audit, cross-grading (Table 2, §5, P-E4) | the grader commands above with `--no-env-fixes --image swebench-sandbox:official` on all 129 tasks, and on the P-E1 predictions | as shipped: gold **43/129**, empty **4/129** (`data/audit/shipped-env/{gold-all,empty-all}`); repaired image (`--image swebench-sandbox:latest`) without task fixes, subset-20 gold: **5/20** (`subset20-repaired-nofixes`, the Table 2 attribution cell); ab1 cross-grading: **25/148** patches discordant, all pass → fail, each of the 4 passes 4/37 as shipped (`data/audit/cross-grading/discordance.json`, rows with `run` starting `ab1/`) |
| Agent experiments P-E1 (Table 5) | runner + grader per [docs/runner.md](docs/runner.md) §4; needs a 24 GB GPU (or an A100) and an agent config | agent configurations are not released before the competition ends; the kit runs any submission directory or zip |

Per-task grading time depends on the host: 16–60 s per task under amd64 emulation on an M3 Mac at 3 workers.

## Limits

- The scorer's image is private, so equivalence with it is unverified. Our repairs make *our* grader accept gold
  patches; the scorer applies none. `--no-env-fixes` (runner and grader) and `build.sh --official` give the
  environment as shipped.
- Public tasks span four repositories; the hidden test set is about 120 tasks from private repositories.
- A soundness list is environment-relative: `harness/subsets/unsound.txt` states the environment it holds for.

## Licence

Apache-2.0 ([LICENSE](LICENSE)). Third-party material and attributions, including the CC BY 4.0 community soundness
data and the vendored google-adk 2.2.0 serializer: [NOTICE](NOTICE). Pinned versions: [VERSIONS.md](VERSIONS.md).
