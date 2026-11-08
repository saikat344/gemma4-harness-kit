# Harness contracts

What the official harness actually does, read from the source of the pinned packages. These are
the behaviours an agent configuration must live with and that the kit reproduces or measures.

**Versions.** swegemma 0.2.11, adk-submission 0.2.13, adk-eval-core 0.1.0 (Kaggle dataset
`metric/gemma-4-developer-agent-wheelhouse`; not on PyPI), google-adk 1.36.1, vLLM 0.19.1.
**Citations** are paths inside those packages, e.g. `swegemma/harness/verification.py:verify_task`
or `google/adk/apps/llm_event_summarizer.py`. Unzip a wheel to read them; nothing needs to be
installed. (inferred) marks reasoning from source that the source does not state.

| Contract | Section | Test in this kit |
|---|---|---|
| Grading scope and pass criterion | 1 | gold/empty audit ([grader.md](grader.md)) |
| Generated `pytest.ini` | 2 | gold/empty audit (the anyio class) |
| Stripping of test/config paths | 1, 3 | `harness/runner/tests/test_checkpoints.py::test_checkpointer_crossings_share_one_exec_and_strip` (calls the official strip function) |
| `submit_patch` last wins; later edits lost | 3 | `harness/runner/tests/test_checkpoints.py::test_checkpoints_e2e_submission_wins_over_later_worktree` |
| No submission and no diff: no grading | 3 | `harness/runner/tests/test_e2e_mock.py::test_empty_agent_gives_no_patch_after_nudges` |
| Tool budget is soft | 3 | `harness/runner/tests/test_e2e_mock.py::test_sample_submission_routes_loras_and_subagent_budget` |
| Compaction not in the official trace | 4 | `harness/runner/tests/test_e2e_mock.py::test_compaction_is_triggered_and_recorded` |
| B1 serializer passes only text parts | 4 | `harness/contracts/test_adk_contracts.py::test_b1_serializer_keeps_text_and_drops_tool_fields` (and `test_b1_vendored_220_serializer_keeps_tool_fields` for the 2.2.0 contrast) |
| Context-limit arithmetic | 5 | `harness/contracts/test_adk_contracts.py::test_context_trigger_ignores_tool_results_after_the_last_usage` |
| Instruction templating | 6 | `harness/contracts/test_adk_contracts.py::test_templating_*` (missing name → KeyError, `{name?}` → empty, JSON braces literal, no re-scan) |
| File tools are `/workspace`-only | 3 | none (observed in real runs) |

The e2e tests need Docker and a sandbox image ([runner.md](runner.md) section 8).
`harness/contracts/test_adk_contracts.py` runs on CPU in seconds and prints the installed ADK
version.

## 1. Grading procedure (`swegemma/harness/verification.py:verify_task`)

Each task gets two fresh containers: one for the agent, one for verification. Setup time is
excluded from the agent clock.

1. **Early exits.** No verification if the agent produced an error and no patch
   (`swegemma/evaluate.py:Evaluator.evaluate_task`: resolved=False). If `test_patch`,
   FAIL_TO_PASS, PASS_TO_PASS and test files are all empty, the result is resolved=False
   ("Missing test specification").
2. **Setup** as for the agent container (snapshot into `/workspace`, editable install, wheel
   injection, `sandbox/setup.py --fast-path`, test config, baseline commit). Then
   `target_test_files` = destination paths of `+++` lines in `test_patch`
   (`swegemma/models/task.py:extract_test_files_from_patch`; deletions excluded, renames use the
   new name).
3. **Strip** protected sections from the agent patch (`strip_protected_files_from_patch`,
   `_is_protected_test_or_config_path`). A diff section is dropped whole if any of its paths is
   protected. Protected: basenames `conftest.py pytest.ini pyproject.toml tox.ini setup.cfg
   .pytest.ini sitecustomize.py usercustomize.py _swegemma_stubs.py`; `*.pth`; `test_*.py`;
   `*_test.py`; any `.py` under a directory named `tests`, `test` or `testing`
   (case-insensitive); every path in `target_test_files`. `submit_patch` already ran the same
   function, so stripping happens twice. Agent-written tests never count.
4. **Apply** the stripped patch (`swegemma/harness/container_setup.py:apply_patch_in_container`):
   `git apply` variants (plain, `-3`, whitespace-insensitive, `--recount`), the same on a
   symlink-normalised copy, `-p0` variants, then GNU `patch` with `--dry-run` guards. Absolute
   paths, `..` and symlink escapes are rejected. If all fail: resolved=False. An empty patch skips
   this step and the tests still run.
5. **Reset** target test files and protected files the agent touched to `HEAD`, then
   `git clean -f` them.
6. **Apply `test_patch`** with the same applier; failure gives resolved=False.
7. Re-run the test-dependency install and write the test config again with `overwrite=True`
   (section 2).
8. **Run**, with timeout `verification_timeout_seconds` (default 300 s, `timeout -k 5s` in Docker):

   ```bash
   cd /workspace && PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -m pytest <targets> \
     --junitxml=/tmp/_swegemma_junit_<12hex>.xml -p no:anyio -o timeout=0 -o python_classes="Test* *Test" -q
   ```

   `<targets>` are the `.py` files from `test_patch` except `conftest.py` (fallback: all
   `test_patch` files, then `.`). This is **every test in each touched file**: not only the new
   tests, and not the whole suite. A pre-existing failure in a touched file sinks the task.
9. **Pass criterion** (all required):
   - pytest exit code 0;
   - a valid JUnit XML (`_validate_junit_xml`): total > 0, passed > 0, no failures, no errors;
   - every required test passed and none was skipped or failed. Required tests are
     FAIL_TO_PASS + PASS_TO_PASS if present (not in the public data), else the names of
     `def test_*` / `async def test_*` lines **added** by `test_patch`
     (`_extract_test_functions_from_patch`); if there are none, names from hunk headers and
     context lines. Matching is by function name only (`_node_matches_requirement`: exact,
     `name[...]`, `name_<suffix>`, `name<digits>`). 22 of the 129 public `test_patch`es add no
     `def test_` and use the fallback;
   - the pytest summary is not `0 passed`, `no tests ran` or `collected 0 items`.
10. **Score** = resolved / total over all tasks, error tasks included
    (`adk_eval_core/runner/runner.py:EvaluationResult.resolution_rate`). How Kaggle aggregates the
    public and private halves is not in the wheels.

## 2. Generated `pytest.ini` and `conftest.py` (`swegemma/harness/container_setup.py:setup_workspace_test_config`)

Before grading, this file is written with `overwrite=True`, replacing any repository `pytest.ini`:

```
[pytest]
addopts = -p no:anyio
norecursedirs = .* build dist venv
python_classes = Test* *Test
python_files = test_*.py *_test.py
filterwarnings =
    ignore::DeprecationWarning
    ignore::UserWarning
```

- A `pytest.ini` always wins over `pyproject.toml`/`setup.cfg`, so repository pytest settings
  (markers, `filterwarnings = error`, `asyncio_mode`, `testpaths`) are ignored in grading.
- `-p no:anyio` is forced and the harness never injects pytest-asyncio, so anyio-marked async
  tests cannot pass (3 public tasks fail gold for this reason; [grader.md](grader.md)).
- `--import-mode=importlib`, added earlier by `sandbox/setup.py`, is overwritten, so grading uses
  pytest's default `prepend` import mode.
- In the agent phase, `setup.py --fast-path` writes or merges its own `pytest.ini` (with
  `--import-mode=importlib -p no:anyio`), and the harness version is skipped when a `[pytest]`
  section exists. The agent therefore sees slightly different pytest settings than the grader.
- `conftest.py` gets only a one-line comment prepended; there is no executable hook. (The
  scorer's private `setup.py` could differ; unverified.)
- Both files are committed into the baseline, so they never appear in diffs.

## 3. The agent loop (`swegemma/harness/agent_runner.py:run_agent_sandbox`)

**Loop.** One ADK `Runner` with `InMemorySessionService`, plugins `[EventDisplayPlugin,
ModelRetryPlugin, ToolErrorPlugin]`, and session state `{problem_description, hints}`. The first
user message comes from `build_agent_prompt` (problem statement, hints, budget lines, execution
rules, workspace tree). The agent's system prompt is the `instruction` from `agent.yaml`.
`runner.run_async` runs inside `asyncio.timeout(max_time_minutes*60)`. When `run_async` returns
without a submitted patch, a nudge message is sent; 3 consecutive nudges end the run (the counter
resets after any turn with a tool call).

**Budgets.**
- Time: the agent clock starts after sandbox setup; `asyncio.timeout` is the hard stop.
  `run_command` timeout = `min(command_timeout, max(5, remaining_seconds))`.
- Turns: `RunConfig(max_llm_calls = max_turns - llm_calls_used)`; 500 if unset.
- **Tool calls are a soft budget** (`swegemma/context.py:SwegemmaContext.check_budget`,
  `swegemma/tools/base.py:budget_gated`). When exhausted, tools return
  `{"status":"error","error_type":"BudgetExceeded",...}` but the model keeps getting turns.
  `submit_patch` and `get_status` are not counted (`swegemma/tools/execution.py`). Time and turns
  are the real stops.
- Output caps: command output 5000 characters per stream, `read_file` 150 lines / 10000
  characters, edit diffs 5000 characters.
- `eval_config.yaml` is parsed by the caller (the getting-started notebook), not by the wheels;
  whether the scorer caps its values is unknown.

**Submission** (`swegemma/tools/execution.py:submit_patch`, B3).
- `submit_patch` snapshots `git add -N . && git diff --binary _swegemma_baseline`, strips
  protected files (section 1, step 3) and stores the result. It does **not** end the run. It can
  be called many times; **the last call wins**, and edits made after it are lost.
- The run ends after a text-only reply once a patch was submitted, or on a time, turn or nudge
  limit.
- Without a submission, the same diff of the working tree is taken as a **fallback** and graded
  if non-empty (also after budget termination). Scratch files left in `/workspace` that are not
  protected end up in it. If the diff is empty, the agent error is "Agent completed execution
  without calling submit_patch." and verification is skipped.
- In real runs 39% of resolved task-runs never called `submit_patch` and were graded from the
  fallback diff (see the paper, "B3 in practice").

**Tools.** A closed registry (`swegemma/tools/__init__.py:create_tools`): `run_command, read_file,
write_file, edit_file, submit_patch, get_status, get_code_neighbors, search_similar_code,
get_code_subgraph`. File tools take `filepath` and accept **only paths inside `/workspace`**
(`swegemma/tools/workspace.py:_resolve_workspace_path`): `/tmp/x.py` fails with "Path traversal
detected: ... escapes workspace root". Scratch files outside the patch must go through
`run_command`. ADK rejects a call only when a mandatory parameter is missing and silently drops
unknown keys; rejected calls cost a turn and time but no tool budget
([reliability.md](reliability.md)).

## 4. Compaction and the summarizer (B1, B2)

The getting-started notebook, and therefore the runner, passes
`EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336,
event_retention_size=5)` and `ContextCacheConfig(min_tokens=2048, ttl_seconds=1800,
cache_intervals=10)`. The `swegemma eval` CLI passes neither, so it is not scorer-equivalent in
context handling.

**Mechanism** (`google/adk/flows/llm_flows/compaction.py`, `google/adk/apps/compaction.py`,
`google/adk/apps/llm_event_summarizer.py`, google-adk 1.36.1):
- Before each LLM call, if the previous prompt had >= 14,336 tokens, every not-yet-compacted
  event except the last 5 is summarized (the split moves earlier so no tool call is separated from
  its response; pending calls are never compacted). The previous summary is passed as a seed, so
  summaries roll forward.
- The summarizer is `LlmEventSummarizer(llm=agent.canonical_model)`: the same Gemma model, one
  user message with ADK's fixed default prompt, no system instruction and no generation config
  from the submission. Its time counts against the task budget.
- The summary replaces the window as one event, which is appended with
  `session_service.append_event` instead of being yielded. **The official trace therefore records
  no compactions**; the runner logs them in `compactions.jsonl`.

**B1: the serializer passes only text parts.** `LlmEventSummarizer._format_events_for_prompt`:

```python
for event in events:
  if event.content and event.content.parts:
    for part in event.content.parts:
      if part.text:
        formatted_history.append(f'{event.author}: {part.text}')
return '\\n'.join(formatted_history)
```

`function_call` and `function_response` parts have no `text`, so commands, file contents, test
output and edit results never reach the summarizer. Thought-flagged text parts do pass. The join
string is the two characters backslash-n, so the history arrives as one line (effect unmeasured).
Evidence dropped once cannot come back through later, seeded summaries.

**B2: the issue text reaches the summarizer.** It is a user text part, so it is serialized.
Whether its content survives depends on the summary. In real runs the first 120 characters of the
issue were absent verbatim from the first post-compaction request after 61 of 62 compactions, but
a summary can paraphrase; P-E5 measures what is retained. The agent.yaml `instruction` is not part
of the compacted history and is re-rendered on every call (section 6), so templating
`{problem_description}` into it keeps a verbatim copy of the task in every request. That protects
the task, not the tool observations lost at B1.

**Version pinning.** Upstream commit
[`bdb5582`](https://github.com/google/adk-python/commit/bdb5582) (2026-06-02) adds thoughts, tool
calls and truncated tool responses to the serializer and joins with a real newline. It first
shipped in google-adk 2.2.0. The 1.x line is unchanged through 1.39.1, and the harness pins 1.36.1
(`adk-submission` requires `google-adk>=1.34,<2.0`). On ADK >= 2.2.0 the B1 assertions in
`harness/contracts/test_adk_contracts.py` fail by design; the test says so.

## 5. Context-limit contract

- vLLM serves the model at `--max-model-len 32768` and rejects a request when prompt tokens +
  `max_output_tokens` exceed it. LiteLLM raises `ContextWindowExceededError`, the agent loop ends
  with an exception, and the task keeps only what was already submitted or in the working tree
  (measured in real runs).
- Compaction fires only when the previous prompt reached 14,336 tokens, and only before a call. One
  large tool result can push the next prompt well past 14,336 before compaction gets a chance.
- The submission defaults (`swegemma/config.py:build_submission_limits`) are `max_output_tokens`
  16384 and `thinking_budget` 4096, merged into every agent. With 16,384 output tokens, a prompt
  above 16,384 tokens is rejected. In the paper's A/B this killed 3 and 2 of 37 task-runs per pass.
- Rule: keep `max_output_tokens` <= 32768 − (14,336 + the largest plausible tool-result growth).
  An 8,192 cap removed the deaths in our runs.

## 6. Instruction templating (`google/adk/utils/instructions_utils.py:inject_session_state`)

- The `instruction` is re-rendered from session state on every LLM call. The harness puts
  `problem_description` and `hints` (when non-empty) in state, so `{problem_description}` and
  `{hints?}` work.
- The regex is `{+[^{}]*}+`. A group whose name is a valid state name is looked up: a missing name
  raises `KeyError` (on every call, so the task fails); `{name?}` renders as empty. Any other group
  (`{"a": 1}`, `{}`, `{a b}`) is left literal. Substituted values are not re-scanned.
- So one stray `{word}` in a prompt fails every LLM call. Escape literal braces that look like
  identifiers.

## 7. Executable ADK spec (`harness/adk_spec/`)

`test_compaction_spec.py` has 28 tests that run the real ADK code path: `Runner` → `LlmAgent` flow →
`CompactionRequestProcessor` → window selection → `LlmEventSummarizer` → re-injection into the next request.
Only the model is scripted (a `BaseLlm` returning marked thoughts, `read_file` calls and chosen
`prompt_token_count`s; a `LiteLlm` with a fake client for the transport tests). No network, no GPU, about
3 s. `test_handoff_spec.py` adds 7 tests (`h*`) of what a second `LlmAgent` sees after a hand-off: after
`transfer_to_agent`, every event authored by another agent arrives as role=user `For context:` text (calls and
results as Python dict repr) and that agent's thoughts are dropped (`flows/llm_flows/contents.py`,
`_present_other_agent_message`); an `AgentTool` sub-agent runs in a fresh session with its own instruction
(session state templated) and only the request string, so neither the task nor anything the caller read.
Every test asserts what the pinned 1.36.1 does, so all 35 pass there; on another version a FAIL marks a
behaviour upstream changed. It tests mechanics, not summary quality (that is P-E5, [pe5.md](pe5.md)).

Groups: `s*` what the summarizer receives (§4), `r*` what the agent sees after compaction, `t*`/`l*` the
trigger, `o*` an orphaned call, `v*` visibility in traces (§4.2 of the paper), `k*` model settings the
summarizer inherits, `h*` hand-offs.

Pinned version (the runner env already has the scorer's google-adk 1.36.1, google-genai 2.11.0, litellm 1.83.7):

```bash
uv run --project harness/runner python -m pytest -c harness/adk_spec/pytest.ini harness/adk_spec
```

Version matrix (one venv per version; `data/venvs/` is git-ignored):

```bash
uv venv -p 3.12 data/venvs/adk-1.36.1 && uv pip install -p data/venvs/adk-1.36.1/bin/python "google-adk==1.36.1" "google-genai==2.11.0" "litellm==1.83.7" pytest
```

```bash
uv venv -p 3.12 data/venvs/adk-2.13.0 && uv pip install -p data/venvs/adk-2.13.0/bin/python "google-adk==2.13.0" "litellm==1.83.7" pytest
```

```bash
python3 harness/adk_spec/run_matrix.py data/venvs/adk-1.36.1 data/venvs/adk-2.13.0
```

Add `google-adk==2.2.0`, or upstream main installed from a clone, the same way. The authors' matrix
(1.36.1, 2.2.0, 2.13.0, main 1c84bb95) is `data/audit/adk-spec/matrix.md`. What its FAILs mean:

| Tests failing from | Upstream change | Effect |
|---|---|---|
| `s03`–`s06`, `s12`: 2.2.0 | bdb5582 | the summarizer receives tool calls and truncated tool responses, labelled thoughts, real newlines and a longer prompt |
| `r01`, `r02`: 2.13.0 (first in 2.7.0) | c20ccefe | the summary returns as the agent's own `role=model` turn, and the summarizer's thought part stays in it (sent via LiteLlm as `reasoning_content`, `k04`) |
| `o01`: 2.13.0 (first in 2.12.0) | fix for upstream issue #7236 | an unanswered function call no longer blocks every later compaction |
| `l01`: main only | a72a8ed3 | the trigger counts a large tool result before the next call instead of one call later |
| `h07`: 2.13.0 (not bisected) | — | only the hand-off wording changes (a quoted-transcript block); the other agent's turns are still role=user text without its thoughts (`h01`–`h04` pass) |

`s08` (the previous summarizer's thinking is in the next summarizer's input) passes on every version tested,
main included.
