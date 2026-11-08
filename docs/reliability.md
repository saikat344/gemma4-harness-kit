# Reliability probes: tool-call parser and trajectory analyzer

Two community reports say Gemma 4's agentic failures are mechanical rather than a lack of ability:

- **R1** ([Kaggle discussion 747168](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747168)):
  vLLM's gemma4 tool-call parser mangles `edit_file` calls whose `old_string` contains `"""`
  docstrings. Argument keys come out like `<|"|>,filepath`, ADK rejects the call ("mandatory input
  parameters are not present"), and the model retries the identical call.
- **R2** ([paper-track discussion 751430](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/751430), unverified):
  agents get stuck in A-B-A edit loops as error traces pile up.

The scorer's server, parser and tools are fixed, so the questions are whether the parser or the
model is at fault, which outputs go wrong, how often it happens, and whether it separates resolved
from unresolved tasks. Two tools answer them:

| Tool | Answers | Runs where |
|---|---|---|
| `harness/repro/parser_cases.py` | Does vLLM 0.19.1's gemma4 tool/reasoning parser round-trip well-formed raw outputs? What does it do with model-side format slips? | the vLLM 0.19.1 venv (CPU only), or stub mode without vLLM |
| `harness/analysis/reliability.py` | How often do identical retries, ADK rejections, edit failures, A-B-A reverts and runaway generations happen in run output, and do they correlate with resolved/unresolved? | anywhere, on runner output (`runs/<name>/`) |

## 1. Parser reproduction (R1)

### 1.1 What the vLLM 0.19.1 source says

- Files: `vllm/tool_parsers/gemma4_tool_parser.py` (`Gemma4ToolParser`, registered as `gemma4`),
  `vllm/reasoning/gemma4_reasoning_parser.py` (`Gemma4ReasoningParser`). The `gemma4_utils.py`
  helpers are offline utilities the server does not use.
- Wire format: `<|tool_call>call:func_name{key:<|"|>string<|"|>,num:42,flag:true}<tool_call|>`.
  Strings are wrapped in the special token `<|"|>`; keys and scalars are bare; calls are
  concatenated with no separator; nested `{...}` and `[...]` are supported. Thinking is
  `<|channel>thought\n...<channel|>`. Token ids (real tokenizer): `<|tool_call>` 48,
  `<tool_call|>` 49, `<|"|>` 52, `<|channel>` 100, `<channel|>` 101.
- Non-streaming path (the scorer's ADK call does not stream): the server runs
  `reasoning_parser.extract_reasoning` on `output.text`, then `tool_parser.extract_tool_calls` on
  the rest. The tool parser uses the regex `<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>`
  (DOTALL, non-greedy) and `_parse_gemma4_args`: a string value ends at the **first** `<|"|>`
  after its opening one. There is no escaping and no recovery. `skip_special_tokens` is forced off.
- No match (missing `<tool_call|>`, `call:name {` with a space, `}\n<tool_call|>`, `<turn|>` as
  terminator) leaves the raw `<|tool_call>...` text in `content`. Exceptions are caught, logged and
  treated the same way. The `tools` schema is never consulted.
- Consequence: `"""`, quotes, backslashes, braces, `<|` and `|>` inside a string are harmless **as
  long as the model emits exactly one `<|"|>` before and after each string**. The reported key
  `<|"|>,filepath` needs a stray, doubled or missing delimiter emitted by the model. The parser
  amplifies such a slip; it does not cause it.

### 1.2 What the official harness does with a rejected or mangled call

Read from the swegemma, adk-eval-core and google-adk 1.36.1 sources.

- The file tools take `filepath` (not `file_path`): `edit_file(filepath, old_string, new_string,
  allow_multiple=False)`, `write_file(filepath, content)`, `read_file`; `run_command(command)`.
- ADK `FunctionTool.run_async` drops unknown keys silently and rejects a call only when a
  mandatory parameter is absent. The model then gets an error that names the missing parameter
  and invites a retry; it does not say the key was malformed.
- Unknown tool name: ADK raises `ValueError("Tool 'x' not found ...")`; swegemma's
  `ToolErrorPlugin` turns it into `{"status":"error","error_type":"ValueError",...}`.
- Accounting: the swegemma tool body (`@budget_gated`) never runs for a rejected call, so the tool
  budget does not move. It still costs an LLM turn and wall-clock time. A rejected call also
  counts as a turn with a tool call, which resets the nudge counter, so a rejection loop ends only
  on the time or turn budget.
- **Nothing de-duplicates retries.** The three wheels contain no identical-call or loop detection.
  `ModelRetryPlugin` retries only transport errors (429/5xx).
- Mangling that keeps every required key (e.g. JSON-escaped quotes) passes ADK and fails later as
  `old_string not found`, which looks like an ordinary edit failure. The analyzer reports both.
- ADK re-sends the model's own (garbled) call as history, and the chat template renders it back,
  so in-context copying is a plausible reason for identical retries (inferred).

### 1.3 Cases and hypotheses

`python -m harness.repro.parser_cases --list` prints all case ids. 118 **canonical** cases cover
`edit_file`, `write_file` and `run_command` with: triple-quoted docstrings and every quote mix;
escapes, CRLF and tabs; unicode (CJK, emoji, zero-width, RTL override); JSON-looking text and
braces; `<|`/`|>` sequences; very long strings (up to a 200 KB value, three big calls per turn);
typed arguments and edge values; several calls per turn; calls after thinking. 15 **glitch**
cases encode model-side slips:

| Id | Slip | Expected parser outcome |
|---|---|---|
| H1 | doubled or stray `<|"|>` after a string ending in `"` | the next key becomes `<|"|>,filepath`; ADK rejects the call |
| H2 | other delimiter slips (a `"` emitted as `<|"|>`, a plain `"` closing the string, JSON-escaped quotes, a JSON object) | a required key is lost (rejected) or the call passes corrupted |
| H3 | wrong terminator (`<turn|>`, `<|tool_response>`), whitespace before the end tag, `call:name {` | not parsed; raw text stays in `content` |
| H4 | a call before `<channel|>` (thinking not closed) | not parsed; the call lands in `reasoning` |
| H5 | `finish_reason=length` inside a long string | not parsed |
| collision | values containing the literal tags | delimiter and end-tag collisions are mangled; the start tag survives non-streaming |

For every case the report gives the parse status (OK / MANGLED with a diff / NOT_PARSED /
EXCEPTION), the streaming result (token-sized deltas through `extract_tool_calls_streaming`), a
tokenizer round trip, the chat-template history round trip, what ADK would do with the parsed
arguments, and the time. A canonical failure is a parser bug (exit code 1); a glitch case that
parses fine is reported as a SURPRISE.

### 1.4 Running it

```bash
~/venvs/vllm/bin/python -m harness.repro.parser_cases --tokenizer <weights>/gemma-4-31b-it-qat-w4a16-ct/2 \
    --json runs/repro/parser_cases.json --md runs/repro/parser_cases.md     # about 1-2 min, CPU only
echo exit=$?   # 0: all canonical cases round-trip; 1: a canonical case failed; 2: vllm or tokenizer missing
```

- Run from the repo root in the vLLM venv (it imports `vllm.tool_parsers.ToolParserManager`,
  `vllm.reasoning.ReasoningParserManager` and `vllm.tokenizers.get_tokenizer`). It never touches
  CUDA. The report header prints `vllm_version`, the parser module paths and the special-token ids.
- Without vLLM: `--vllm-src DIR` loads the real parser files extracted from the vLLM 0.19.1 wheel
  and stubs the rest (`harness/repro/vllm_stub.py`); `--fake` uses a tiny built-in parser
  (harness self-test). Both mark the report as not authoritative.

### 1.5 Results

- **Real tokenizer, vLLM 0.19.1 (CPU, 2026-11-02):** canonical 118/118 round-trip, including
  docstrings and all quote mixes; glitch 15/15 behave as predicted (the one glitch that parses,
  `collision.start_tag_in_value`, is the expected verdict). Streaming: 0 canonical failures. Two
  chat-template history round trips fail, both for literal `<|"|>` or `<tool_call|>` inside an
  argument value.
- **Stub mode (real parser source, fake tokenizer):** the same 118/118, and streaming matches for
  all 114 cases below 60k characters (larger cases are skipped: the streaming parser re-parses the
  whole argument string on every token, which is quadratic).
- Reading: the parser is correct for the format it expects and brittle to any slip. Mangled keys
  in real runs therefore point to model-side format slips (inferred from the two results together).

## 2. The analyzer (`harness/analysis/reliability.py`)

```bash
uv run --project harness/runner python -m harness.analysis.reliability runs/<a> runs/<b> --out runs/reliability
#   a parent dir holding several run dirs also works
#   --grades FILE (default <run>/grades.jsonl)   --all-tasks   --json-only   --no-task-text   thresholds below
```

Output: markdown (stdout, and `<out>/reliability.md`) and `<out>/reliability.json` (per-task
metrics, flags, pooled and per-run aggregates, thresholds, mode definitions). The markdown can
quote tool-call keys and short argument fragments from the runs, so treat it like the run
directories ([formats.md](formats.md)).

### 2.1 Sources

The primary source is `tasks/<id>/llm_calls.jsonl`: the tool calls as the parser produced them,
the tool results ADK fed back (the `tool` messages of the next request), latency, completion
tokens, finish reason, and the request's `tools` and `max_completion_tokens`. It is written line by
line, so it survives a killed worker. Results of the last turn (never fed back) come from
`trace_legacy.json` when it exists; otherwise a call missing a required parameter or naming an
unknown tool is classified as rejected (inferred). `trace_legacy.json` alone is the fallback when
there is no call log. `compactions.jsonl` and `result.json` supply compactions and outcome.

### 2.2 Metric definitions

A **turn** is one LLM response with tool calls. A call's **kind** comes from its tool result:
`ok`, `rejected_missing` (ADK: mandatory parameters not present), `rejected_unknown_tool`,
`rejected_other` (invalid arguments), `edit_fail` (`FileEditError`), `write_fail`, `budget`,
`timeout`, `cmd_error` (non-zero exit of the command, e.g. failing pytest; not a harness failure),
`other_error`. "Rejected" is the first three. "Failed" is rejected + edit_fail + write_fail +
budget + timeout + other_error.

| Flag (per task) | Definition | Threshold option |
|---|---|---|
| `identical_repeat` | a call with the same tool and arguments as the previous call | none |
| `retry_loop` | a run of identical consecutive calls of length >= N | `--loop-min-run 3` |
| `identical_after_error` | an identical call right after the same call returned an error | none |
| `adk_rejected` | at least one rejected call | none |
| `parser_signature` | argument keys containing `<|`, `|>`, `"`, `,`, `:` or a space; a string value containing a gemma special token; or raw `<|tool_call>` text in assistant `content` | none |
| `edit_fail`, `edit_fail_heavy` | any edit_file failure; failures >= N or rate >= R (with >= 4 edit_file calls) | `--edit-fail-n 3 --edit-fail-rate 0.5` |
| `aba_revert_any`, `aba_oscillation` | >= 1 revert; reverts >= N or a reverted change re-applied (below) | `--aba-min-reverts 2 --aba-sim 0.8 --aba-min-chars 3` |
| `runaway_generation` | `finish_reason=length`, or completion tokens >= F × the request's output cap (else the default cap) | `--runaway-frac 0.9 --default-cap 8192` |
| `long_tail_generation` | a call with >= N completion tokens | `--tail-tokens 4096` |
| `time_wasted_high` | seconds in failed or retried turns >= F × agent seconds | `--waste-frac 0.2` |
| `tool_budget_hit`, `budget_termination`, `nudged`, `worker_failure` | a `BudgetExceeded` result; ended by a time/tool/turn budget; nudged by the harness; worker killed, timed out or infra error | none |
| `empty_patch` | final patch empty (`NO_PATCH`) | none |
| `compaction`, `compaction_lost_problem` | a compaction fired; the first agent call after a compaction does not contain the verbatim probe (the first 120 characters of the problem statement from `tasks.jsonl`, or with `--no-task-text` the first paragraph after `Problem Statement:` in the first prompt) in its system or user messages | none |

- `compaction_lost_problem` tests verbatim presence only. A summary that paraphrases the issue
  still sets it, so it does not measure whether the issue's content survived (that is B2,
  measured by P-E5; see [harness-contracts.md](harness-contracts.md)).
- **edit_file failure causes:** `old_string_not_found`, `multiple_matches`, `empty_old_string`,
  `identical_old_new`, `file_empty`, `bad_path`, `file_not_found`, `other`. ADK rejections of
  edit_file calls are counted separately (`edit_file_rejected_by_adk`).
- **A-B-A:** each successful `edit_file` becomes a change (removed text, added text) taken from
  the unified diff in its result (fallback: a line diff of the old/new arguments); `write_file`
  becomes a change against the previous write of the same path. Edit *j* reverts an earlier
  unreverted edit *k* on the same file when `sim(removed_j, added_k)` and `sim(added_j, removed_k)`
  both average >= 0.8 (difflib ratio) **and** that direction beats "same direction" by 0.1, so
  incremental edits of one line (1 -> 2 -> 3) are not reverts. Strings shorter than 25 characters
  must match exactly. A *re-apply* is a later edit that repeats a reverted change. Failed edits
  are ignored. `git checkout/restore/stash/reset --hard/revert/apply -R` commands are counted
  separately as `revert_commands`.
- **Time:** per turn, LLM latency plus the gap to the next recorded call (tool execution, ADK
  overhead). A turn is *wasted* when all its calls failed or repeat an earlier call (the previous
  call, or any earlier failed call). Wasted seconds are bucketed (rejected > retry > edit failure
  > budget/timeout > other) so they are not double counted; `share_of_agent` divides by
  `result.json` `timings.agent_s`.
- **Generation tail:** p50/p90/p99/max completion tokens per task, calls at the cap, calls with
  `finish_reason=length`, and their latency.
- **Correlation** (per mode, pooled and per run): tasks flagged, resolve rate of flagged vs
  unflagged tasks, share of unresolved and of resolved tasks affected, two-sided Fisher exact p.
  Only graded tasks count. With a few dozen tasks per run, treat p as a rough guide and prefer
  paired comparisons between runs. A mode common in both resolved and unresolved tasks is
  background; one common in unresolved and rare in resolved is a lead.

Caveats: the last turn's gap is unknown (0); repeats in a different order (X Y X Y) are not
consecutive; A-B-A is a heuristic on change text, not on file states, so oscillation between three
variants is caught only pairwise.

The paper's Table 6 reports these metrics for real runs; only aggregates are published.

## 3. Tests

```bash
uv run --project harness/runner python -m pytest harness/analysis/tests harness/repro/tests -q
```

- `harness/analysis/tests/test_reliability.py` builds synthetic run directories
  (`harness/analysis/tests/synth.py`), one scenario per detector. The analyzer can be run on them
  without any real trajectory.
- `harness/analysis/tests/test_e2e_mock.py` (needs Docker and a sandbox image, about 30 s) runs the
  real runner against two `mock_openai` modes: `reliability` (three identical `edit_file` calls
  with the key `<|"|>,filepath`, an unknown tool, a non-matching edit twice, the gold hunk applied,
  reverted, re-applied, reverted and applied again, one 8192-token turn, then submit) and `loop`
  (the mangled call forever). It checks that every detector fires on real runner output.
- `harness/repro/tests` covers the parser harness with the fake parser, deliberately flawed
  parsers, the real parser source (skipped when no extracted vLLM source is present) and the CLI.
