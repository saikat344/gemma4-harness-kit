# P-E5: serializer replay (text-only 1.36.1 vs typed 2.2.0)

Paper §4.1 / Table 7. Implementation: `harness/analysis/pe5.py` (`prepare`, `run`, `report`); its docstring
records the exact fact rules, including the deviations from this spec noted below. Shipped results:
`data/audit/pe5/`.

**Question.** When Gemma 4 31B summarizes the same compacted window, how much task and working-state
information survives (a) with google-adk 1.36.1's text-only serializer (what the scorer runs) and (b) with the
2.2.0 serializer, which also includes thoughts, tool calls and truncated tool responses? This separates loss
*before* summarization (B1: the fact never reaches the summarizer) from loss *during* it (B2: the fact reaches
it, but the summary drops it).

## Inputs

- Runner logs (`runs/<name>/tasks/*/`, [formats.md](formats.md)) of an agent whose runs compact. The paper
  uses the two baseline passes of its A/B (P-E1, scorer serving flags). Take the **first** compaction of each
  task-run that has one, and sample 20 distinct tasks, stratified by repo, with seed 20261103 (all tasks if
  fewer than 20).
- Per sample: `compactions.jsonl` (`after_llm_call_index`, events in window, the logged summary) and
  `llm_calls.jsonl` (the last agent request before the compaction, and the logged `kind=summarizer` request).

## Reconstructing the window

1. Take the messages of the agent request at `after_llm_call_index` (OpenAI format, exactly as ADK sent it).
2. Rebuild ADK `Event`s from them, dropping the system message: assistant `content` → a text part;
   `reasoning_content` → a `thought=True` text part; `tool_calls` → `function_call` parts; `tool` messages →
   `function_response` parts; user messages → text parts. Drop the trailing events that compaction retains (5,
   with the call/response pairing rule of `google/adk/apps/compaction.py` in 1.36.1). The window is what
   remains (only first compactions are used, so there is no previous summary to seed).
3. **Validity check (before any GPU time):** the 1.36.1 serializer on the rebuilt events must reproduce the
   logged summarizer request's history text, ignoring whitespace. Report how many samples match; drop the
   others and say why.

## Arms

- **T (text-only):** `LlmEventSummarizer._format_events_for_prompt` from google-adk 1.36.1 (the pinned
  runner environment).
- **Y (typed):** the same function from google-adk 2.2.0 (upstream commit bdb5582), vendored with its
  truncation constant, attribution and Apache-2.0 header in `harness/analysis/adk220_format.py`; 2.x is not
  installed.
- Both arms otherwise send the identical summarizer request: ADK's default summarizer prompt
  ([harness-contracts.md](harness-contracts.md) §4), one user message, no system instruction, and the `params`
  copied from the logged summarizer request, to the same served model.
- 2 samples per arm per window: 20 × 2 × 2 = 80 calls.

## Facts (defined mechanically, frozen before the GPU run)

Per sample, written to `facts.jsonl` before any arm runs:

- **Issue facts (≤5):** identifiers from the problem statement, in priority order: backticked spans; dotted
  or CamelCase names; file paths; error class names. Tokens shorter than 4 characters are dropped.
  *Implementation deviation:* tokens equal to the repository owner or name, Python keywords and builtin names
  are dropped, and a candidate contained in (or containing) an already chosen fact is skipped.
- **Tool facts (≤5), from the window's tool calls and results:** the target file of the last `edit_file`;
  failing test ids from pytest output (`FAILED a::b`, `ERROR a::b`); exception class names in tool output;
  file paths read or edited. *Implementation deviation:* exception names count only when raised in
  `run_command` output (not when they merely appear in source read with `read_file`).

## Measures

Per arm, pooled over samples (match = case-insensitive substring):

- **availability:** the fact appears in the summarizer *input*. B1 predicts tool facts are mostly unavailable
  in T; they can only be present if the model restated them in text or thoughts.
- **retention:** the fact appears in the *summary*, reported as x/N (all facts) and x/available (conditional
  on availability).
- input tokens, output tokens and latency per call.

## Outputs

`<out>/{facts.jsonl, requests.jsonl, calls.jsonl, results.json}` and a generated markdown draft
(`report --doc`, default `<out>/pe5-results.md`) with the validity count, per-arm issue and tool
availability/retention, tokens and latency, and 2–3 shortened example summary pairs. Paper Table 7 reports
issue facts retained x/N and tool facts retained y/M per arm, plus mean input tokens per arm (the cost of the
typed serializer).

## Out of scope

The downstream resolve effect (it would need a full agent A/B with a patched ADK), later compactions (rolling
summaries), and summarizer prompt variants.
