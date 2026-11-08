# Telemetry and output formats

Schemas of the files written by the runner (`python -m harness.runner`) and the grader
(`python -m harness.grader`). Field lists are taken from the code: `harness/runner/capture.py`,
`worker.py`, `cli.py`, `report.py`, `checkpoints.py`, `export_sft.py`, and `harness/grader/cli.py`.

## Privacy: do not publish raw run directories of private tasks

- `llm_calls.jsonl` holds every request exactly as sent: the system instruction (your agent
  prompt), the full problem statement, file contents, test output, and the model's complete
  output including reasoning.
- `compactions.jsonl` holds summary text that paraphrases the task. `trace_legacy.json`,
  `traces/`, `logs/` and `patch.diff` hold the same material in other forms. `result.json` holds
  up to 50 tool-error messages (500 characters each).
- `run.json` records the host name, absolute data and submission paths, the endpoint URL and the
  argv (the API key is redacted).
- The grader's `logs/<id>.log` holds the full pytest output.

Share aggregates (`summary.json`, grader `summary.json`, per-task verdicts) instead. The analyzer
reports in [reliability.md](reliability.md) can quote argument fragments; check them before
publishing.

## Runner: `runs/<name>/`

```
run.json  predictions.jsonl  summary.json  summary.md  submission/
tasks/<id>/ result.json llm_calls.jsonl compactions.jsonl patch.diff spec.json
            trace_legacy.json traces/trace_<id>.json logs/<id>.log worker.log
            env_fixes.log containers.txt [checkpoints.jsonl checkpoints/<t>.patch]
attempts/   (old task dirs after --rerun-all)
```

### `tasks/<id>/llm_calls.jsonl`

One JSON line per LLM call, agent and summarizer alike, written and flushed as each call
completes (so it survives a killed worker). Recorded by wrapping
`google.adk.models.lite_llm.LiteLLMClient.acompletion`, the single path for agent calls and the
compaction summarizer. Observer only.

| Field | Content |
|---|---|
| `call_index` | 1-based order within the task |
| `kind` | `summarizer` if the request has no tools and its first text contains ADK's summarizer prompt, else `agent` |
| `t_start` | Unix time at call start |
| `budget_before` | `tool_calls_used`, `max_tool_calls`, `llm_calls_used`, `max_turns`, `agent_elapsed_s`, `remaining_time_s`, `patch_submitted` |
| `request.model` | LiteLLM model id, e.g. `openai/<served id>` or `openai/<lora name>` |
| `request.messages` | OpenAI-format messages exactly as sent (after compaction and the adk-submission reasoning patch) |
| `request.tools` | tool declarations as sent |
| `request.params` | the remaining call kwargs except `api_key`: `extra_body` (`chat_template_kwargs`, `thinking_token_budget`), `temperature`, `top_p`, `max_completion_tokens`, ... |
| `latency_s` | wall seconds of the call |
| `response` | the full ChatCompletion as JSON, including `reasoning_content` and parsed `tool_calls`; `null` plus `note` for a streaming call |
| `finish_reason`, `usage` | copied from the first choice and the response |
| `error` | `Type: message` if the call raised; then there is no `response` |

### `tasks/<id>/compactions.jsonl`

One line per ADK compaction event, observed at `InMemorySessionService.append_event` (the
official trace never contains these; see [harness-contracts.md](harness-contracts.md)).

| Field | Content |
|---|---|
| `t` | Unix time |
| `after_llm_call_index` | `call_index` of the last LLM call before the compaction (the summarizer call) |
| `start_timestamp`, `end_timestamp` | the compacted window, in ADK event timestamps |
| `events_in_window` | session events inside the window |
| `session_events_total` | session events at that moment |
| `summary` | the summary text that replaces the window |
| `budget` | same fields as `budget_before` above |

### `tasks/<id>/result.json`

Written atomically by the worker; the orchestrator writes it for workers it had to stop.

Completed task (`status: done`):

| Field | Content |
|---|---|
| `status`, `instance_id`, `repo`, `task_index`, `started_at` | identity and start time |
| `patch_chars`, `patch_files`, `empty_patch` | final patch stats |
| `patch_submitted` | `submit_patch` was called at least once |
| `patch_source` | `submit_patch`, `fallback_diff` (working-tree diff, no submission) or `none` |
| `agent_error` | the harness's agent error string, or null |
| `termination` | `time_budget`, `turn_budget`, `tool_budget`, `agent_exception`, `submitted`, `no_submit_no_diff`, `nudges_exhausted`, `ended_without_submit`, `other`. Budget reasons take precedence over `submitted`, and `submitted` over `agent_exception` (see `agent_exception_after_submit` in the same file), so a task that submitted and then hit the time cap is `time_budget`. |
| `timings` | `setup_s` (sandbox setup), `agent_s` (the harness's agent clock), `total_s` |
| `budget_final` | budget state at the end (fields as `budget_before`) |
| `harness_tool_calls_used`, `harness_llm_calls_used` | the harness's own counters |
| `tool_call_breakdown`, `tool_calls_total` | calls per tool, from the trace |
| `tool_errors`, `tool_errors_total` | `{tool: {error_type: n}}`; ADK missing-argument rejections appear as `MissingArguments` |
| `tool_error_samples` | up to 50 `{tool, error_type, message, elapsed_s}` |
| `unparsed_tool_call_text` | text/thinking entries containing raw `<|tool_call>` |
| `nudges`, `thinking_entries` | from the trace |
| `compactions`, `compactions_in_trace` | lines in `compactions.jsonl`; compactions in the official trace (expected 0) |
| `llm_calls_agent`, `llm_calls_summarizer`, `llm_call_errors` | from `llm_calls.jsonl` |
| `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `total_tokens`, `max_prompt_tokens` | summed over successful calls |
| `finish_reasons` | `{reason: n}` |
| `llm_time_s`, `completion_tok_per_llm_s` | summed latency; completion tokens per LLM second |
| `worker_wall_s` | worker process wall time |

Other statuses: `killed` (SIGTERM/SIGINT), `worker_timeout` (stopped by the orchestrator's hard
timeout), `infra_error` (exception, with `traceback`, or a worker that exited without a result).
These carry `status`, `instance_id`, `error` and, when the worker wrote them, `worker_wall_s`.

### `tasks/<id>/checkpoints.jsonl` (with `--checkpoints`)

One line per cap t (minutes): `t`, `elapsed_actual`, `reason` (`crossed` or `ended_before`),
`where` (`before_call`, `after_response`, `end`), `call_index`, `has_submitted`, `submitted_sha`,
`worktree_sha`, `effective_source` (`submitted`, `worktree`, `none`, `error`), `effective_sha`,
`effective_chars`, `capture_s`. The effective patch is in `checkpoints/<t>.patch` (empty file =
none). See [runner.md](runner.md) section 4.

### Other per-task files

- `patch.diff`: the final graded patch (submission or fallback diff, already stripped).
- `trace_legacy.json`: the harness's ADK event trace (`SessionTrace` legacy format; entry types
  include `thinking`, `text`, `tool_call`, `tool_response`, and harness entries
  `system_instruction`, `task_prompt`, `continuation_nudge`, each with `elapsed_s`).
- `traces/trace_<id>.json`: the same trace in the harness's ATIF v1.7 format (what `swegemma eval`
  writes).
- `logs/<id>.log` (harness console log), `worker.log`, `env_fixes.log` (one block per repair
  command), `containers.txt` (container ids, for cleanup), `spec.json` (worker input).

### `run.json`

`name`, `created_at`, `argv` (API key redacted), `git` (harness revision, `-dirty` if modified),
`host`, `platform`, `versions` (swegemma, adk-submission, adk-eval-core, google-adk, google-genai,
litellm, docker, python), `endpoint`, `served_model`, `endpoint_models`, `submission_source`,
`submission_sha256`, `submission` (declared model, adapters, ...), `budgets` (resolved, with
`overrides`), `compaction`, `context_cache`, `image` (`image`, `id`, `arch`, `created`), `data`
(tasks, snapshots, graphs, embeddings, wheels paths), `task_ids`, `concurrency`, `timing_scale`,
`env_fixes`, `checkpoints`, `deviations` (list of strings), `sessions` (one per invocation:
`started_at`, `argv`, `n_tasks`, `ended_at`, `wall_s`).

### `predictions.jsonl`

One row per selected task, in task order: `instance_id`, `patch`, `id`, `prediction`, `status`.
`patch` and `prediction` hold the same text (both naming conventions, so the grader and
Kaggle-style tools read it); an empty patch, or a task that is not `done`, is `NO_PATCH`.
`status` is the task's `result.json` status or `missing`. Checkpoint exports
(`at/predictions_at_<t>.jsonl`) add `checkpoint_t`, `checkpoint_reason` and `checkpoint_source`.

### `summary.json` and `summary.md`

`summary.json` (rebuilt by `python -m harness.runner report`):
- run: `run`, `served_model`, `submission_sha256`, `budgets`, `env_fixes`, `deviations`,
  `n_selected`, `concurrency`, `wall_time_s` (sum over sessions);
- outcome: `status_counts`, `infra_failures` (`{id: error}`), `empty_patch_rate`,
  `empty_patches`, `submitted_rate`, `fallback_diff_patches`, `termination`;
- distributions (`n`, `mean`, `median`, `p90`, `max`, `sum`): `agent_minutes_per_task`,
  `setup_seconds_per_task`, `total_seconds_per_task`, `tool_calls_per_task`,
  `llm_calls_per_task`, `tokens_per_task`, `max_prompt_tokens`, `completion_tok_per_llm_s`;
- tools: `tool_call_breakdown`, `tool_errors`, `tool_errors_total`, `edit_file_errors`,
  `edit_file_errors_by_type`, `edit_file_error_rate` (excluding `BudgetExceeded`),
  `command_timeouts`, `budget_exceeded_tool_responses`, `agent_time_budget_hits`;
- model: `unparsed_tool_call_text`, `nudges_total`, `compactions_total`, `llm_call_errors`,
  `finish_reasons`;
- `kaggle_projection`: `method` (900 s per 1M tokens, a community calibration), `per_task_s`,
  `tasks_over_330s`, `projected_hours_for_120_tasks`, `timing_scale`, `scaled_agent_s`,
  `scaled_hours_for_120_tasks`;
- `per_task`: one row per task (`instance_id`, `status`, `repo`, `termination`, `patch_chars`,
  `tool_calls`, `tool_errors`, `llm_calls`, `compactions`, `nudges`, `total_tokens`,
  `max_prompt_tokens`, `setup_s`, `agent_s`, `scaled_agent_s`, `kaggle_tokens_s`, and `resolved`
  when grades exist);
- with grades (`<run>/grades.jsonl` or `--grades`): `resolved`, `graded`, `resolved_by_repo`.

`summary.md` renders the same content as a bullet list plus a per-task table, including the line
`- env fixes: on|off`.

### SFT export (`export-sft`)

One JSON line per maximal conversation segment: `messages` (the request messages plus the
assistant reply), `tools`, and `meta` (`call_index`, `kind`, `finish_reason`, `usage`, sampling
`params`, `run`, `instance_id`, `resolved`, `submission_sha256`). It inherits the privacy notes
above. `--only-resolved` matches grades by instance id, not by patch, so use one grades file per run.

## Grader: next to `--out`

### `grades.jsonl`

One row per task, appended as tasks finish and rewritten in task order at the end.

| Field | Content |
|---|---|
| `instance_id`, `repo`, `mode` | `mode` is `predictions`, `gold` or `empty` |
| `passed` | the official `verify_task` verdict |
| `error_class` | `none`, `tests_failed`, `collection_or_usage_error`, `no_tests_collected`, `timeout`, `required_test_not_passed`, `junit_invalid`, `patch_apply_failed`, `test_patch_apply_failed`, `missing_test_spec`, `eval_error`, `empty_patch`, `missing_snapshot`, `setup_error`, `grader_timeout`, `grader_crash`, or `exit_<code>` |
| `error` | the harness error message, else the pytest summary line (up to 300 characters) |
| `test_exit_code` | pytest exit code; -1 when tests did not run |
| `pytest_counts` | `{passed, failed, error, skipped, ...}` from the last summary line |
| `failed_tests` | up to 50 `FAILED`/`ERROR` node ids |
| `required_tests` | FAIL_TO_PASS + PASS_TO_PASS, else the test names added by `test_patch` |
| `target_test_files` | files touched by `test_patch` (the files pytest runs) |
| `patch_lines`, `patch_lines_after_strip`, `patch_empty` | patch size before and after the official stripping |
| `env_fixes` | number of repair commands applied (0 with `--no-env-fixes`) |
| `duration_seconds` | per-task wall time including container setup |

### `summary.json`

`mode`, `image`, `image_arch`, `network`, `tasks_path`, `predictions`, `env_fixes` (bool),
`skipped_missing_snapshot`, `wall_seconds`, `workers`, `total`, `resolved`, `resolution_rate`,
`by_repo` (`{repo: {total, resolved}}`), `error_classes` (`{class: n}`), `failed_ids`.

Also written: `logs/<id>.log` (inputs, repair output, full pytest stdout and stderr) and
`grader.log`.
