# Agent runner and vLLM serving

`harness/runner` executes a submission directory or zip (`agent.yaml`, prompts, skills, adapters,
`eval_config.yaml`) against tasks the way the Kaggle scorer's Stage 1 does, as reconstructed from
the official getting-started notebook (NB1). It talks to a self-hosted OpenAI-compatible vLLM
endpoint (`harness/serve`) and writes predictions for the [grader](grader.md), plus per-call
telemetry. Task subsets live in `harness/subsets/`.

Status (2026-11-02): tested end to end against a mock OpenAI server with real Docker sandboxes,
and used for real Gemma 4 31B runs on one A100-80GB and on one 24 GB RTX 3090.

## 1. Architecture

```
python -m harness.runner run ...                 (orchestrator, harness/runner/cli.py)
  ├─ copies the submission into runs/<name>/submission/ and validates it (as NB1 does)
  ├─ budgets = eval_config.yaml, parsed exactly like NB1 (core.load_budgets)
  ├─ GET <endpoint>/models → base model id + LoRA ids; refuses to run if a submission adapter isn't served
  └─ for each task (asyncio semaphore = --concurrency):
       subprocess  python -m harness.runner.worker tasks/<id>/spec.json     (ONE PROCESS PER TASK)
         ├─ EvalConfig(... sandbox='docker', NB1 compaction + context-cache configs ...)
         ├─ model registry = adk_submission.VllmServer.create_model_registry(...)   (external endpoint)
         ├─ swegemma.harness.agent_runner.run_agent_sandbox(...)   ← the official Phase 1, unchanged
         ├─ observers: LiteLLMClient.acompletion wrapper            → llm_calls.jsonl
         │             InMemorySessionService.append_event wrapper   → compactions.jsonl
         │             ContainerManager.start wrapper                → containers.txt (cleanup after a kill)
         └─ result.json, patch.diff, trace files
  └─ report.build → predictions.jsonl, summary.json, summary.md
```

- Grading (Phase 2) is not in the runner. Grade `predictions.jsonl` with `harness.grader`, which
  runs the official `verify_task`.
- One process per task, because the swegemma tools are synchronous: `run_command` blocks the
  event loop, so in-process concurrency would serialise tasks and distort the agent clock.
- Workers get a hard wall-clock kill at `max_time_minutes*60 + --setup-allowance` (default
  900 s): SIGTERM, then SIGKILL after 30 s, then removal of the containers in `containers.txt`.
- Observers never change a request or a result.
- **Environment repairs.** The agent sandbox uses `harness.common.envfix.FixingContainerManager`,
  the same per-task repairs the grader applies ([grader.md](grader.md)); `--no-env-fixes` turns
  them off. They run right after each `setup.py` call inside `run_agent_sandbox`, i.e. during
  sandbox setup and before `setup_baseline_commit`, so the agent's first tool call already sees
  the repaired environment. They only change files under `/usr/local/lib/python3.13/site-packages`,
  so `git status` is clean and `git diff HEAD` is empty at the agent's first command (verified on
  fastapi_9425, and by `test_env_fixes_in_agent_sandbox_fastapi`). `run.json` and `summary.json`
  carry `env_fixes`; `tasks/<id>/env_fixes.log` lists each command's output.
- Runner and grader match each other, not the scorer, which applies no repairs. Use
  `--no-env-fixes` in both to see the environment as shipped.

## 2. Setup

Needs x86_64 Linux with Docker (the sandbox is amd64 only), `uv`, and the data from
[data.md](data.md). The sandbox image is built by `harness/sandbox/build.sh` ([grader.md](grader.md)).

```bash
# runner venv (CPU torch; no vLLM). Resolves the three Kaggle wheels from data/wheelhouse.
uv sync --project harness/runner --python 3.12
uv run --project harness/runner python -m pytest harness/runner/tests -q      # mock tests, about 3 min

# vLLM venv, separate so vLLM's pins never touch the harness env
uv venv ~/venvs/vllm --python 3.12
VP=~/venvs/vllm/bin/python
uv pip install --python $VP --index-strategy unsafe-best-match \
   --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
uv pip install --python $VP data/wheelhouse/vllm-0.19.1-*.whl              # vLLM's other deps from PyPI
# pin everything the scorer pins, as NB1 does (no-deps reinstall of the wheelhouse):
uv pip install --python $VP --no-deps --reinstall $(ls data/wheelhouse/*.whl | grep -v -E 'swegemma|adk_submission|adk_eval_core')
uv pip install --python $VP 'huggingface-hub==1.33.0'   # PyPI's 2.x breaks transformers 5.13.1 (2026-11-02)
$VP -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"   # 0.19.1 5.13.1
```

On WSL2, Triton needs `build-essential python3.12-dev`. Weights: [data.md](data.md).

## 3. Serving (`harness/serve`)

```bash
export VLLM_PYTHON=~/venvs/vllm/bin/python
uv run --project harness/runner python -m harness.serve.serve \
  --model-dir <weights>/gemma-4-31b-it-qat-w4a16-ct/2 --profile 3090 [--submission <dir|zip>] [--print] [--log FILE]
uv run --project harness/runner python -m harness.serve.health_check --endpoint http://127.0.0.1:8000/v1 --long --bench 2
```

`harness/serve/serve.sh <model-dir> [args]` is a thin wrapper. `serve.py` builds the argv with the
official `adk_submission.VllmServer.build_cmd()` and NB1's `VllmConfig`, so the flags are the
scorer's: `--max-model-len 32768 --dtype bfloat16 --enable-auto-tool-choice --tool-call-parser
gemma4 --reasoning-parser gemma4 --reasoning-config {...} --no-scheduler-reserve-full-isl`, plus
the LoRA flags when the submission has adapters and the 12B `--hf-overrides`. The server
environment is NB1's. A profile then appends its flags and prints each one as a deviation;
`--print` prints the argv without starting, so it can be run anywhere. The base model id is the
model path (the scorer passes no `--served-model-name`); the runner discovers it from
`/v1/models`. LoRA module names are the adapter directory names. To change adapters, restart
vLLM with `--submission`.

| Profile | Added to the scorer argv | Use |
|---|---|---|
| `a100` | nothing (util 0.90, no TP) | one A100-80GB: the scorer argv minus `--tensor-parallel-size` |
| `4xl4` | `--tensor-parallel-size 4` | four L4s: the scorer's hardware class |
| `3090` (default) | util 0.945, `--max-num-seqs 1`, `--max-num-batched-tokens 512`, no image/video, `--compilation-config {"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}` | 31B on one 24 GB RTX 3090 at 32k with bf16 KV |
| `scorer` | nothing | 12B/E4B/E2B on one GPU |
| `small` | util 0.90, `--max-num-seqs 4`, no image/video | 12B/E4B/E2B with headroom |

Cloud launchers exist in the authors' tree but are not shipped; `serve.py --print` prints the
exact argv to run anywhere.

**31B on a 24 GB RTX 3090 (measured 2026-11-02, WSL2).** The checkpoint is 21.67 GiB (int4
linears 15.3, bf16 embeddings 2.6, a separate bf16 `lm_head` 2.6, vision tower 1.1); 18.4 GiB of
weights load on the GPU. WSL2 CUDA sees 22.76 of 24 GiB, so utilisation above 0.945 fails.
torch.compile costs about 2.6 GiB and breaks 32k, so the profile turns it off (`mode 0`) and keeps
one full CUDA graph for batch-1 decode. Result: 4.08 GiB of KV (4,448 tokens, 1.11x at 32k),
34.6 tok/s single stream, and the health check passes including a 28k-token prompt.
`--enforce-eager` instead gives ~10 tok/s. fp8 KV is not an option on Ampere here: vLLM forces
Triton attention for Gemma 4, Triton e4m3 is unsupported on sm86, and e5m2 is rejected with fp8
checkpoints. Serve one request at a time and run agents at `--concurrency 1`. Do not lower
`--max-model-len`: compaction at 14,336 prompt tokens plus the output cap must fit in 32,768
([harness-contracts.md](harness-contracts.md)).

`health_check.py` (stdlib only) checks `/health` and `/v1/models` (max_model_len 32768); a tool
call with thinking on that parses into `message.tool_calls` with no tool-call or channel markup
leaking into content; `reasoning_content`; a multi-turn tool-result round trip with reasoning
mirrored as `reasoning` (as `adk_submission` does); `enable_thinking=false`; the
`thinking_token_budget` (warning only); every LoRA id; `--long` (a ~28k-token prompt); `--bench N`
(decode tok/s).

## 4. Running

```bash
R="uv run --project harness/runner python -m harness.runner"
$R --submission data/kaggle/sample_submission --task-ids dev40 --endpoint http://127.0.0.1:8000/v1 \
   --out runs/<name> --concurrency 1
# resume after a crash or Ctrl-C: run the SAME command again (done tasks are skipped; killed ones re-run)
$R ... --retry-failed      # also re-run infra_error / worker_timeout tasks
$R ... --rerun-all         # fresh pass in the same dir (old task dirs move to runs/<name>/attempts/)
harness/grader/.venv/bin/python -m harness.grader --predictions runs/<name>/predictions.jsonl \
   --out runs/<name>/grades.jsonl --workers 2
$R report runs/<name> [--timing-scale X] [--grades runs/<name>/grades.jsonl]
$R export-sft runs/<name> --out sft/<name>.jsonl --grades runs/<name>/grades.jsonl --only-resolved
```

- `--task-ids` takes a subset name from `harness/subsets/` (`dev40`, `all129`), a file
  of ids, or a comma list. `--limit N` for a smoke test. `dev40` is 40 public tasks stratified by
  repository (seed 20261102); `unsound.txt` lists the 17 tasks to drop from agent comparisons
  (3 of them are in dev40).
- Data paths default to `$GEMMA4_DATA_DIR`, `$GEMMA4_SNAPSHOTS_DIR`, `$GEMMA4_WHEELS_DIR`, else
  `data/kaggle`. The image defaults to `$GEMMA4_SANDBOX_IMAGE`, else `swebench-sandbox:latest`.
- `--override max_tool_calls=40` (etc.) and `--no-compaction` are deliberate deviations,
  recorded in `run.json` and `summary.md`.
- Each task uses a 4 GiB / 2 vCPU sandbox. The agent clock is per task, but contention slows each
  task, so compare minutes per task only between runs at the same concurrency.
- `--status-dir DIR` writes a small live-progress JSON per run (counts, current task, ETA; no
  URLs, keys, prompts or paths). It never blocks or fails the run.

### Time checkpoints (`--checkpoints`)

One long-budget run tells you what would have been graded at shorter caps:

```bash
$R --submission <dir> --task-ids dev40 --out runs/<name> --override max_time_minutes=15 --checkpoints 5,7.5,10
$R checkpoints runs/<name> [--out-dir runs/<name>/at] [--at 5,10]     # -> at/predictions_at_<t>.jsonl
```

- At cap t the graded patch is the last `submit_patch` snapshot before t (even an empty one, as
  the scorer does), else the stripped working-tree diff at t, else none. Caps are checked where
  the scorer checks its budget: before each LLM call and when its response arrives. Each crossing
  costs one `git diff` exec on a temporary index, so the agent's `git status` is unchanged.
- A task that ends before t gets its final patch (`ended_before`). Caps at or above
  `max_time_minutes` are not observable.
- **Caveat:** a 15-minute agent at minute 5 is not a 5-minute agent. The prompt and `get_status`
  report its budget, and warnings and nudges fire at different times. Treat `predictions_at_5` as
  a lower bound for a 5-minute agent, not as its score.

### Timing against Kaggle 4×L4

`summary.json` gives two estimates per task: `kaggle_tokens_s` = total tokens (prompt plus
completion, summed over calls) × 900 s / 1M, a community calibration for 4×L4 (unverified here);
and `scaled_agent_s` = local agent seconds × `--timing-scale`, which you calibrate once from the
same submission run locally and on Kaggle. Agent time excludes sandbox setup, like the scorer's
budget clock; Kaggle's 12-hour limit includes setup.

## 5. Output formats

`runs/<name>/` holds `run.json`, `submission/` (frozen copy; resume refuses a different one),
`predictions.jsonl`, `summary.json`, `summary.md`, and per task `tasks/<id>/` with `result.json`,
`llm_calls.jsonl`, `compactions.jsonl`, `patch.diff`, `trace_legacy.json`,
`traces/trace_<id>.json` (the harness's ATIF v1.7 trace), logs and, with `--checkpoints`,
`checkpoints.jsonl`. Field lists: [formats.md](formats.md).

`export-sft` turns `llm_calls.jsonl` into chat samples `{messages, tools, meta}`. A call whose
request plus response is a prefix of the next call is dropped, so you get one sample per maximal
conversation segment; segments break at compactions. `--grades/--only-resolved` labels or filters
by outcome. Grades are matched by instance id, not by patch, so use one grades file per run.

## 6. Deviations from the scorer (printed per run and stored in `run.json`)

1. Serving hardware and profile flags (section 3). The flags change throughput and timing, not
   outputs.
2. The sandbox image is built locally from `Dockerfile.public` plus our stage `final`
   ([grader.md](grader.md)). The scorer uses a Kaggle-built image (runtime possibly gVisor).
3. Environment repairs apply in the agent sandbox by default, as in the grader (`--no-env-fixes`).
4. The model registry points at an external endpoint instead of a server spawned in-process.
   Requests are the same: LiteLlm `openai/<id>` with no `num_retries`, the reasoning patch
   installed, `litellm.drop_params`.
5. Snapshot hydration from a `secret/` bundle is not done (public tasks carry their patches).
6. The scorer's real concurrency, any caps on `eval_config.yaml` budgets, and its private
   `setup.py` are unknown.

## 7. Findings from building the runner (source-verified, checked with the mock)

- **B1: the compaction summarizer never sees tool calls or results.** ADK token-threshold
  compaction runs before each LLM call once the previous prompt reached 14,336 tokens and
  summarises everything except the last 5 events. In the pinned google-adk 1.36.1 the serializer
  passes only `part.text` to the summarizer, so `function_call` and `function_response` parts
  (commands, file contents, test output, edit results) never reach it. The agent.yaml
  `instruction` is not part of the compacted history and is sent on every call.
- **B2: the issue text does reach the summarizer.** It is a user text part. Whether its content
  survives depends on the summary the model writes; P-E5 measures that. The mock e2e test
  (`test_compaction_is_triggered_and_recorded`) uses a fixed summary, so it checks the plumbing
  (the original prompt is replaced by the summary event), not what a real summary retains.
- **The official trace never records these compactions**: the compaction event is appended
  straight to the session service, so `traces/*.json` shows 0. Use `compactions.jsonl`, or
  `llm_calls_summarizer` in `result.json`.
- **The tool-call budget is not a hard stop.** After it is exhausted, tools return
  `BudgetExceeded`, but the model keeps getting calls, and `submit_patch`/`get_status` still work.
  The sample submission's 10-call budget still ended with a submitted patch.
- An `edit_file` call with a missing argument returns ADK's `{"error": "...mandatory input
  parameters are not present..."}`, not the swegemma error shape. The runner counts it as
  `MissingArguments`.

Details and source references: [harness-contracts.md](harness-contracts.md).

## 8. Tests (no GPU)

```bash
uv run --project harness/runner python -m pytest harness/runner/tests -q
```

- `test_units.py` (seconds): parsing all 129 gold patches, NB1 budget semantics, submission
  validation and zips, subsets, termination and error classification, SFT de-duplication, grade
  loading, the health check against the mock, and the serve argv against the scorer flags.
- `test_status.py`: the `--status-dir` writer (atomic writes, no URL/path leaks, unwritable or
  hanging directories).
- `test_checkpoints.py`: checkpoint rules and export, plus an e2e run where the mock edits,
  submits, then writes a file after the submission (the submission still wins).
- `test_e2e_mock.py` (about 2.5 min) runs real Docker sandboxes against
  `harness/runner/mock_openai.py`, which returns Gemma-style parsed tool calls and
  `reasoning_content`. In oracle mode the mock turns the gold patch into `edit_file`/`write_file`
  calls, after a parallel `run_command` + `read_file`, a malformed `edit_file` and a non-matching
  one. It covers two tasks at concurrency 2 with resume, report and SFT export; LoRA routing and
  a 10-call tool budget; refusal when a LoRA is missing; the empty agent (3 nudges, then
  `NO_PATCH`); unparsed tool-call text with `finish_reason=length`; compaction via inflated usage;
  a hard worker timeout with container cleanup; a time-budget override; and the environment
  repairs in the agent sandbox, on and off. The last one needs the fastapi_9425 snapshot (214 MB)
  and `swebench-sandbox:latest`, and is skipped otherwise.
- The tests use `swebench-sandbox:latest` if present, else a throwaway image built from the
  competition's `docker/Dockerfile.sandbox` (no wheels):
  `docker build -f data/kaggle/docker/Dockerfile.sandbox -t gemma4-runner-test:latest <ctx with imp.py telnetlib.py>`.
- Full-chain check (2026-11-02, amd64 emulation): the oracle runner on httpx_3672, requests_6592
  and rich_3006, graded by `harness.grader` at 2/3. requests_6592 needs the network, and its gold
  patch fails the same way.
- Mock by hand: `python -m harness.runner.mock_openai --port 8011 --mode oracle|empty|rawtool|probe [--lora main_lora] [--delay 2] [--inflate-prompt-tokens 20000]`.

## 9. Open risks

- Throughput on one GPU is not Kaggle's. Calibrate `--timing-scale` from one real Kaggle run.
- That `--limit-mm-per-prompt` frees the vision-encoder budget is inferred, not read from source;
  check the vLLM load log.
- Interface changes in `harness/common` or `harness/sandbox` need a runner re-test.
