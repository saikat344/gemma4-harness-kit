# Grader

`python -m harness.grader` grades predictions (instance id -> unified diff) with the official
`swegemma.harness.verification.verify_task` (swegemma 0.2.11, installed from the competition
wheelhouse) in a Docker sandbox whose first stage is the competition's `docker/Dockerfile.public`.
Grading logic is inherited, not reimplemented: patch stripping, patch application, `test_patch`,
the generated `pytest.ini`, the pytest command, and the JUnit and required-test checks all come
from the wheel (see [harness-contracts.md](harness-contracts.md)). Our additions are sandbox
packages and per-task environment repairs. One flag turns all of them off.

Status (2026-11-02): validated on a 20-task subset under amd64 emulation, and on all 129 public
tasks natively on x86_64 (sections "Validation" and "Full 129-task validation").

## Files

| Path | What |
|---|---|
| `harness/sandbox/compose_dockerfile.py` | Composes the build Dockerfile from your downloaded `Dockerfile.public` (see "Sandbox image"). |
| `harness/sandbox/Dockerfile.fixes` | Stage `final`: extra test packages and override wheels. |
| `harness/sandbox/build.sh` | Builds `swebench-sandbox:latest` (linux/amd64). `--official` builds only stage `official` as `swebench-sandbox:official`. |
| `harness/sandbox/task_env_fixes.json` | Shell commands run after every `sandbox/setup.py` call, by repo (`*` = all) or by task, each with its reason. |
| `harness/sandbox/gemma4_fixes/*.py` | In-container helpers installed in `/opt/gemma4/bin`: `unshadow_workspace.py`, `resolve_from_wheels.py`. |
| `harness/common/envfix.py` | `FixingContainerManager`: the official `ContainerManager` plus the fixes. The runner uses it too, so the agent sandbox matches grading. |
| `harness/common/paths.py`, `tasks.py` | Environment-configurable paths; task and prediction loaders. |
| `harness/grader/cli.py` | The CLI. |
| `harness/grader/setup_venv.sh`, `requirements.txt` | Host venv (Python 3.12, uv). |
| `harness/grader/fetch_snapshots.sh`, `validation_subset.txt` | Snapshot download with a free-space guard. |

## Setup (one-time, from the repo root)

Fetch the competition data first ([data.md](data.md)).

```bash
harness/grader/setup_venv.sh          # -> harness/grader/.venv (GEMMA4_GRADER_VENV to move it)
harness/sandbox/build.sh              # -> swebench-sandbox:latest (about 40 s warm on a Mac, 61 s native x86_64)
harness/grader/fetch_snapshots.sh     # 20-task validation subset (2.2 GB); "all" = 129 snapshots (21.5 GB)
```

Prerequisites:
- `$GEMMA4_DATA_DIR` (default `data/kaggle`) holds `tasks.jsonl`, `sandbox/setup.py`,
  `docker/{Dockerfile.public,imp.py,telnetlib.py}` and the competition `wheels/` (124 files, 27.8 MB).
- `data/wheelhouse/` holds the three Kaggle wheels (`GEMMA4_WHEELHOUSE` to move it).

Environment variables: `GEMMA4_DATA_DIR`, `GEMMA4_SNAPSHOTS_DIR`, `GEMMA4_WHEELS_DIR`,
`GEMMA4_RUNS_DIR`, `GEMMA4_CACHE_DIR` (default `~/.cache/gemma4`), `GEMMA4_SANDBOX_IMAGE`, and
`GEMMA4_MIN_FREE_GB` (fetch guard, default 8).

### Sandbox image

`Dockerfile.public` is not redistributed. `build.sh` reads your download at
`$GEMMA4_DATA_DIR/docker/Dockerfile.public`, checks its sha256
(`c92dfd9cd9ce019c4910a5334d6ecac606b612ec736a933b4af36da9c6fdf06c`), and composes:

- stage `official`: the public file verbatim, except that the base image is pinned to a digest
  and the unpinned pip packages are pinned to the versions that resolved on 2026-11-02
  (pytest 9.2.0 among them). Pins are the only changed lines.
- stage `final`: `Dockerfile.fixes` appended (test packages and override wheels, below).

The build context also takes `imp.py` and `telnetlib.py` (the shims the public Dockerfile copies)
and `wheels/` from your download. To check that only pins changed:

```bash
python harness/sandbox/compose_dockerfile.py --public data/kaggle/docker/Dockerfile.public --diff
```

The image must be linux/amd64: fastapi tasks need the x86-only `pydantic_core` wheel.

## Grading

```bash
PY=harness/grader/.venv/bin/python
$PY -m harness.grader --predictions runs/<name>/predictions.jsonl --out runs/<name>/grades.jsonl --workers 3
$PY -m harness.grader --predictions submission.parquet --out runs/<name>/grades.jsonl   # columns id, prediction
$PY -m harness.grader --gold  --out runs/<name>/gold/grades.jsonl    # soundness: each gold patch should pass
$PY -m harness.grader --empty --out runs/<name>/empty/grades.jsonl   # soundness: tests run on an empty patch, which should fail
```

- **Prediction formats:** `.jsonl` (`instance_id|id` plus `model_patch|prediction|patch`);
  `.json` (`{id: patch}` or a list of objects); `.csv` and `.parquet` (columns `id`, `prediction`).
  `NO_PATCH`, null and empty all mean an empty patch.
- **Empty patch:** in predictions mode it is graded unresolved without running tests, as the
  scorer does (no patch plus an agent error skips Phase 2). `--run-empty-patches` runs the tests anyway.
- **Filtering:** `--task-ids`, `--repos fastapi rich`. Tasks without a local snapshot are skipped
  and listed in `summary.json`; `--include-missing-snapshots` counts them as failures.
- **Other options:** `--resume` (skip ids already in `--out`), `--no-env-fixes` (official
  environment), `--image`, `--network` (default `none`, as the scorer), `--verification-timeout`
  (default 300 s), `--task-timeout` (grader-side cap, default 3600 s).
- **Outputs next to `--out`:** `grades.jsonl`, `summary.json`, `logs/<id>.log` (inputs, env-fix
  output, full pytest stdout and stderr), `grader.log`, and a final stdout line
  `RESOLVED r/n (x%) ...`. Field lists: [formats.md](formats.md).
- **Workers:** each worker runs one 4 GiB / 2 vCPU container. With an 8 GB Docker VM use
  `--workers 3` or fewer.

To see the public harness exactly as shipped, for any run:
`build.sh --official` once, then grade with `--no-env-fixes --image swebench-sandbox:official`.

## Validation: 20-task subset (amd64 emulation on an M3 Mac, 3 workers, 2026-11-02)

`gold`/`empty` use the repaired image with fixes on; `*-official` use `build.sh --official` with
`--no-env-fixes`, i.e. the public harness as shipped.

| Run | Resolved | Wall time |
|---|---|---|
| gold | **16/20** | 202 s |
| empty | **1/20** | 189 s |
| gold-official | 6/20 | 182 s |
| empty-official | 0/20 | 178 s |

| Task | gold | empty | gold (official env) |
|---|---|---|---|
| fastapi_9425, 11194, 12942, 13713, 14099 | PASS | fail | fail |
| fastapi_14077 | PASS | **PASS** | fail |
| httpx_3672 | PASS | fail | fail |
| requests_6589, 6592, 6629, 6757 | fail | fail | fail |
| requests_6644 | PASS | fail | fail |
| rich_2725, 3006, 3043, 3061, 3063, 3067 | PASS | fail | PASS |
| rich_3064, rich_3130 | PASS | fail | fail |

Under emulation a task takes 16–60 s including container setup and wheel injection.

## Full 129-task validation (native x86_64, Docker on WSL2, 6 workers, 2026-11-02)

Image built natively (x86_64, Python 3.13.18, pytest 9.2.0, 124 wheels), fixes on, `--network none`.

| Run | fastapi | httpx | requests | rich | Total | Wall time | s/task per container |
|---|---|---|---|---|---|---|---|
| gold | 62/67 | 1/1 | 5/13 | 46/48 | **114/129** | 587 s | median 25.3, mean 26.7, max 61.6 |
| empty | 2/67 | 0/1 | 0/13 | 0/48 | **2/129** | 867 s | median 25.9, mean 33.6, max 321.3 |

Wall times are `wall_seconds` from each run's `summary.json`. Most container time is setup and
wheel injection; pytest itself takes 0.3–40 s.

**Unsound tasks (17)** are listed in `harness/subsets/unsound.txt`, one reason per line. Gold
failures (15) by cause:

| Cause | Tasks | Notes |
|---|---|---|
| Network (`--network none`) | requests_6589, 6592, 6629, 6757, 7328, 7433, 7502, 7505 (8) | `TestTimeout::test_connect_timeout` and `test_total_timeout_connect` get `ENETUNREACH`, not `ConnectTimeout`. requests_6757 also fails 2 mTLS tests (certificates expired or rejected by Python 3.13 strict X.509). |
| Forced `-p no:anyio` | fastapi_15745, 15763, 15785 (3) | The same 3 async tests in `test_router_include_context.py` fail: "async def functions are not natively supported". |
| Python 3.13 | rich_3472, rich_3486 (2) | rich_3472 expects the pre-3.13 AttributeError text; rich_3486's required test is `skipif(sys.version_info.minor >= 11)`, so it is skipped and counts as not passed. |
| Our image (httpbin pulls in brotlicffi) | fastapi_14356 (1) | httpx then advertises `br` in Accept-Encoding. Fixable by uninstalling `brotlicffi` for non-requests tasks. |
| Unresolved dependency skip | fastapi_14186 (1) | A required test is skipped; the skip reason is not in the log (rerun with `-rs`). |

Empty passes (2): fastapi_14077 and fastapi_14851 (required checks already pass at the base commit).

Other observations:
- 3 empty-patch runs hit the 300 s verification timeout: rich_3480, rich_3772, rich_4006. Their
  gold runs take 16–24 s, so without the fix the suite hangs. A failing agent patch on these
  tasks costs about 5 minutes of grading.
- The network, anyio and Python 3.13 classes come from harness flags or the interpreter, so they
  plausibly affect the scorer too (inferred; its image is private).
- An independent community audit
  ([Gluzdov, CC BY 4.0](https://www.kaggle.com/code/dmitriigluzdov/gemma-4-task-soundness-serving-and-evaluation))
  excludes 15 tasks; 12 overlap with our 17. Its environment differs (Python 3.12 there, 3.13
  here; empty-control timeouts; network error modes). A soundness list is only valid together
  with its environment.

## Divergences from the scorer (and why)

The scorer's Stage 1/2 scripts, its private data bundle (with its own `setup.py`) and its
Kaggle-built `swebench-sandbox` image are not public. The public harness as shipped passes only
6/20 gold patches, and the failures are systematic, so (inferred) the scorer environment differs
from the public files. We add these repairs. They run during sandbox setup, before the baseline
commit, so they never enter an agent's diff.

1. **Un-shadow the workspace package** (`*`, `unshadow_workspace.py`). swegemma injects every
   wheel from `wheels/` into site-packages after the editable install, including
   `requests-2.34.2` and `httpx-0.28.1`. For src-layout repos the injected copy wins over
   `/workspace/src` (`.pth` paths come after site-packages), so gold and empty patches test the
   installed library, not the repository (verified: `requests.__file__` resolved to
   site-packages). We delete site-packages copies, and their non-editable dist-info, of packages
   that also exist at the top of `/workspace` or `/workspace/src`. fastapi and rich use a root
   layout and are unaffected.
2. **Per-task version resolution from `wheels/`** (`*`, `resolve_from_wheels.py`). The harness
   injects only the newest version of each package, e.g. starlette 1.6.0. fastapi snapshots
   require `starlette<0.49`, so every fastapi gold patch fails to collect. `wheels/` holds 50
   starlette versions, 5 sqlmodel, 3 typer, 2 flask/werkzeug, and pytest 6.2.5/8.3.4, matching
   pins in the snapshots' requirement files; (inferred) the scorer selects them per task. The
   resolver intersects the editable install's `Requires-Dist` (no extras) with the root
   `requirements*.txt` files (following `-r`), then installs the highest satisfying py3.13 wheel
   from `/wheels`, transitively. It never touches pytest, pluggy or pytest plugins; the image's
   pytest is used, as in the scorer. Observed swaps: starlette 1.6.0 -> 0.48.0 (0.47.3 on
   fastapi_12942), sqlmodel 0.0.39 -> 0.0.24/0.0.25, typer -> 0.16.0 on fastapi; click -> 8.2.1 on httpx.
3. **Extra test packages in the image** (pinned; the full set is in `/opt/gemma4-freeze.txt`):
   - `typing-inspection==0.4.4`: needed by pydantic 2.13.4 from `wheels/`.
   - `inline-snapshot==0.36.1`, `dirty-equals==0.9.0`, `python-multipart==0.0.33`: fastapi tests.
   - `pytest-httpbin==2.1.0`, `httpbin==0.10.4`, `trustme==1.2.1`: requests tests (2.1.0's
     certificates pass Python 3.13 strict X.509; httpbin 0.10.0 cannot import under Werkzeug 3).
   - `attrs==21.4.0`: without the real package, rich's `examples/attrs.py` is imported instead
     and calls `sys.exit` (pytest INTERNALERROR).
   - Side effects: the httpbin pytest plugin loads in every task; httpbin pulls in `brotlicffi`,
     which breaks fastapi_14356 (above).
4. **Per-repo and per-task pins** (override wheels in `/opt/gemma4/wheels`):

   | Scope | Fix | Evidence |
   |---|---|---|
   | requests (all) | werkzeug 2.3.8 + flask 2.3.3 (both in `wheels/`) | `test_redirecting_to_bad_url[...InvalidURL]` fails under Werkzeug 3; fixed on 6589/6592/6629 |
   | rich_3064, rich_3130 | Pygments 2.15.1 | gold fails with 2.21.0; passes with the pin |
   | rich_3296, 3469, 3782, 4070, 4079 | Pygments 2.16.1 / 2.17.2 / 2.19.1 / 2.19.2 | community repairs (Gluzdov), not verified here |
   | fastapi_14962, fastapi_14964 | ujson 5.11.0 + orjson 3.11.3 | community repairs (Gluzdov), not verified here |

5. **Grader-only behaviour:** the `--gold`/`--empty` modes; a grader-side per-task timeout;
   swegemma's unpacked-wheel tar cache moves to `$GEMMA4_CACHE_DIR/swegemma-tmp-<hash of the
   wheels listing>` (upstream keys it only by name, so a changed `wheels/` would reuse a stale tar).
6. **Not reproduced:** gVisor (`runsc`) and the Kaggle docker proxy; Phase 1 stripping by
   `submit_patch` (the grader receives the final diff, and `verify_task` strips again with the
   same function); the scorer's pytest version and Python patch level (ours: 9.2.0, 3.13.18).

### Open network question

With `network_mode='none'` (the scorer's `ContainerConfig`), connecting to the tarpit address in
requests' `TestTimeout` fails at once with `ENETUNREACH`. A Docker `--internal` network does not
help (no default route; tested). On Linux, a bridge network whose egress is dropped by iptables
(`DOCKER-USER`) should produce real timeouts; pass it with `--network <name>`. Untested.
