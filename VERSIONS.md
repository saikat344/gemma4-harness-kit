# Pinned versions

Everything the paper's numbers were produced with. The scorer installs the same versions of the harness
packages; anything marked "ours" is a pin we chose where upstream leaves a version open.

## Harness (host side, Python 3.12)

| Package | Version | Where it is pinned |
|---|---|---|
| swegemma | 0.2.11 | Kaggle wheelhouse (not on PyPI); `harness/runner/pyproject.toml`, `harness/grader/setup_venv.sh` |
| adk-submission | 0.2.13 | same |
| adk-eval-core | 0.1.0 | same |
| google-adk | 1.36.1 | `harness/runner/pyproject.toml`, `harness/grader/requirements.txt` |
| google-genai | 2.11.0 | same |
| litellm | 1.83.7 (scorer image range >=1.83.7,<=1.83.14) | `harness/runner/uv.lock` |
| openai | 2.30.0 | `harness/runner/uv.lock` |
| transformers | 5.13.1 | `harness/runner/pyproject.toml` |
| torch | 2.10.0 (CPU build; the runner never needs CUDA) | `harness/runner/pyproject.toml` |
| docker (Python SDK) | 7.2.0 | `harness/runner/uv.lock`, `harness/grader/requirements.txt` |
| pytest | 9.2.0 | `harness/runner/uv.lock` |

The complete resolved set is `harness/runner/uv.lock` (runner, contracts, analysis, repro tests) and
`harness/grader/requirements.txt` (the grader venv's 14 direct pins; pip resolves the rest).

## Serving (separate venv on the GPU host)

| Package | Version | Note |
|---|---|---|
| vllm | 0.19.1 | from the Kaggle wheelhouse, installed `--no-deps` with the wheelhouse pins (docs/runner.md) |
| huggingface-hub | 1.33.0 | ours: PyPI's 2.x breaks transformers 5.13.1 |
| Model | `gemma-4-31b-it-qat-w4a16-ct` | Kaggle `google/gemma-4` (the scorer's files) or HF `google/gemma-4-31B-it-qat-w4a16-ct` at revision `ca9d5250d08695be35f2d632b434938b6d4aecfe` (the release commit: its `chat_template.jinja` and `tokenizer_config.json` have the byte sizes the Kaggle files list; hashes unavailable) |
| Chat template of the paper's runs | HF revision `52f3f65bc7a02d555763bc923bd1d9094898219d` | HF main at the time, which has a different chat template from the release commit (same weights). The agent runs and the P-E5 replay in the paper used it, so serve this revision to reproduce them and `ca9d5250…` to match the scorer. The effect on solves was not measured pairwise. |

Scorer serving flags: `harness/serve/serve.py --model-dir <dir> --profile 4xl4 --print` prints the argv for the
scorer's hardware (four L4, `--tensor-parallel-size 4`); every profile prints its deviations from the scorer.

## Sandbox image (`swebench-sandbox:latest`, linux/amd64)

| Item | Value |
|---|---|
| Base | `python:3.13-slim@sha256:ae14d4ce8771606477a81446e7ef429edd6762305fc50c84507c1041fa8285d1` (ours: the public Dockerfile does not pin it) |
| Python in the image | 3.13.18 |
| Competition `Dockerfile.public` | sha256 `c92dfd9cd9ce019c4910a5334d6ecac606b612ec736a933b4af36da9c6fdf06c` (checked at build time) |
| pip pins added to stage `official` (ours) | pytest 9.2.0, typer 0.27.3, pdm-backend 2.5.0, setuptools 84.2.1, wheel 0.48.0, poetry-core 2.5.0, hatchling 1.33.0, flit-core 4.2.0, editables 0.6 (pytest-timeout 2.1.0 is pinned upstream) |
| Stage `final` additions | `harness/sandbox/Dockerfile.fixes` (test packages and override wheels, with versions) |
| Per-task repairs | `harness/sandbox/task_env_fixes.json` |

All pins are the versions that resolved on 2026-11-02.

## Source

Exported by the authors' `scripts/export_release.py` from source commit `1077b9b9b6cdadafc5959a05cd442205a1e8fee7` (2026-11-08T10:49:38-08:00).
