#!/usr/bin/env bash
# Launch vLLM with the scorer's flags + a single-3090 profile (see harness/serve/serve.py, docs/runner.md §3).
#   harness/serve/serve.sh <model-dir> [--profile 3090|scorer|small|a100|4xl4] [--submission <dir|zip>] [--print] [--log FILE]
# Needs: runner venv (uv sync --project harness/runner) and VLLM_PYTHON=<vLLM venv python> (default ~/venvs/vllm/bin/python).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ $# -ge 1 ]] || { sed -n '2,4p' "$0"; exit 1; }
MODEL_DIR="$1"; shift
export VLLM_PYTHON="${VLLM_PYTHON:-$HOME/venvs/vllm/bin/python}"
cd "$REPO"
exec uv run --project "$REPO/harness/runner" python -m harness.serve.serve --model-dir "$MODEL_DIR" "$@"
