#!/usr/bin/env bash
# Create the grader's host venv (Python 3.12) with the official swegemma wheels.
#   GEMMA4_GRADER_VENV   venv path            (default: harness/grader/.venv)
#   GEMMA4_WHEELHOUSE    dir with the 3 wheels (default: data/wheelhouse)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${GEMMA4_GRADER_VENV:-$REPO/harness/grader/.venv}"
WH="${GEMMA4_WHEELHOUSE:-$REPO/data/wheelhouse}"

for w in swegemma-0.2.11 adk_submission-0.2.13 adk_eval_core-0.1.0; do
  ls "$WH"/$w-*.whl >/dev/null 2>&1 || { echo "missing $WH/$w-*.whl (kaggle datasets download metric/gemma-4-developer-agent-wheelhouse -f <file> -p $WH)"; exit 1; }
done

uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" --no-deps \
  "$WH"/swegemma-0.2.11-*.whl "$WH"/adk_submission-0.2.13-*.whl "$WH"/adk_eval_core-0.1.0-*.whl
uv pip install --python "$VENV/bin/python" -r "$REPO/harness/grader/requirements.txt"
"$VENV/bin/python" -c "from swegemma.harness.verification import verify_task; print('grader venv OK:', '$VENV')"
