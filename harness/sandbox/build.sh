#!/usr/bin/env bash
# Build the scorer-like sandbox image (linux/amd64) as swebench-sandbox:latest.
# The Dockerfile is composed at build time (compose_dockerfile.py): stage "official" = the competition's own
# docker/Dockerfile.public from your download (sha256-checked) with the base digest and pip versions pinned,
# stage "final" = Dockerfile.fixes. Neither the competition Dockerfile nor its imp.py/telnetlib.py ship with this kit.
#   GEMMA4_DATA_DIR       competition data root with docker/{Dockerfile.public,imp.py,telnetlib.py} and wheels/
#                         (default: <repo>/data/kaggle)
#   GEMMA4_WHEELS_DIR     override for the wheels dir (default: $GEMMA4_DATA_DIR/wheels)
#   GEMMA4_SANDBOX_IMAGE  tag (default: swebench-sandbox:latest)
#   --official            build only stage "official" (Dockerfile.public + pins, no local fixes) as swebench-sandbox:official
# Extra args are passed to docker build (e.g. --no-cache).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DATA="${GEMMA4_DATA_DIR:-$REPO/data/kaggle}"
WHEELS="${GEMMA4_WHEELS_DIR:-$DATA/wheels}"
TAG="${GEMMA4_SANDBOX_IMAGE:-swebench-sandbox:latest}"
TARGET=final
if [[ "${1:-}" == "--official" ]]; then TARGET=official; TAG="${TAG}-official"; TAG="${TAG/:latest-official/:official}"; shift; fi

for f in "$DATA/docker/Dockerfile.public" "$DATA/docker/imp.py" "$DATA/docker/telnetlib.py"; do [[ -f "$f" ]] || { echo "missing $f"; exit 1; }; done
n=$(ls "$WHEELS"/*.whl 2>/dev/null | wc -l | tr -d ' ')
[[ "$n" -ge 100 ]] || { echo "expected the 124 competition wheels in $WHEELS, found $n"; exit 1; }

CTX="$(mktemp -d "${TMPDIR:-/tmp}/gemma4-sandbox-ctx.XXXXXX")"
trap 'rm -rf "$CTX"' EXIT
python3 "$HERE/compose_dockerfile.py" --public "$DATA/docker/Dockerfile.public" --out "$CTX/Dockerfile"
cp "$DATA/docker/imp.py" "$DATA/docker/telnetlib.py" "$CTX/"
cp -r "$HERE/gemma4_fixes" "$CTX/gemma4_fixes"
mkdir "$CTX/wheels" && cp "$WHEELS"/*.whl "$CTX/wheels/"

docker build --platform linux/amd64 --target "$TARGET" -t "$TAG" "$@" "$CTX"
docker run --rm --platform linux/amd64 "$TAG" sh -c 'echo "built: $(uname -m) $(python -V) pytest $(python -m pytest --version 2>&1 | tail -1)"; ls /wheels | wc -l'
