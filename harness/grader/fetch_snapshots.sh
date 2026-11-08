#!/usr/bin/env bash
# Download task snapshots from the competition (one Kaggle API call per file, 4 in parallel).
#   fetch_snapshots.sh [LIST]   LIST = file of instance ids (default: harness/grader/validation_subset.txt);
#                               "all" = every id in tasks.jsonl (21.5 GB; point GEMMA4_DATA_DIR at a large disk).
#   GEMMA4_DATA_DIR  data root (snapshots land in $GEMMA4_DATA_DIR/snapshots)
#   GEMMA4_MIN_FREE_GB  stop when free space on the data disk would fall below this (default 8)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DATA="${GEMMA4_DATA_DIR:-$REPO/data/kaggle}"
MIN_FREE_GB="${GEMMA4_MIN_FREE_GB:-8}"
LIST="${1:-$HERE/validation_subset.txt}"
mkdir -p "$DATA/snapshots"
if [[ "$LIST" == "all" ]]; then
  ids=$(python3 -c "import json,sys; [print(json.loads(l)['instance_id']) for l in open(sys.argv[1]) if l.strip()]" "$DATA/tasks.jsonl")
else
  ids=$(grep -v '^\s*#' "$LIST" | awk '{print $1}' | grep .)
fi
free_gb() { df -Pk "$DATA/snapshots" | awk 'NR==2{print int($4/1048576)}'; }
one() {
  id="$1"; f="$DATA/snapshots/$id.tgz"
  [[ -s "$f" ]] && return 0
  if (( $(free_gb) < MIN_FREE_GB + 1 )); then echo "SKIP $id: free space below ${MIN_FREE_GB}GB+1"; return 0; fi
  kaggle competitions download gemma-4-developer-agent -f "snapshots/$id.tgz" -p "$DATA" -q >/dev/null 2>&1 || echo "FAIL $id"
}
export -f one free_gb; export DATA MIN_FREE_GB
echo "$ids" | xargs -P 4 -n 1 bash -c 'one "$0"'
echo "snapshots in $DATA/snapshots: $(ls "$DATA/snapshots" | wc -l | tr -d ' '); free: $(free_gb) GB"
