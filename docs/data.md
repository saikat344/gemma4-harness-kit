# Data acquisition

The kit ships no competition data, harness wheels or model weights. Fetch them yourself with the
commands below. File counts and sizes are from the competition listing (524 files) and the
wheelhouse listing (41 files), checked 2026-11-03.

## Prerequisites

- A Kaggle account that has **accepted the rules** of the
  [Gemma 4 Developer Agent competition](https://www.kaggle.com/competitions/gemma-4-developer-agent).
  Downloads fail until you do.
- The Kaggle CLI with an API token in `~/.kaggle/` (`kaggle.json`, mode 600). Python 3.12 works;
  `uv tool install --python 3.12 kaggle --with six` installs it in isolation.
- Bash for the snippets below.

Quirk: `kaggle competitions download <comp> -f a/b/c -p DIR` writes `DIR/a/b/c`, so pass the data
root as `-p`.

## Layout

Everything defaults to `data/kaggle/` under the repo root. Set `GEMMA4_DATA_DIR` to put it
elsewhere (`GEMMA4_SNAPSHOTS_DIR` and `GEMMA4_WHEELS_DIR` override single parts).

```
data/kaggle/                      ($GEMMA4_DATA_DIR)
  tasks.jsonl                     129 rows, 1,984,455 B
  sandbox/setup.py                13,883 B
  docker/Dockerfile.public        988 B, sha256 c92dfd9cd9ce019c4910a5334d6ecac606b612ec736a933b4af36da9c6fdf06c
  docker/Dockerfile.sandbox       773 B   (only for the runner tests' throwaway image)
  docker/imp.py                   834 B
  docker/telnetlib.py             23,336 B
  wheels/                         124 .whl, 27,810,465 B (27.8 MB)
  snapshots/                      129 .tgz, 21,496,844,379 B (21.5 GB) for all tasks
  sample_submission/              10 files, 0.44 MB (optional: the official example agent)
  graphs/, embeddings/            127 + 127 files, 0.89 GB (optional: graph tools only)
data/wheelhouse/
  swegemma-0.2.11-py3-none-any.whl          119,670 B
  adk_submission-0.2.13-py3-none-any.whl    66,528 B
  adk_eval_core-0.1.0-py3-none-any.whl      90,060 B
  (the other 38 wheels, incl. vllm-0.19.1, only for the vLLM serving venv; 41 files, 880,842,601 B in all)
```

What reads what:
- `harness/sandbox/build.sh` reads `docker/Dockerfile.public` (and checks its sha256),
  `docker/imp.py`, `docker/telnetlib.py` and `wheels/` from your download
  ([grader.md](grader.md)).
- The grader and runner read `tasks.jsonl`, `sandbox/setup.py`, `wheels/` and `snapshots/`.
- `harness/grader/setup_venv.sh` and `uv sync --project harness/runner` install the three
  Kaggle-authored wheels from `data/wheelhouse/` (`GEMMA4_WHEELHOUSE` moves it for the grader).

## Competition files

Small files (about 2 MB):

```bash
C=gemma-4-developer-agent
D=${GEMMA4_DATA_DIR:-data/kaggle}
for f in tasks.jsonl sandbox/setup.py docker/Dockerfile.public docker/Dockerfile.sandbox docker/imp.py docker/telnetlib.py; do
  kaggle competitions download "$C" -f "$f" -p "$D" -q
done
shasum -a 256 "$D/docker/Dockerfile.public"     # expect c92dfd9c...6fdf06c (sha256sum on Linux)
```

If a file arrives as `<name>.zip`, unzip it in place (inferred; not observed for these files).

Wheels (124 files): list the competition files, then fetch each wheel.

```bash
C=gemma-4-developer-agent; D=${GEMMA4_DATA_DIR:-data/kaggle}; tok=""; : > files.csv
while :; do
  out=$(kaggle competitions files "$C" --csv --page-size 200 ${tok:+--page-token "$tok"})
  echo "$out" | grep -v -e '^Next Page Token' -e '^name,size' >> files.csv
  tok=$(echo "$out" | sed -n 's/^Next Page Token = //p')
  [ -n "$tok" ] || break
done
wc -l < files.csv                                  # 524
grep '^wheels/' files.csv | cut -d, -f1 | xargs -P 4 -I{} kaggle competitions download "$C" -f {} -p "$D" -q
ls "$D"/wheels/*.whl | wc -l                       # 124
```

Snapshots: `harness/grader/fetch_snapshots.sh` downloads one file per API call, 4 in parallel,
and stops before free space on the data disk drops below `GEMMA4_MIN_FREE_GB` (default 8).

```bash
harness/grader/fetch_snapshots.sh              # the 20-task validation subset (2.2 GB)
harness/grader/fetch_snapshots.sh all          # all 129 (21.5 GB)
harness/grader/fetch_snapshots.sh my_ids.txt   # any file of instance ids, one per line
```

Snapshot counts by repository: fastapi 67, rich 48, requests 13, httpx 1. The smallest is
`httpx_3672` (1.5 MB); `requests_6592` is 36 MB; `fastapi_9425` is 214 MB. The runner's mock tests
need `httpx_3672` and `requests_6592` (plus `requests_6629` for the checkpoint e2e test and
`fastapi_9425` for the environment-repair test, which is skipped without it).

Optional: `sample_submission/` (the official example agent, used in the runner examples) with the
same `-f` pattern, e.g. `grep '^sample_submission/' files.csv`.

Alternatively, `kaggle competitions download gemma-4-developer-agent -p "$D"` fetches everything
as one archive (about 22.4 GB of files, snapshots, graphs and embeddings included) to unzip in place.

## Harness wheels

The three Kaggle-authored wheels are not on PyPI. They are in the dataset
`metric/gemma-4-developer-agent-wheelhouse`.

```bash
mkdir -p data/wheelhouse
for w in swegemma-0.2.11-py3-none-any.whl adk_submission-0.2.13-py3-none-any.whl adk_eval_core-0.1.0-py3-none-any.whl; do
  kaggle datasets download metric/gemma-4-developer-agent-wheelhouse -f "$w" -p data/wheelhouse
done
# the whole wheelhouse (881 MB), needed only for the vLLM serving venv (runner.md section 2):
kaggle datasets download metric/gemma-4-developer-agent-wheelhouse -p data/wheelhouse --unzip
```

Pinned versions in the wheelhouse that matter here: google-adk 1.36.1, google-genai 2.11.0,
vllm 0.19.1, transformers 5.13.1, tokenizers 0.22.2, compressed-tensors 0.15.0.1.

## Model weights

Needed only for real agent runs and for the parser suite's tokenizer. The scorer loads
`/kaggle/input/models/google/gemma-4/other/<alias>/2`.

| Alias | Kaggle (`google/gemma-4/other/<alias>/2`) | Hugging Face (ungated) |
|---|---|---|
| gemma-4-31b-it-qat-w4a16-ct | 23,297,590,856 B (one `model.safetensors` of 23,265,352,448 B) | `google/gemma-4-31B-it-qat-w4a16-ct`, revision `ca9d5250d08695be35f2d632b434938b6d4aecfe` (same safetensors size) |
| gemma-4-12b-it-qat-w4a16-ct | 10.3 GB | `google/gemma-4-12B-it-qat-w4a16-ct` |
| gemma-4-e4b-it-qat-w4a16-ct | 11.5 GB | `google/gemma-4-E4B-it-qat-w4a16-ct` |
| gemma-4-e2b-it-qat-w4a16-ct | 8.3 GB | `google/gemma-4-E2B-it-qat-w4a16-ct` |

```bash
M=<weights dir>/google/gemma-4/other
kaggle models instances versions download google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2 \
  -p $M/gemma-4-31b-it-qat-w4a16-ct/2 --untar
# or, matching the Kaggle files by size:
hf download google/gemma-4-31B-it-qat-w4a16-ct --revision ca9d5250d08695be35f2d632b434938b6d4aecfe \
  --local-dir $M/gemma-4-31b-it-qat-w4a16-ct/2
```

The chat template depends on the revision. The Kaggle files list `chat_template.jinja` at 16,934 B and
`tokenizer_config.json` at 2,741 B, the sizes in release commit `ca9d5250…`. The agent runs and the P-E5 replay in the
paper used HF main at revision `52f3f65bc7a02d555763bc923bd1d9094898219d` (18,683 B, same weights), found to differ only
afterwards. Use `--revision 52f3f65…` to reproduce those runs. The effect on solves was not measured pairwise.

Keep the `.../<alias>/2` directory name: `serve.py` infers the alias from the path (or pass
`--alias`). The Kaggle CLI cannot fetch single files of a model; use Hugging Face to get only
`config.json` and the tokenizer files. Weights are under the model's own licence terms on Kaggle
and Hugging Face.

## Community soundness data

An independent per-task soundness audit, with dependency repairs, is published by Dmitrii Gluzdov
under CC BY 4.0:
[Gemma 4: Task Soundness, Serving and Evaluation](https://www.kaggle.com/code/dmitriigluzdov/gemma-4-task-soundness-serving-and-evaluation).
The kit links to it and does not copy it. Some of our per-task pins come from it (credited in
[grader.md](grader.md) and in the release NOTICE). It excludes 15 tasks; 12 overlap with our 17
unsound tasks (`harness/subsets/unsound.txt`), and the differences track environment choices.
