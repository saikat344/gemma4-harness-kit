"""Task and prediction loading helpers (stdlib only; pandas/pyarrow only for .parquet).

Prediction formats accepted by :func:`load_predictions`:
- ``.jsonl``: one object per line with an id key (``instance_id`` | ``id``) and a patch key
  (``model_patch`` | ``prediction`` | ``patch``).
- ``.json``: either ``{instance_id: patch}`` or a list of objects as in ``.jsonl``.
- ``.parquet`` / ``.csv``: columns ``id`` (or ``instance_id``) and ``prediction`` (or ``patch``/``model_patch``),
  the Kaggle ``submission.parquet`` shape.
The string ``NO_PATCH`` (and None/NaN) means an empty patch.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

NO_PATCH = 'NO_PATCH'
_ID_KEYS = ('instance_id', 'id')
_PATCH_KEYS = ('model_patch', 'prediction', 'patch')


def load_tasks(path: Path) -> list[dict[str, Any]]:
    """Reads tasks.jsonl into a list of dicts (file order)."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def repo_short(task: dict[str, Any]) -> str:
    return str(task.get('repo', '')).rsplit('/', 1)[-1]


def normalize_patch(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float):  # NaN from pandas
        return ''
    text = str(value)
    if text.strip() in ('', NO_PATCH):
        return ''
    return text


def _pick(row: dict[str, Any], keys: tuple[str, ...], what: str, src: Path) -> Any:
    for k in keys:
        if k in row:
            return row[k]
    raise ValueError(f'{src}: row has no {what} column (expected one of {keys}): keys={sorted(row)}')


def _rows_to_map(rows: list[dict[str, Any]], src: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        iid = str(_pick(row, _ID_KEYS, 'id', src))
        out[iid] = normalize_patch(_pick(row, _PATCH_KEYS, 'patch', src))
    return out


def load_predictions(path: Path) -> dict[str, str]:
    """Returns {instance_id: patch_text} ('' for empty / NO_PATCH)."""
    suffix = path.suffix.lower()
    if suffix == '.jsonl':
        rows = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
        return _rows_to_map(rows, path)
    if suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            return {str(k): normalize_patch(v) for k, v in data.items()}
        return _rows_to_map(list(data), path)
    if suffix == '.csv':
        with open(path, encoding='utf-8', newline='') as f:
            return _rows_to_map(list(csv.DictReader(f)), path)
    if suffix == '.parquet':
        import pandas as pd  # only needed for parquet

        df = pd.read_parquet(path)
        return _rows_to_map(df.to_dict(orient='records'), path)
    raise ValueError(f'Unsupported predictions format: {path} (use .jsonl/.json/.csv/.parquet)')
