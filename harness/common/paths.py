"""Configurable filesystem locations shared by the harness tools.

Every location can be overridden with an environment variable, so the same code runs on
a laptop (repo-local ``data/``) and on a GPU box with the bulk data on another disk:

- ``GEMMA4_DATA_DIR``      competition data root holding tasks.jsonl, snapshots/, wheels/, sandbox/setup.py
                           (default ``<repo>/data/kaggle``)
- ``GEMMA4_SNAPSHOTS_DIR`` snapshot ``.tgz`` directory (default ``$GEMMA4_DATA_DIR/snapshots``)
- ``GEMMA4_WHEELS_DIR``    competition ``wheels/`` directory (default ``$GEMMA4_DATA_DIR/wheels``)
- ``GEMMA4_RUNS_DIR``      run outputs (default ``<repo>/runs``)
- ``GEMMA4_CACHE_DIR``     host-side cache, e.g. swegemma's unpacked-wheel tars (default ``~/.cache/gemma4``)
- ``GEMMA4_SANDBOX_IMAGE`` sandbox image tag (default ``swebench-sandbox:latest``, the name swegemma expects)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = 'swebench-sandbox:latest'


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, '').strip()
    return Path(raw).expanduser().resolve() if raw else default


def data_dir() -> Path:
    return _env_path('GEMMA4_DATA_DIR', REPO_ROOT / 'data' / 'kaggle')


def tasks_path() -> Path:
    return data_dir() / 'tasks.jsonl'


def snapshots_dir() -> Path:
    return _env_path('GEMMA4_SNAPSHOTS_DIR', data_dir() / 'snapshots')


def wheels_dir() -> Path:
    return _env_path('GEMMA4_WHEELS_DIR', data_dir() / 'wheels')


def sandbox_setup_script() -> Path:
    return data_dir() / 'sandbox' / 'setup.py'


def runs_dir() -> Path:
    return _env_path('GEMMA4_RUNS_DIR', REPO_ROOT / 'runs')


def cache_dir() -> Path:
    return _env_path('GEMMA4_CACHE_DIR', Path.home() / '.cache' / 'gemma4')


def sandbox_image() -> str:
    return os.environ.get('GEMMA4_SANDBOX_IMAGE', '').strip() or DEFAULT_IMAGE


def snapshot_path(instance_id: str, snapshots: Path | None = None) -> Path:
    return (snapshots or snapshots_dir()) / f'{instance_id}.tgz'
