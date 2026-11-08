"""Shared helpers for the runner: submission loading, eval_config budgets, tasks, subsets, endpoint.

Everything that decides *what* runs mirrors the official getting-started notebook (NB1 cells 4, 8, 10;
docs/harness-contracts.md). Deviations are explicit and recorded in run.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSETS_DIR = REPO_ROOT / 'harness' / 'subsets'
DEFAULT_DATA_DIR = REPO_ROOT / 'data' / 'kaggle'

ALLOWED_MODEL_NAMES = frozenset({
    'gemma-4-31b-it-qat-w4a16-ct',
    'gemma-4-12b-it-qat-w4a16-ct',
    'gemma-4-e4b-it-qat-w4a16-ct',
    'gemma-4-e2b-it-qat-w4a16-ct',
})
_SAFE_PATH_COMPONENT = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')

# NB1 cell 10 defaults when eval_config.yaml omits a key.
EVAL_DEFAULTS = {
    'timeout_seconds': 300,
    'verification_timeout_seconds': 300,
    'max_tool_calls': 100,
    'max_time_minutes': 60.0,
    'max_turns': None,
}

# NB1 cell 10 context management (the CLI passes neither; that's why `swegemma eval` is not scorer-equivalent).
CONTEXT_CACHE = {'min_tokens': 2048, 'ttl_seconds': 1800, 'cache_intervals': 10}
COMPACTION = {
    'compaction_interval': 15,
    'overlap_size': 2,
    'token_threshold': 14336,
    'event_retention_size': 5,
}


# --------------------------------------------------------------------------------------------------
# Submission
# --------------------------------------------------------------------------------------------------


def prepare_submission(src: Path, run_dir: Path) -> Path:
    """Copy a submission dir or extract a submission zip into run_dir/submission (immutable snapshot).

    The copy makes resume safe against edits to the source dir and gives the trajectory a fixed config.
    """
    dest = run_dir / 'submission'
    if dest.exists():
        return dest
    src = src.expanduser().resolve()
    tmp = run_dir / 'submission.tmp'
    if tmp.exists():
        shutil.rmtree(tmp)
    if src.is_file() and src.suffix == '.zip':
        with zipfile.ZipFile(src) as zf:
            for info in zf.infolist():
                p = Path(info.filename)
                if p.is_absolute() or '..' in p.parts:
                    raise ValueError(f'Unsafe path in submission zip: {info.filename}')
            zf.extractall(tmp)
        # Kaggle wants agent.yaml at the archive root; accept one wrapping directory with a warning.
        if not (tmp / 'agent.yaml').is_file():
            subdirs = [d for d in tmp.iterdir() if d.is_dir()]
            if len(subdirs) == 1 and (subdirs[0] / 'agent.yaml').is_file():
                print(f'WARNING: agent.yaml is not at the zip root (found in {subdirs[0].name}/); '
                      'Kaggle requires it at the root.')
                inner = subdirs[0]
                for child in inner.iterdir():
                    child.rename(tmp / child.name)
                inner.rmdir()
    elif src.is_dir():
        shutil.copytree(src, tmp, symlinks=True)
    else:
        raise FileNotFoundError(f'Submission not found or not a dir/zip: {src}')
    if not (tmp / 'agent.yaml').is_file():
        raise FileNotFoundError(f'No agent.yaml at submission root: {src}')
    tmp.rename(dest)
    return dest


def validate_submission(agent_dir: Path) -> dict[str, Any]:
    """NB1 cells 8 and 12: one declared allowed model, safe adapter names, no symlinks, validate_directory."""
    from adk_submission import discover_adapters, validate_directory
    from adk_submission.discovery import discover_declared_models
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, build_submission_limits

    for p in agent_dir.rglob('*'):
        if p.is_symlink():
            raise ValueError(f'Symlinks are not permitted: {p.relative_to(agent_dir)}')
    limits, _ = build_submission_limits()
    validate_directory(agent_dir, limits=limits)

    declared = discover_declared_models(agent_dir)
    if len(declared) != 1:
        raise ValueError(f'Expected exactly 1 declared model, got: {sorted(declared)}')
    model = next(iter(declared))
    if model not in ALLOWED_MODEL_NAMES:
        raise ValueError(f'Unsupported model {model!r}. Allowed: {sorted(ALLOWED_MODEL_NAMES)}')

    adapters = discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    adapter_map = getattr(adapters, 'adapters', None) or {}
    for name in adapter_map:
        if not _SAFE_PATH_COMPONENT.fullmatch(name):
            raise ValueError(f'Unsafe adapter name {name!r}: must match [A-Za-z0-9][A-Za-z0-9._-]*')
    return {
        'declared_model': model,
        'adapters': {
            name: {'path': str(info.path), 'rank': getattr(info, 'rank', None)}
            for name, info in adapter_map.items()
        },
    }


def dir_sha256(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------
# Budgets
# --------------------------------------------------------------------------------------------------


@dataclass
class Budgets:
    timeout_seconds: int
    verification_timeout_seconds: int
    max_tool_calls: int
    max_time_minutes: float
    max_turns: int | None
    source: str = 'eval_config.yaml'
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_budgets(agent_dir: Path, overrides: dict[str, str] | None = None) -> Budgets:
    """Exactly NB1 cell 10: eval_config.yaml|yml, `evaluation:` section or the root dict, same defaults.

    `overrides` (from --override key=value) are a deliberate deviation and are recorded.
    """
    cfg_file = next(
        (c for name in ('eval_config.yaml', 'eval_config.yml') if (c := agent_dir / name).is_file()),
        None,
    )
    raw = (yaml.safe_load(cfg_file.read_text(encoding='utf-8')) or {}) if cfg_file else {}
    ev = raw.get('evaluation', raw)
    turns_raw = ev.get('max_turns', ev.get('max_llm_calls'))
    b = Budgets(
        timeout_seconds=int(ev.get('timeout_seconds', 300)),
        verification_timeout_seconds=int(ev.get('verification_timeout_seconds', 300)),
        max_tool_calls=int(ev.get('max_tool_calls', 100)),
        max_time_minutes=float(ev.get('max_time_minutes', 60.0)),
        max_turns=int(turns_raw) if turns_raw is not None else None,
        source=cfg_file.name if cfg_file else 'defaults (no eval_config.yaml)',
    )
    for key, val in (overrides or {}).items():
        if key not in EVAL_DEFAULTS:
            raise ValueError(f'Unknown budget override {key!r}; allowed: {sorted(EVAL_DEFAULTS)}')
        cast = float if key == 'max_time_minutes' else int
        new = None if val.lower() in ('none', 'null') else cast(val)
        b.overrides[key] = {'from': getattr(b, key), 'to': new}
        setattr(b, key, new)
    return b


# --------------------------------------------------------------------------------------------------
# Tasks and subsets
# --------------------------------------------------------------------------------------------------


def load_task_rows(tasks_path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(tasks_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_task_ids(spec: str | None, all_ids: list[str]) -> list[str]:
    """--task-ids accepts: a subset name (harness/subsets/<name>.txt), a file of ids, or a comma list.

    Order follows the selector (subset file order); unknown ids are an error.
    """
    if not spec:
        return list(all_ids)
    path = Path(spec)
    cand = SUBSETS_DIR / f'{spec}.txt'
    if cand.is_file():
        path = cand
    if path.is_file():
        ids = [ln.split('#', 1)[0].strip() for ln in path.read_text().splitlines()]
        ids = [i for i in ids if i]
    else:
        ids = [s.strip() for s in spec.split(',') if s.strip()]
    known = set(all_ids)
    missing = [i for i in ids if i not in known]
    if missing:
        raise ValueError(f'{len(missing)} task id(s) not in tasks file: {missing[:5]}')
    seen: set[str] = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


# --------------------------------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------------------------------


def http_json(url: str, payload: dict | None = None, timeout: float = 10.0, api_key: str = 'EMPTY') -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method='POST' if data else 'GET')
    req.add_header('Authorization', f'Bearer {api_key}')
    if data:
        req.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def probe_endpoint(endpoint: str, api_key: str = 'EMPTY') -> dict[str, Any]:
    """GET {endpoint}/models. Base model = entry without `parent` (vLLM sets parent on LoRA modules)."""
    try:
        data = http_json(endpoint.rstrip('/') + '/models', api_key=api_key)
    except (urllib.error.URLError, OSError) as e:
        raise ConnectionError(f'Endpoint {endpoint} unreachable: {e}') from e
    models = data.get('data', [])
    base = [m for m in models if not m.get('parent')]
    loras = [m for m in models if m.get('parent')]
    return {
        'models': models,
        'base_ids': [m['id'] for m in base],
        'lora_ids': [m['id'] for m in loras],
        'max_model_len': next((m.get('max_model_len') for m in base if m.get('max_model_len')), None),
    }
