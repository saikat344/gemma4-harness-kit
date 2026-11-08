"""compose_dockerfile.py: pins only the base image and the unpinned pip packages; everything else is copied."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import compose_dockerfile as cd  # noqa: E402

REPO = HERE.parents[1]
DATA = Path(os.environ.get('GEMMA4_DATA_DIR', REPO / 'data' / 'kaggle'))

# synthetic stand-in with the same shape as the competition file (not its text)
FAKE = ('# header\nFROM python:3.13-slim\nRUN apt-get update \\\n'
        '    && pip install --no-cache-dir pytest pytest-timeout==2.1.0 typer wheel \\\n    && true\nWORKDIR /w\n')


def test_pin_official_changes_only_from_and_pip_lines():
    out = cd.pin_official(FAKE)
    changed = [(a, b) for a, b in zip(FAKE.splitlines(), out.splitlines()) if a != b]
    assert len(out.splitlines()) == len(FAKE.splitlines()) and len(changed) == 2
    assert changed[0][1] == f'FROM python:3.13-slim@{cd.BASE_DIGEST} AS official'
    assert 'pytest==9.2.0 pytest-timeout==2.1.0 typer==0.27.3 wheel==0.48.0 \\' in changed[1][1]


def test_unknown_unpinned_package_is_an_error():
    with pytest.raises(cd.ComposeError):
        cd.pin_official(FAKE.replace('typer', 'mystery-pkg'))


def test_sha_mismatch_is_an_error(tmp_path):
    p = tmp_path / 'Dockerfile.public'
    p.write_text(FAKE)
    with pytest.raises(cd.ComposeError):
        cd.compose(p)
    _, composed = cd.compose(p, allow_other_sha=True)
    assert 'AS official' in composed and 'FROM official AS final' in composed


def test_real_download_composes_if_present():
    p = DATA / 'docker' / 'Dockerfile.public'
    if not p.exists():
        pytest.skip(f'{p} not downloaded (docs/data.md)')
    assert hashlib.sha256(p.read_bytes()).hexdigest() == cd.PUBLIC_SHA256
    public, composed = cd.compose(p)
    official = cd.pin_official(public)
    assert sum(a != b for a, b in zip(public.splitlines(), official.splitlines())) == 2
    assert composed.count('FROM ') == 2
