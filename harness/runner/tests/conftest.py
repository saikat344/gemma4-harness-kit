from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get('GEMMA4_DATA_DIR', REPO / 'data' / 'kaggle'))
FIXTURE_SUB = Path(__file__).resolve().parent / 'fixtures' / 'mock_submission'
SAMPLE_SUB = DATA / 'sample_submission'

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _docker_image() -> str | None:
    """Prefer the real sandbox image; fall back to the throwaway test image (docs/runner.md)."""
    try:
        import docker
        client = docker.from_env()
        for tag in filter(None, (os.environ.get('GEMMA4_TEST_IMAGE'), 'swebench-sandbox:latest',
                                 'gemma4-runner-test:latest')):
            try:
                client.images.get(tag)
                return tag
            except Exception:
                continue
    except Exception:
        return None
    return None


@pytest.fixture(scope='session')
def image() -> str:
    img = _docker_image()
    if not img:
        pytest.skip('no docker daemon or no sandbox image (swebench-sandbox:latest / gemma4-runner-test:latest)')
    return img


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def mock_server():
    """Factory: start an in-process mock OpenAI server; returns (endpoint, state)."""
    from harness.runner.mock_openai import MockState, serve

    servers = []

    def _start(mode: str = 'oracle', loras: list[str] | None = None, delay: float = 0.0, inflate: int = 0):
        tasks = DATA / 'tasks.jsonl'
        state = MockState(tasks if tasks.exists() else None, mode, '/models/gemma-4-31b-it-qat-w4a16-ct', loras or [], delay,
                          inflate)
        port = free_port()
        httpd = serve(port, state)
        servers.append(httpd)
        return f'http://127.0.0.1:{port}/v1', state

    yield _start
    for s in servers:
        s.shutdown()


def require_snapshots(*ids: str) -> None:
    missing = [i for i in ids if not (DATA / 'snapshots' / f'{i}.tgz').exists()]
    if missing:
        pytest.skip(f'snapshots missing: {missing}')


def run_runner(*args: str, timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, '-m', 'harness.runner', *args], cwd=REPO, capture_output=True,
                          text=True, timeout=timeout)
