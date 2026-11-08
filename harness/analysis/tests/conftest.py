from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# reuse the runner's docker/mock fixtures for the end-to-end test
from harness.runner.tests.conftest import DATA, FIXTURE_SUB, image, mock_server, require_snapshots, run_runner  # noqa: E402,F401
