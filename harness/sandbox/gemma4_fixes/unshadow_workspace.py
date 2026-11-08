"""Remove injected site-packages copies of the repo's OWN packages so tests import /workspace.

swegemma injects every competition wheel (newest version) into site-packages AFTER the editable
install. For src-layout repos (requests: src/requests, httpx: src/httpx) the injected copy
(requests 2.34.2, httpx 0.28.1) then shadows the workspace, because .pth/editable paths are
appended after site-packages and the root-conftest sys.path trick only adds /workspace.
Without this, gold and empty patches test the same installed library. Root-layout repos
(fastapi, rich) are normally saved by the root conftest, but a plugin importing them first
(e.g. inline-snapshot imports rich) would still pick the injected copy.

Only deletes site-packages entries that share a name with a top-level package in /workspace
or /workspace/src, plus their non-editable *.dist-info. Idempotent. Run with python3 -S.
"""

import re
import shutil
import sys
from pathlib import Path

WS = Path('/workspace')
SP = Path('/usr/local/lib/python3.13/site-packages')
SKIP = {'tests', 'test', 'testing', 'docs', 'docs_src', 'scripts', 'examples', 'benchmarks', 'tools', 'src'}


def norm(name: str) -> str:
    return re.sub(r'[-_.]+', '_', name).lower()


def workspace_packages() -> set[str]:
    names: set[str] = set()
    for root in (WS / 'src', WS):
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if d.is_dir() and not d.name.startswith('.') and d.name not in SKIP and (d / '__init__.py').exists():
                names.add(d.name)
    return names


def main() -> int:
    removed = []
    pkgs = workspace_packages()
    for name in sorted(pkgs):
        target = SP / name
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
            removed.append(str(target))
        for di in SP.glob('*.dist-info'):
            dist = di.name[: -len('.dist-info')].rsplit('-', 1)[0]
            if norm(dist) == norm(name) and not (di / 'direct_url.json').exists():
                shutil.rmtree(di)
                removed.append(str(di))
    print(f'unshadow: workspace packages={sorted(pkgs)} removed={removed}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
