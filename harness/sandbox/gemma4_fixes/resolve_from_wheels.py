"""Swap injected wheels for the versions the snapshot's own pins/ranges ask for (from /wheels only).

swegemma's fast path injects the NEWEST version of every competition wheel (e.g. starlette
1.6.0), but /wheels deliberately holds many versions (50 starlette, 5 sqlmodel, 3 typer, 2
werkzeug/flask...), matching the pins in the snapshots' requirement files. fastapi snapshots
declare starlette<0.49, so with 1.6.0 every fastapi gold patch fails to collect. The public
scorer code has no per-task resolution, so (inferred) the real scorer environment resolves
these some other way; this script is our stand-in. See docs/grader.md.

Constraints = the editable install's Requires-Dist (no extras) + root requirements*.txt lines
(following -r), intersected per package. For every constrained package that /wheels has, if the
installed version does not satisfy the constraint, the highest satisfying py3.13-compatible
wheel replaces it (files removed + wheel unpacked, like swegemma's injector). Repeats for the
swapped packages' own requirements. Never touches pytest/pluggy/pip/setuptools/wheel or the
repo's own package. Offline, idempotent.
"""

import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

WS = Path('/workspace')
SP = Path('/usr/local/lib/python3.13/site-packages')
WHEELS = Path('/wheels')
SKIP = {'pytest', 'pluggy', 'pip', 'setuptools', 'wheel', 'pytest_timeout', 'pytest_asyncio',
        'pytest_httpbin', 'pytest_xdist', 'pytest_cov', 'coverage'}
ENV = default_environment()


def norm(n: str) -> str:
    return re.sub(r'[-_.]+', '_', n).lower()


def py313_ok(fn: str) -> bool:  # same rule as swegemma's _is_wheel_compatible_py313
    parts = fn[:-4].split('-')
    py, abi = parts[-3], parts[-2]
    if abi in ('cp38', 'cp39', 'cp310', 'cp311', 'cp312') and py in ('cp38', 'cp39', 'cp310', 'cp311', 'cp312'):
        return False
    return abi == 'none' or 'abi3' in abi or 'cp313' in py


def wheel_index(extra_dirs=()) -> dict:
    idx: dict = {}
    for w in [*WHEELS.glob('*.whl'), *(x for d in extra_dirs for x in Path(d).glob('*.whl'))]:
        if not py313_ok(w.name):
            continue
        name, ver = w.name.split('-')[:2]
        try:
            idx.setdefault(norm(name), []).append((Version(ver), w))
        except InvalidVersion:
            pass
    return idx


def installed() -> tuple[dict, set]:
    vers, editable = {}, set()
    for di in SP.glob('*.dist-info'):
        name, ver = di.name[:-len('.dist-info')].rsplit('-', 1)
        du = di / 'direct_url.json'
        if du.exists() and 'editable' in du.read_text(errors='ignore'):
            editable.add(norm(name))
            continue
        try:
            vers[norm(name)] = (Version(ver), di)
        except InvalidVersion:
            pass
    return vers, editable


def req_ok(r: Requirement) -> bool:
    if r.marker is None:
        return True
    try:
        return r.marker.evaluate({**ENV, 'extra': ''})
    except Exception:
        return False


def metadata_reqs(meta: str) -> list:
    out = []
    for line in meta.splitlines():
        if line.startswith('Requires-Dist:'):
            try:
                r = Requirement(line.split(':', 1)[1].strip())
            except InvalidRequirement:
                continue
            if req_ok(r):
                out.append(r)
    return out


def requirement_file_reqs(path: Path, seen: set) -> list:
    if path in seen or not path.is_file():
        return []
    seen.add(path)
    out = []
    for raw in path.read_text(errors='ignore').splitlines():
        line = raw.split(' #', 1)[0].strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith(('-r ', '--requirement ')):
            out += requirement_file_reqs(path.parent / line.split(None, 1)[1].strip(), seen)
            continue
        if line.startswith('-'):
            continue
        try:
            r = Requirement(line)
        except InvalidRequirement:
            continue
        if req_ok(r):
            out.append(r)
    return out


def project_reqs(editable: set) -> list:
    reqs = []
    for di in SP.glob('*.dist-info'):
        du = di / 'direct_url.json'
        if du.exists() and 'file:///workspace' in du.read_text(errors='ignore') and (di / 'METADATA').exists():
            reqs += metadata_reqs((di / 'METADATA').read_text(errors='ignore'))
    if not reqs and (WS / 'pyproject.toml').exists():  # editable install failed: PEP 621 fallback
        import tomllib
        try:
            data = tomllib.loads((WS / 'pyproject.toml').read_text())
            for s in data.get('project', {}).get('dependencies', []):
                r = Requirement(s)
                if req_ok(r):
                    reqs.append(r)
        except Exception:
            pass
    seen: set = set()
    for f in sorted(WS.glob('requirements*.txt')):
        reqs += requirement_file_reqs(f, seen)
    return reqs


def top_levels(whl: Path) -> set:
    tops = set()
    with zipfile.ZipFile(whl) as zf:
        for n in zf.namelist():
            top = n.split('/', 1)[0]
            if top.endswith('.data'):
                parts = n.split('/')
                if len(parts) > 2 and parts[1] in ('purelib', 'platlib'):
                    tops.add(parts[2])
            elif not top.endswith('.dist-info'):
                tops.add(top)
    return tops


def unpack(whl: Path) -> None:
    with zipfile.ZipFile(whl) as zf:
        for m in zf.infolist():
            if m.filename.endswith(('.pyc', '/RECORD')):
                continue
            parts = m.filename.split('/')
            if parts[0].endswith('.data'):
                if len(parts) > 2 and parts[1] in ('purelib', 'platlib'):
                    m.filename = '/'.join(parts[2:])
                else:
                    continue
            zf.extract(m, SP)


def main() -> int:
    # --pin SPEC (repeatable): extra constraint, e.g. Pygments==2.15.1 (task_env_fixes.json)
    # --wheels-dir DIR (repeatable): extra wheel source, e.g. /opt/gemma4/wheels
    args, pins, extra = sys.argv[1:], [], []
    while args:
        a = args.pop(0)
        if a == '--pin':
            pins.append(Requirement(args.pop(0)))
        elif a == '--wheels-dir':
            extra.append(args.pop(0))
    idx = wheel_index(extra)
    vers, editable = installed()
    own = editable
    pending = project_reqs(editable) + pins
    swapped, notes = [], []
    seen_specs: dict = {}
    for _ in range(6):
        if not pending:
            break
        for r in pending:
            seen_specs.setdefault(norm(r.name), SpecifierSet())
            seen_specs[norm(r.name)] &= r.specifier
        pending = []
        for name, spec in sorted(seen_specs.items()):
            if name in SKIP or name in own or name not in idx:
                continue
            cur = vers.get(name)
            if cur is not None and spec.contains(cur[0], prereleases=True):
                continue
            cands = sorted((v, w) for v, w in idx[name] if spec.contains(v, prereleases=True))
            if not cands:
                notes.append(f'{name}: no wheel satisfies {spec}; kept {cur[0] if cur else None}')
                continue
            new_v, new_w = cands[-1]
            if cur is not None:
                old_w = next((w for v, w in idx[name] if v == cur[0]), None)
                for t in (top_levels(old_w) if old_w else set()) | top_levels(new_w):
                    p = SP / t
                    if p.is_dir() and not p.is_symlink():
                        shutil.rmtree(p)
                    elif p.exists():
                        p.unlink()
            for di in SP.glob('*.dist-info'):  # all copies (image pip install + injected)
                if norm(di.name[:-len('.dist-info')].rsplit('-', 1)[0]) == name and not (di / 'direct_url.json').exists():
                    shutil.rmtree(di, ignore_errors=True)
            unpack(new_w)
            vers[name] = (new_v, None)
            swapped.append(f'{name} {cur[0] if cur else None}->{new_v}')
            with zipfile.ZipFile(new_w) as zf:
                meta = next((n for n in zf.namelist() if n.endswith('.dist-info/METADATA')), None)
                if meta:
                    pending += metadata_reqs(zf.read(meta).decode('utf-8', 'ignore'))
    print('resolve_from_wheels: swapped=' + json.dumps(swapped) + (' notes=' + json.dumps(notes) if notes else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
