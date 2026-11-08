"""Per-task sandbox environment fixes layered on top of the official swegemma ContainerManager.

The official harness injects the competition ``wheels/`` into site-packages (newest version per
package) and then runs ``sandbox/setup.py``. Some public tasks need packages that the public
wheels/Dockerfile do not provide, or an older ("period-correct") version of one. Packages that
every task can share are baked into the image (``harness/sandbox/Dockerfile.fixes``); version
overrides that only some tasks need are listed in ``harness/sandbox/task_env_fixes.json`` and
applied here, right after each ``setup.py`` call, i.e. after the wheel injection and before
any test run. Override wheels are baked into the image at ``/opt/gemma4/wheels`` (no network
inside the sandbox).

Usage (grader and runner alike)::

    mgr = FixingContainerManager(ContainerConfig(image=...), instance_id='rich_2725', repo='Textualize/rich')

This is a deliberate divergence from the scorer (see docs/grader.md). A repair that exits nonzero
raises ``EnvFixError`` (a swegemma ``ContainerSetupError``): an infrastructure failure, not a verdict.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

from swegemma.harness.container_setup import ContainerSetupError
from swegemma.sandbox import ContainerConfig, ContainerManager

from harness.common.paths import REPO_ROOT

logger = logging.getLogger(__name__)

FIXES_PATH = REPO_ROOT / 'harness' / 'sandbox' / 'task_env_fixes.json'
_SETUP_CALL_RE = re.compile(r'^python3 (?:/tmp|/sandbox)/setup\.py\b')


def load_fixes(path: Path = FIXES_PATH) -> dict[str, Any]:
    if not path.exists():
        return {'tasks': {}, 'repos': {}}
    data = json.loads(path.read_text(encoding='utf-8'))
    data.setdefault('tasks', {})
    data.setdefault('repos', {})
    return data


def fix_commands(instance_id: str, repo: str = '', fixes: dict[str, Any] | None = None) -> list[str]:
    """Shell commands to run in the sandbox for this task ('*', then repo-level, then task-level)."""
    fixes = fixes if fixes is not None else load_fixes()
    short = repo.rsplit('/', 1)[-1]
    cmds: list[str] = []
    for entry in (fixes['repos'].get('*'), fixes['repos'].get(short), fixes['tasks'].get(instance_id)):
        if entry:
            cmds.extend(entry.get('commands', []))
    return cmds


class EnvFixError(ContainerSetupError):
    """A required per-task environment repair exited nonzero (audit F7, 2026-11-05).

    Subclasses swegemma's ``ContainerSetupError``, which both ``verify_task`` and the agent runner
    re-raise instead of folding into a task result: the grader records ``error_class='setup_error'``
    and the runner worker ``status='infra_error'``, i.e. an infrastructure failure, never a model one.
    """


# Integration shim (audit F1, 2026-11-05). swegemma 0.2.11 ``verify_task`` (verification.py ~655)
# treats a missing JUnit XML as fatal only when ``type(docker).__name__`` is one of
# {'ContainerManager', 'SubprocessManager', 'DockerSandbox', 'SubprocessSandbox'}; any other class
# name is assumed to be a test double, and pytest exit 0 + a "passed" stdout summary is then accepted
# without the XML (so the required-test check never runs). This wrapper IS a real Docker sandbox, so
# it reports the upstream class name to get the same strict treatment as ContainerManager. The name
# is also forced on subclasses (e.g. the runner's RecordingContainerManager). ``__qualname__`` keeps
# the real name for debugging. Regression: harness/runner/tests/test_grader_junit_shim.py.
_REAL_SANDBOX_NAME = 'ContainerManager'


def mark_real_sandbox(cls: type) -> type:
    """Make swegemma's class-name sandbox detection treat ``cls`` as a real ContainerManager."""
    cls.__name__ = _REAL_SANDBOX_NAME
    return cls


@mark_real_sandbox
class FixingContainerManager(ContainerManager):
    """ContainerManager that runs per-task fix commands after every sandbox/setup.py call.

    Every repair is required: the first one that exits nonzero raises ``EnvFixError`` (a
    ``ContainerSetupError``) so the caller sees a setup failure rather than a silently unrepaired
    environment. ``fix_results`` records each repair actually run (command, exit code, ok), and
    ``applied_fixes`` lists the commands that succeeded.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        mark_real_sandbox(cls)

    def __init__(
        self,
        config: ContainerConfig | None = None,
        *,
        instance_id: str = '',
        repo: str = '',
        commands: list[str] | None = None,
    ):
        super().__init__(config)
        self.instance_id = instance_id
        self.fix_cmds = commands if commands is not None else fix_commands(instance_id, repo)
        self.fix_log: list[str] = []
        self.fix_results: list[dict[str, Any]] = []
        self._fix_lock = threading.Lock()

    @property
    def applied_fixes(self) -> list[str]:
        """Repair commands that ran and exited 0 (one entry per setup.py call they followed)."""
        with self._fix_lock:
            return [r['command'] for r in self.fix_results if r['ok']]

    def exec(self, container_id: str, command: str, *, timeout: int | None = None):  # type: ignore[override]
        res = super().exec(container_id, command, timeout=timeout)
        if self.fix_cmds and _SETUP_CALL_RE.match(command.strip()):
            for cmd in self.fix_cmds:
                r = super().exec(container_id, cmd, timeout=timeout)
                ok = r.exit_code == 0
                line = f'[envfix {self.instance_id}] exit={r.exit_code} $ {cmd}'
                if r.stdout.strip():
                    line += f'\n  {r.stdout.strip()[-1000:]}'
                if not ok:
                    line += f'\n  stdout: {r.stdout[-2000:]}\n  stderr: {r.stderr[-2000:]}'
                    logger.warning(line)
                with self._fix_lock:
                    self.fix_log.append(line)
                    self.fix_results.append({'command': cmd, 'exit_code': r.exit_code, 'ok': ok})
                if not ok:
                    # Later repairs may depend on this one, and the environment is already not the
                    # one requested: stop and report a setup failure (audit F7).
                    raise EnvFixError(
                        f'env repair failed for {self.instance_id or container_id[:12]} '
                        f'(exit {r.exit_code}): {cmd}\n{(r.stderr or r.stdout)[-1000:]}')
        return res
