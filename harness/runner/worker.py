"""Run ONE task's Phase 1 (agent -> patch) exactly as the scorer-like NB1 driver does, in its own process.

Invoked by the orchestrator as `python -m harness.runner.worker <task_dir>/spec.json`.
One process per task because the swegemma tools are synchronous (run_command blocks the event loop);
in-process asyncio concurrency would serialise tasks and distort the agent clock.

What is official code (unchanged): EvalConfig, SwegemmaContext, ContainerManager, run_agent_sandbox
(sandbox setup, prompt, ADK Runner/App with compaction + context cache, budgets, nudges, submit/fallback
diff, protected-file stripping), the model registry built by adk_submission.VllmServer.create_model_registry.
What we add (observers only): LLM call capture, container-id recording, result.json metrics.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

NB1_ENV = {
    'TRANSFORMERS_NO_TF': '1',
    'TRANSFORMERS_NO_FLAX': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'LITELLM_LOCAL_MODEL_COST_MAP': 'True',
    'LITELLM_TELEMETRY': 'False',
    'LITELLM_LOG': 'WARNING',
    'OPENAI_API_KEY': 'EMPTY',
    'LOCAL_API_KEY': 'EMPTY',
    'OTEL_SDK_DISABLED': 'true',
    'OTEL_PYTHON_DISABLED': 'true',
}


def _api_key(spec: dict) -> str:
    """Endpoint key: spec (old runs / tests) else GEMMA4_ENDPOINT_API_KEY set by the orchestrator."""
    return spec.get('api_key') or os.environ.get('GEMMA4_ENDPOINT_API_KEY') or 'EMPTY'


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        stream=sys.stdout,
    )
    for name in ('LiteLLM', 'LiteLLM Router', 'LiteLLM Proxy', 'litellm', 'httpx', 'httpcore', 'openai',
                 'urllib3', 'opentelemetry', 'opentelemetry.context', 'docker'):
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        if 'litellm' in name.lower():
            lg.propagate = False
            lg.handlers.clear()


def _build_model_registry(endpoint: str, served_model: str, declared_model: str, adapters: Any, api_key: str):
    """Same registry NB1 builds (VllmServer.create_model_registry, prefix openai/, no num_retries),
    pointed at an already-running endpoint instead of a server this process spawns."""
    from adk_submission import VllmConfig, VllmServer

    class _ExternalVllm(VllmServer):
        @property
        def base_url(self) -> str:  # type: ignore[override]
            return endpoint.rstrip('/')

    srv = _ExternalVllm(
        VllmConfig(model=served_model, tool_call_parser='gemma4', reasoning_parser='gemma4'),
        adapter_manifest=adapters,
    )
    with contextlib.suppress(OSError):
        os.unlink(srv.log_path)  # BaseInferenceServer creates an unused temp log file
    return srv.create_model_registry(aliases=[declared_model], model_prefix='openai/', api_key=api_key)


def _key_wheel_cache(wheels: Path) -> None:
    """swegemma caches the unpacked wheels tar in $TMPDIR/swegemma_sp_cache_v8 keyed only by name, so a
    changed wheels/ would silently reuse a stale tar. Key TMPDIR by the wheels listing; same directory
    naming as harness/grader, so runner and grader share the cache."""
    import hashlib
    import tempfile

    try:
        from harness.common.paths import cache_dir
        base = cache_dir()
    except Exception:
        base = Path.home() / '.cache' / 'gemma4'
    listing = '\n'.join(sorted(f'{p.name}:{p.stat().st_size}' for p in wheels.glob('*.whl')))
    d = base / f'swegemma-tmp-{hashlib.sha256(listing.encode()).hexdigest()[:12]}'
    d.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(d)


def _recording_manager(container_config: Any, record_path: Path, *, env_fixes: bool, instance_id: str,
                       repo: str):
    """ContainerManager that records container ids; with env_fixes it is harness.common.envfix's
    FixingContainerManager (same per-task test-env fixes the grader applies; see docs/grader.md)."""
    from swegemma.sandbox import ContainerManager

    base: Any = ContainerManager
    kwargs: dict[str, Any] = {}
    if env_fixes:
        from harness.common.envfix import FixingContainerManager

        base = FixingContainerManager
        kwargs = {'instance_id': instance_id, 'repo': repo}

    class RecordingContainerManager(base):  # type: ignore[misc, valid-type]
        """Logs container ids so the orchestrator can remove them if this process is killed."""

        def start(self) -> str:
            cid = super().start()
            with open(record_path, 'a', encoding='utf-8') as f:
                f.write(cid + '\n')
            return cid

    return RecordingContainerManager(container_config, **kwargs)


# --------------------------------------------------------------------------------------------------
# Metrics from the trace and the call log
# --------------------------------------------------------------------------------------------------


def _parse_result(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


def _error_of(result: Any) -> tuple[str, str] | None:
    """Classify a tool result as an error: (error_type, message) or None."""
    if isinstance(result, dict):
        inner = result.get('result') if isinstance(result.get('result'), dict) else result
        if inner.get('status') == 'error':
            return str(inner.get('error_type') or inner.get('error_code') or 'Error'), str(
                inner.get('error_message') or inner.get('error') or '')[:500]
        if 'error' in inner and 'status' not in inner:
            msg = str(inner['error'])
            et = 'MissingArguments' if 'mandatory input parameters' in msg else 'ToolInvocationError'
            return et, msg[:500]
    elif isinstance(result, str) and result.lower().startswith('error'):
        return 'Error', result[:500]
    return None


def trace_metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    tool_calls = Counter()
    tool_errors: dict[str, Counter] = defaultdict(Counter)
    error_samples: list[dict[str, Any]] = []
    unparsed_tool_text = 0
    for e in entries:
        t = e.get('type')
        if t == 'tool_call':
            tool_calls[e.get('tool', '?')] += 1
        elif t == 'tool_response':
            err = _error_of(_parse_result(e.get('result', '')))
            if err:
                tool_errors[e.get('tool', '?')][err[0]] += 1
                if len(error_samples) < 50:
                    error_samples.append({'tool': e.get('tool'), 'error_type': err[0], 'message': err[1],
                                          'elapsed_s': e.get('elapsed_s')})
        elif t in ('text', 'final', 'thinking'):
            if '<|tool_call>' in (e.get('content') or ''):
                unparsed_tool_text += 1
    return {
        'tool_call_breakdown': dict(tool_calls),
        'tool_calls_total': sum(tool_calls.values()),
        'tool_errors': {k: dict(v) for k, v in tool_errors.items()},
        'tool_errors_total': sum(sum(v.values()) for v in tool_errors.values()),
        'tool_error_samples': error_samples,
        'unparsed_tool_call_text': unparsed_tool_text,
        'nudges': sum(1 for e in entries if e.get('type') == 'continuation_nudge'),
        'compactions': sum(1 for e in entries if e.get('type') == 'compaction'),
        'thinking_entries': sum(1 for e in entries if e.get('type') == 'thinking'),
    }


def call_metrics(calls_path: Path) -> dict[str, Any]:
    agg = Counter()
    finish = Counter()
    max_prompt = 0
    latencies: list[float] = []
    n_agent = n_summ = n_err = 0
    if calls_path.exists():
        for line in calls_path.read_text(encoding='utf-8').splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get('error'):
                n_err += 1
                continue
            if rec.get('kind') == 'summarizer':
                n_summ += 1
            else:
                n_agent += 1
            u = rec.get('usage') or {}
            pt = int(u.get('prompt_tokens') or 0)
            ct = int(u.get('completion_tokens') or 0)
            agg['prompt_tokens'] += pt
            agg['completion_tokens'] += ct
            details = u.get('completion_tokens_details') or {}
            agg['reasoning_tokens'] += int((details or {}).get('reasoning_tokens') or 0)
            max_prompt = max(max_prompt, pt)
            finish[str(rec.get('finish_reason'))] += 1
            if rec.get('latency_s') is not None:
                latencies.append(float(rec['latency_s']))
    total = agg['prompt_tokens'] + agg['completion_tokens']
    llm_s = sum(latencies)
    return {
        'llm_calls_agent': n_agent,
        'llm_calls_summarizer': n_summ,
        'llm_call_errors': n_err,
        'prompt_tokens': agg['prompt_tokens'],
        'completion_tokens': agg['completion_tokens'],
        'reasoning_tokens': agg['reasoning_tokens'],
        'total_tokens': total,
        'max_prompt_tokens': max_prompt,
        'finish_reasons': dict(finish),
        'llm_time_s': round(llm_s, 3),
        'completion_tok_per_llm_s': round(agg['completion_tokens'] / llm_s, 2) if llm_s else None,
    }


def agent_exception_after_submit(agent_error: str | None, submitted: bool) -> bool:
    """A sandbox execution error that killed the agent after a submit_patch (termination stays 'submitted')."""
    return submitted and 'sandbox execution error' in (agent_error or '').lower()


def classify_termination(agent_error: str | None, submitted: bool, patch: str, nudges: int) -> str:
    e = (agent_error or '').lower()
    if 'session timeout' in e:
        return 'time_budget'
    if 'turns budget' in e or 'maximum allowed llm turns' in e:
        return 'turn_budget'
    if 'tool call budget' in e:
        return 'tool_budget'
    if submitted:  # before agent_exception; the exception is kept as agent_exception_after_submit
        return 'submitted'
    if 'sandbox execution error' in e:
        return 'agent_exception'
    if 'without calling submit_patch' in e:
        return 'no_submit_no_diff'
    if nudges >= 3:
        return 'nudges_exhausted'
    return 'ended_without_submit' if patch.strip() else 'other'


# --------------------------------------------------------------------------------------------------


async def _run(spec: dict[str, Any]) -> dict[str, Any]:
    import litellm
    from adk_submission import discover_adapters
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
    from swegemma.context import SwegemmaContext
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.harness.agent_runner import run_agent_sandbox
    from swegemma.harness.verification import save_trace_artifact
    from swegemma.models import load_tasks
    from swegemma.sandbox import ContainerConfig
    from swegemma.warnings import _suppress_warnings

    from harness.runner import capture

    _suppress_warnings()
    litellm.drop_params = True

    task_dir = Path(spec['task_dir'])
    agent_dir = Path(spec['submission_dir'])
    data = spec['data']
    b = spec['budgets']
    iid = spec['instance_id']

    if data.get('wheels_dir'):
        _key_wheel_cache(Path(data['wheels_dir']))
    tasks = {t.instance_id: t for t in load_tasks(Path(data['tasks_path']))}
    task = tasks[iid]

    adapters = discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    models = _build_model_registry(spec['endpoint'], spec['served_model'], spec['declared_model'],
                                   adapters, _api_key(spec))
    limits, gen = build_submission_limits()
    cfg = EvalConfig(
        tasks_path=Path(data['tasks_path']),
        snapshots_dir=Path(data['snapshots_dir']),
        results_dir=task_dir,
        submission_dir=agent_dir,
        models=models,
        sandbox='docker',
        image=spec['image'],
        timeout_seconds=b['timeout_seconds'],
        verification_timeout_seconds=b['verification_timeout_seconds'],
        max_time_minutes=b['max_time_minutes'],
        max_tool_calls=b['max_tool_calls'],
        max_turns=b['max_turns'],
        limits=limits,
        generation_constraints=gen,
        adapter_manifest=adapters,
        context_cache_config=(ContextCacheConfig(**spec['context_cache']) if spec.get('context_cache') else None),
        events_compaction_config=(EventsCompactionConfig(**spec['compaction']) if spec.get('compaction') else None),
        graph_dir=data['graph_dir'],
        embeddings_dir=data['embeddings_dir'],
        wheels_dir=Path(data['wheels_dir']) if data.get('wheels_dir') else None,
        concurrency=1,
        task_ids=[iid],
        display_mode='quiet',
        verbose=False,
    )
    cmd_timeout = cfg.harness.command_timeout_seconds if cfg.harness.command_timeout_seconds is not None else 300
    mgr = _recording_manager(ContainerConfig(image=cfg.image, timeout_seconds=cmd_timeout),
                             task_dir / 'containers.txt', env_fixes=bool(spec.get('env_fixes')),
                             instance_id=iid, repo=task.repo)
    context = SwegemmaContext(
        docker_manager=mgr,
        container_id='',
        problem_statement=task.problem_statement,
        hints_text=getattr(task, 'hints_text', '') or '',
        task=task,
        repo=task.repo,
        graph_dir=cfg.graph_dir,
        embeddings_dir=cfg.embeddings_dir,
        budget=cfg.budget,
        harness=cfg.harness,
    )
    recorder = capture.CallRecorder(task_dir / 'llm_calls.jsonl', context)
    checkpointer = None
    if spec.get('checkpoints'):
        from harness.runner.checkpoints import Checkpointer
        checkpointer = Checkpointer(task_dir, context, spec['checkpoints'],
                                    test_files=list(getattr(task, 'test_files', None) or ()))
    capture.install(recorder, checkpointer)
    capture.install_compaction_hook(task_dir / 'compactions.jsonl', recorder)

    snapshot_path, base_snapshot_path, patch_path = resolve_task_snapshot_paths(
        cfg.snapshots_dir, task.instance_id, task.repo)
    if not snapshot_path.exists():
        raise FileNotFoundError(f'Snapshot not found: {snapshot_path}')

    wall0 = time.time()
    t0 = time.perf_counter()
    try:
        patch, agent_error, trace = await run_agent_sandbox(
            mgr, cfg, task, snapshot_path,
            base_snapshot_path=base_snapshot_path, patch_path=patch_path,
            task_index=spec.get('task_index', 1), total_tasks=spec.get('total_tasks', 1),
            context=context,
        )
    finally:
        recorder.close()
        with contextlib.suppress(Exception):
            mgr.cleanup_all()
        if getattr(mgr, 'fix_log', None):  # also on a failed repair (EnvFixError), so the infra_error has its log
            (task_dir / 'env_fixes.log').write_text('\n'.join(mgr.fix_log) + '\n', encoding='utf-8')
    total_s = time.perf_counter() - t0

    patch = patch or ''
    (task_dir / 'patch.diff').write_text(patch, encoding='utf-8')
    if checkpointer is not None:
        checkpointer.finish(patch, bool(context.patch_submitted))
    save_trace_artifact(trace, task_dir, iid)  # ATIF v1.7, same file the harness writes
    trace.save(task_dir / 'trace_legacy.json', format='legacy')
    entries = trace.to_dict(format='legacy')['entries']

    tm = trace_metrics(entries)
    cm = call_metrics(task_dir / 'llm_calls.jsonl')
    comp_path = task_dir / 'compactions.jsonl'
    tm['compactions_in_trace'] = tm.pop('compactions')
    tm['compactions'] = len(comp_path.read_text().splitlines()) if comp_path.exists() else 0
    setup_s = ((context.agent_start_time - context.task_start_time)
               if context.agent_start_time is not None else None)
    submitted = bool(context.patch_submitted)
    return {
        'status': 'done',
        'instance_id': iid,
        'repo': task.repo,
        'task_index': spec.get('task_index'),
        'started_at': wall0,
        'patch_chars': len(patch.strip()),
        'patch_files': patch.count('diff --git'),
        'empty_patch': not patch.strip(),
        'patch_submitted': submitted,
        'patch_source': 'submit_patch' if submitted and patch.strip() else ('fallback_diff' if patch.strip() else 'none'),
        'agent_error': agent_error,
        'termination': classify_termination(agent_error, submitted, patch, tm['nudges']),
        'agent_exception_after_submit': agent_exception_after_submit(agent_error, submitted),
        'timings': {
            'total_s': round(total_s, 3),
            'setup_s': round(setup_s, 3) if setup_s is not None else None,
            'agent_s': round(float(context.agent_elapsed_seconds or 0.0), 3),
        },
        'budget_final': recorder.budget_state(),
        'harness_tool_calls_used': context.tool_calls_used,
        'harness_llm_calls_used': context.llm_calls_used,
        **tm,
        **cm,
    }


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    spec_path = Path(argv[0])
    spec = json.loads(spec_path.read_text())
    task_dir = Path(spec['task_dir'])
    for k, v in NB1_ENV.items():
        os.environ.setdefault(k, v)
    os.environ['OPENAI_API_KEY'] = _api_key(spec)
    _configure_logging(spec.get('log_level', 'INFO'))
    result_path = task_dir / 'result.json'
    t0 = time.time()

    async def _main() -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        me = asyncio.current_task()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, me.cancel)
        return await _run(spec)

    try:
        result = asyncio.run(_main())
    except (asyncio.CancelledError, KeyboardInterrupt):
        result = {'status': 'killed', 'instance_id': spec['instance_id'], 'error': 'worker cancelled (SIGTERM/SIGINT)'}
    except Exception as e:
        result = {'status': 'infra_error', 'instance_id': spec['instance_id'],
                  'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()}
    result['worker_wall_s'] = round(time.time() - t0, 3)
    tmp = result_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
    tmp.replace(result_path)
    print(f"[worker] {spec['instance_id']}: {result['status']} "
          f"{result.get('termination', '')} patch_chars={result.get('patch_chars', 0)}")
    return 0 if result['status'] == 'done' else 1


if __name__ == '__main__':
    sys.exit(main())
