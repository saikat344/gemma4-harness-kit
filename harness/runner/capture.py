"""Record every LLM request/response the agent makes (observer only; it never changes a call).

The harness logs ADK events (trace.json) but not the raw chat requests. For analysis and SFT data we
want the exact OpenAI-format `messages` + `tools` the model saw (after compaction and after the
adk_submission reasoning patch), the raw response (content, reasoning_content, tool_calls), usage and
latency, plus the budget state at call time. We wrap `google.adk.models.lite_llm.LiteLLMClient
.acompletion`, which is the single choke point for agent calls *and* ADK's compaction summarizer.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

_SUMMARIZER_MARKER = 'The following is a conversation history between a user and an AI agent'


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    for attr in ('model_dump', 'dict', 'to_dict'):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _jsonable(fn())
            except Exception:
                pass
    if hasattr(obj, '__dict__'):
        return {k: _jsonable(v) for k, v in vars(obj).items() if not k.startswith('_')}
    return repr(obj)


def _first_text(messages: list[Any]) -> str:
    for m in messages or []:
        c = m.get('content') if isinstance(m, dict) else getattr(m, 'content', None)
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and isinstance(p.get('text'), str):
                    return p['text']
    return ''


class CallRecorder:
    """Appends one JSON line per LLM call to `path`. Thread-safe; flushes every line (crash-safe)."""

    def __init__(self, path: Path, context: Any | None = None) -> None:
        self.path = path
        self.context = context
        self._lock = threading.Lock()
        self._n = 0
        self._fh = open(path, 'a', encoding='utf-8')  # noqa: SIM115

    def budget_state(self) -> dict[str, Any]:
        ctx = self.context
        if ctx is None:
            return {}
        b = getattr(ctx, 'budget', None)
        return {
            'tool_calls_used': getattr(ctx, 'tool_calls_used', None),
            'max_tool_calls': getattr(b, 'tool_calls', None),
            'llm_calls_used': getattr(ctx, 'llm_calls_used', None),
            'max_turns': getattr(b, 'turns', None),
            'agent_elapsed_s': round(float(getattr(ctx, 'agent_elapsed_seconds', 0.0) or 0.0), 3),
            'remaining_time_s': (
                round(r, 3) if (r := getattr(ctx, 'remaining_time_seconds', None)) is not None else None
            ),
            'patch_submitted': bool(getattr(ctx, 'patch_submitted', False)),
        }

    def write(self, rec: dict[str, Any]) -> None:
        line = json.dumps(_jsonable(rec), ensure_ascii=False, default=repr)
        with self._lock:
            self._fh.write(line + '\n')
            self._fh.flush()

    def next_index(self) -> int:
        with self._lock:
            self._n += 1
            return self._n

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def install(recorder: CallRecorder, checkpointer: Any | None = None) -> None:
    """Patch LiteLLMClient.acompletion in this process (one task per process, so no cross-talk).

    With a checkpoints.Checkpointer, poll it before each call and when the response arrives (time checkpoints)."""
    from google.adk.models import lite_llm

    cls = lite_llm.LiteLLMClient
    orig = getattr(cls, '_gemma4_orig_acompletion', None) or cls.acompletion
    cls._gemma4_orig_acompletion = orig

    async def acompletion(self, model, messages, tools, **kwargs):  # noqa: ANN001
        idx = recorder.next_index()
        if checkpointer is not None:
            checkpointer.poll('before_call', idx)
        kind = 'summarizer' if (not tools and _SUMMARIZER_MARKER in _first_text(messages)) else 'agent'
        rec: dict[str, Any] = {
            'call_index': idx,
            'kind': kind,
            't_start': time.time(),
            'budget_before': recorder.budget_state(),
            'request': {
                'model': model,
                'messages': messages,
                'tools': tools,
                'params': {k: v for k, v in kwargs.items() if k not in ('api_key',)},
            },
        }
        t0 = time.perf_counter()
        try:
            resp = await orig(self, model, messages, tools, **kwargs)
        except BaseException as e:
            rec['latency_s'] = round(time.perf_counter() - t0, 3)
            rec['error'] = f'{type(e).__name__}: {e}'
            recorder.write(rec)
            raise
        rec['latency_s'] = round(time.perf_counter() - t0, 3)
        if kwargs.get('stream'):
            rec['response'] = None
            rec['note'] = 'streaming response not captured'
        else:
            dumped = _jsonable(resp)
            rec['response'] = dumped
            try:
                choice = (dumped.get('choices') or [{}])[0]
                rec['finish_reason'] = choice.get('finish_reason')
                rec['usage'] = dumped.get('usage')
            except Exception:
                pass
        recorder.write(rec)
        if checkpointer is not None:
            checkpointer.poll('after_response', idx)
        return resp

    cls.acompletion = acompletion


def install_compaction_hook(path: Path, recorder: CallRecorder) -> None:
    """Record ADK compaction events. Token-threshold compaction (the scorer's EventsCompactionConfig
    token_threshold=14336) runs inside the LLM flow and appends its event straight to the session, so it
    never reaches the harness's plugin/trace callback; the official trace therefore shows 0 compactions.
    We observe InMemorySessionService.append_event instead (no behaviour change)."""
    from google.adk.sessions.in_memory_session_service import InMemorySessionService

    cls = InMemorySessionService
    orig = getattr(cls, '_gemma4_orig_append_event', None) or cls.append_event
    cls._gemma4_orig_append_event = orig
    lock = threading.Lock()

    async def append_event(self, session, event):  # noqa: ANN001
        res = await orig(self, session, event)
        comp = getattr(getattr(event, 'actions', None), 'compaction', None)
        if comp:
            summary = ''
            content = getattr(comp, 'compacted_content', None)
            for part in getattr(content, 'parts', None) or []:
                if getattr(part, 'text', None):
                    summary += part.text
            n_events = sum(
                1 for e in getattr(session, 'events', []) or []
                if comp.start_timestamp <= (getattr(e, 'timestamp', 0) or 0) <= comp.end_timestamp
            )
            rec = {
                't': time.time(),
                'after_llm_call_index': recorder._n,
                'start_timestamp': comp.start_timestamp,
                'end_timestamp': comp.end_timestamp,
                'events_in_window': n_events,
                'session_events_total': len(getattr(session, 'events', []) or []),
                'summary': summary,
                'budget': recorder.budget_state(),
            }
            line = json.dumps(_jsonable(rec), ensure_ascii=False, default=repr)
            with lock, open(path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        return res

    cls.append_event = append_event
