"""P-E5 serializer replay: what survives ADK compaction with the text-only 1.36.1 serializer (arm T) vs the
typed 2.2.0 serializer (arm Y), on the same compacted windows. Spec: docs/pe5.md.

    uv run --project harness/runner python -m harness.analysis.pe5 prepare \\
        --runs runs/pe5/source/baseline-v0-A runs/pe5/source/baseline-v0-B --tasks data/kaggle/tasks.jsonl --out runs/pe5
    uv run --project harness/runner python -m harness.analysis.pe5 run --endpoint http://127.0.0.1:8000/v1 --out runs/pe5
    uv run --project harness/runner python -m harness.analysis.pe5 report --out runs/pe5

prepare (no network)
  For every task-run with a compactions.jsonl, take the FIRST compaction. Its `after_llm_call_index` is the
  call index of the summarizer call itself (capture.py records the counter after the summarizer returned), so
  the window is rebuilt from the last agent call before it: that call's request messages (system dropped) plus
  its response, then the tool results / nudges that followed (taken from the next agent request, or
  placeholders when the run ended). Messages become google-adk 1.36.1 Events (assistant content -> text part,
  reasoning_content -> thought part, tool_calls -> function_call parts, consecutive tool messages -> one
  function_response event, user -> text part), and ADK's own
  `apps.compaction._events_to_compact_for_token_threshold(event_retention_size=5)` picks the window (last 5
  retained, split moved so no call is separated from its response, pending calls never compacted).
  Validity: 1.36.1 `LlmEventSummarizer._format_events_for_prompt(window)` must equal the history text of the
  logged summarizer request, ignoring whitespace. Then 20 distinct tasks are sampled (stratified by repo,
  seed 20261103), facts are frozen to facts.jsonl and both arms' requests go to requests.jsonl.
run
  Sends each request `--samples` times per arm (sequential, resumable; OpenAI chat completions, model id
  from /v1/models, bearer key from $VLLM_API_KEY if set) and appends to calls.jsonl.
report
  Availability (fact in the arm's summarizer input) and retention (fact in the summary) per arm, tokens,
  latency -> results.json and a generated markdown draft (--doc; default <out>/pe5-results.md).

Facts (mechanical, case-insensitive substring match; tokens shorter than 4 chars dropped):
  issue (<=5, from tasks.jsonl problem_statement, URLs removed): backticked spans (outside ``` fences,
    <=80 chars, one line) -> dotted or CamelCase names -> file paths (with an extension) -> error class names,
    each in order of appearance, de-duplicated. Deviation from the spec: tokens equal to the repo owner or
    name (e.g. `fastapi`), Python keywords and builtin names (`None`, `yield`, `input`) are dropped, and a
    candidate contained in (or containing) an already chosen fact is skipped.
  tool (<=5, from the window's tool calls and results): the target of the last edit_file first, then
    round-robin over failing test ids (`FAILED a::b`, `ERROR a::b` in tool output), exception class names
    RAISED in run_command output (at line start, optionally module-qualified; after pytest's `E   `; or after
    `raise`; fields other than status/error_type/filepath/strategy; fixed before the GPU run, 2026-11-03,
    so names that only appear in source read via read_file do not count), and file paths read or edited
    (read_file/edit_file/write_file `filepath`, `/workspace/` and `./` stripped; parser-mangled values that
    are not a plain path are skipped), each in order of appearance.
"""

from __future__ import annotations

import argparse
import builtins
import json
import keyword
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SEED = 20261103
N_TASKS = 20
RETENTION = 5
ARMS = ('T', 'Y')
ARM_DESC = {'T': 'text-only serializer (google-adk 1.36.1, what the scorer runs)',
            'Y': 'typed serializer (google-adk 2.2.0, thoughts + tool calls + truncated tool responses)'}
MAX_ISSUE_FACTS = 5
MAX_TOOL_FACTS = 5
MIN_FACT_LEN = 4
_DROP_PARAMS = {'api_base', 'api_key', 'base_url', 'stream', 'model', 'messages', 'tools'}
_NON_OUTPUT_KEYS = {'status', 'error_type', 'filepath', 'strategy'}
DEFAULT_DOC: Path | None = None  # None: <out>/pe5-results.md


# ----------------------------------------------------------------------------------------------------- io

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a torn last line from a killed worker
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)


def _adk():
    """google-adk 1.36.1 pieces (imported lazily so `--help` works without the runner venv)."""
    from google.adk.apps import compaction
    from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
    from google.adk.events.event import Event
    from google.genai import types
    return compaction, LlmEventSummarizer, Event, types


def template_136() -> str:
    _, summ, _, _ = _adk()
    return summ._DEFAULT_PROMPT_TEMPLATE


def format_136(events: list[Any]) -> str:
    _, summ, _, _ = _adk()
    return summ(llm=None)._format_events_for_prompt(events)


def format_220(events: list[Any]) -> str:
    from harness.analysis.adk220_format import format_events_for_prompt
    return format_events_for_prompt(events)


def build_prompt(history: str) -> str:
    return template_136().format(conversation_history=history)


# ------------------------------------------------------------------------------------------ reconstruction

def _text_items(content: Any) -> list[str]:
    if content is None:
        return []
    if isinstance(content, str):
        return [content] if content else []
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get('text'), str) and p['text']:
                out.append(p['text'])
            elif isinstance(p, str) and p:
                out.append(p)
        return out
    return [str(content)]


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw or '{}')
        return v if isinstance(v, dict) else {'value': v}
    except (json.JSONDecodeError, TypeError):
        return {'_raw': raw}


def _parse_response(content: Any) -> Any:
    """Inverse of lite_llm's `_safe_json_serialize(function_response.response)`."""
    if isinstance(content, list):
        content = ''.join(_text_items(content))
    if isinstance(content, str):
        try:
            v = json.loads(content)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
    return content if isinstance(content, dict) else {'result': content}


class _Builder:
    def __init__(self, agent_name: str) -> None:
        _, _, self.Event, self.types = _adk()
        self.agent = agent_name
        self.events: list[Any] = []
        self.call_names: dict[str, str] = {}
        self._pending_tool: list[Any] = []
        self._ts = 1000.0

    def _emit(self, author: str, role: str, parts: list[Any]) -> None:
        self._ts += 1.0
        self.events.append(self.Event(author=author, timestamp=self._ts, invocation_id='pe5',
                                      content=self.types.Content(role=role, parts=parts)))

    def _flush_tool(self) -> None:
        if self._pending_tool:
            self._emit(self.agent, 'user', self._pending_tool)
            self._pending_tool = []

    def assistant(self, msg: dict[str, Any]) -> None:
        self._flush_tool()
        t = self.types
        parts = []
        reasoning = msg.get('reasoning_content')
        if reasoning is None:
            reasoning = msg.get('reasoning')
        for txt in _text_items(reasoning):
            parts.append(t.Part(text=txt, thought=True))
        for txt in _text_items(msg.get('content')):
            parts.append(t.Part.from_text(text=txt))
        for tc in msg.get('tool_calls') or []:
            fn = tc.get('function') or {}
            name = fn.get('name') or ''
            p = t.Part.from_function_call(name=name, args=_parse_args(fn.get('arguments')))
            p.function_call.id = tc.get('id')
            self.call_names[tc.get('id')] = name
            parts.append(p)
        if parts:
            self._emit(self.agent, 'model', parts)

    def tool(self, msg: dict[str, Any], response: Any | None = None) -> None:
        cid = msg.get('tool_call_id')
        fr = self.types.FunctionResponse(id=cid, name=self.call_names.get(cid, ''),
                                         response=response if response is not None else _parse_response(msg.get('content')))
        self._pending_tool.append(self.types.Part(function_response=fr))

    def user(self, msg: dict[str, Any]) -> None:
        self._flush_tool()
        parts = [self.types.Part.from_text(text=x) for x in _text_items(msg.get('content'))]
        if parts:
            self._emit('user', 'user', parts)

    def add(self, msg: dict[str, Any]) -> None:
        role = msg.get('role')
        if role == 'system':
            return
        if role == 'assistant':
            self.assistant(msg)
        elif role in ('tool', 'tool_responses'):
            self.tool(msg)
        else:
            self.user(msg)

    def finish(self) -> list[Any]:
        self._flush_tool()
        return self.events


def _resp_message(call: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return (call.get('response') or {}).get('choices', [{}])[0].get('message')
    except (AttributeError, IndexError):
        return None


def logged_history(summarizer_call: dict[str, Any]) -> str | None:
    msgs = summarizer_call.get('request', {}).get('messages') or []
    if not msgs:
        return None
    text = ''.join(_text_items(msgs[0].get('content')))
    prefix = template_136().split('{conversation_history}')[0]
    return text[len(prefix):] if text.startswith(prefix) else None


def detect_agent_name(history: str) -> str | None:
    names = Counter(m.group(1) for m in re.finditer(r'(?:^|\\n)([A-Za-z_][\w-]*): ', history))
    names.pop('user', None)
    return names.most_common(1)[0][0] if names else None


def _norm_ws(s: str) -> str:
    return re.sub(r'\s+', '', s)


def reconstruct(task_dir: Path, agent_name: str | None = None) -> dict[str, Any]:
    """Rebuild the first compaction window of one task-run and check it against the logged summarizer request."""
    comps = read_jsonl(task_dir / 'compactions.jsonl')
    calls = read_jsonl(task_dir / 'llm_calls.jsonl')
    out: dict[str, Any] = {'task_dir': str(task_dir), 'valid': False, 'reason': None}
    if not comps:
        out['reason'] = 'no compaction'
        return out
    comp = comps[0]
    idx = comp.get('after_llm_call_index')
    out['after_llm_call_index'] = idx
    by_idx = {c.get('call_index'): c for c in calls}
    summ = by_idx.get(idx)
    if not summ or summ.get('kind') != 'summarizer':
        cands = [c for c in calls if c.get('kind') == 'summarizer' and (c.get('call_index') or 0) <= (idx or 0)]
        summ = cands[-1] if cands else None
    if not summ:
        out['reason'] = f'no summarizer call at/before call {idx}'
        return out
    sidx = summ['call_index']
    out['summarizer_call_index'] = sidx
    hist = logged_history(summ)
    if hist is None:
        out['reason'] = 'summarizer request does not start with the 1.36.1 prompt template'
        return out
    agents_before = [c for c in calls if c.get('kind') == 'agent' and c['call_index'] < sidx and _resp_message(c)]
    if not agents_before:
        out['reason'] = 'no agent call with a response before the summarizer'
        return out
    last = agents_before[-1]
    nxt = next((c for c in calls if c.get('kind') == 'agent' and c['call_index'] > sidx), None)
    name = agent_name or detect_agent_name(hist) or 'agent'
    b = _Builder(name)
    for m in last['request']['messages']:
        b.add(m)
    resp = _resp_message(last)
    b.assistant(resp)
    # what followed the last response: its tool results (or a nudge), all retained by construction
    tail_source = 'none'
    call_ids = [tc.get('id') for tc in resp.get('tool_calls') or []]
    next_msgs = (nxt or {}).get('request', {}).get('messages') or []
    if call_ids:
        by_id = {m.get('tool_call_id'): m for m in next_msgs if m.get('role') in ('tool', 'tool_responses')}
        tail_source = 'next_request' if all(c in by_id for c in call_ids) else 'placeholder'
        for cid in call_ids:
            m = by_id.get(cid) or {'role': 'tool', 'tool_call_id': cid}
            b.tool(m, None if cid in by_id else {'result': '<not logged: run ended after compaction>'})
    elif next_msgs:
        last_asst = max((i for i, m in enumerate(next_msgs) if m.get('role') == 'assistant'), default=None)
        if last_asst is not None:
            for m in next_msgs[last_asst + 1:]:
                if m.get('role') == 'user':
                    b.user(m)
                    tail_source = 'next_request'
    events = b.finish()
    compaction, _, _, _ = _adk()
    window = compaction._events_to_compact_for_token_threshold(events=events, event_retention_size=RETENTION)
    t_hist = format_136(window)
    out.update({
        'agent_name': name, 'last_agent_call_index': last['call_index'], 'n_events': len(events),
        'n_window': len(window), 'tail_source': tail_source,
        'logged_events_in_window': comp.get('events_in_window'), 'logged_session_events': comp.get('session_events_total'),
        'exact_match': t_hist == hist,
        'logged_summary_prompt_tokens': (summ.get('usage') or {}).get('prompt_tokens'),
        'logged_summary_completion_tokens': (summ.get('usage') or {}).get('completion_tokens'),
        'logged_summary_latency_s': summ.get('latency_s'),
        'params': {k: v for k, v in (summ.get('request', {}).get('params') or {}).items()
                   if k not in _DROP_PARAMS and v is not None},
    })
    smsg = _resp_message(summ) or {}
    out['logged_summary'] = ''.join(_text_items(smsg.get('content')))
    out['logged_summary_reasoning'] = ''.join(_text_items(smsg.get('reasoning_content') or smsg.get('reasoning')))
    if _norm_ws(t_hist) == _norm_ws(hist):
        out['valid'] = True
    else:
        a, c = _norm_ws(t_hist), _norm_ws(hist)
        i = next((k for k in range(min(len(a), len(c))) if a[k] != c[k]), min(len(a), len(c)))
        out['reason'] = (f'history mismatch at normalized char {i} (rebuilt {len(a)} vs logged {len(c)} chars): '
                         f'rebuilt …{a[max(0, i - 60):i + 60]!r} logged …{c[max(0, i - 60):i + 60]!r}')
    out['_window'] = window
    out['_hist_T'] = t_hist
    return out


# ------------------------------------------------------------------------------------------------ facts

_URL = re.compile(r'https?://\S+|www\.\S+')
_FENCE = re.compile(r'```.*?```', re.DOTALL)
_BACKTICK = re.compile(r'`([^`\n]{1,80})`')
_DOTTED = re.compile(r'(?<![\w/.-])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+(?![\w/-])')  # not part of a path
_CAPWORD = re.compile(r'\b[A-Z][A-Za-z0-9]+\b')
_PATH = re.compile(r'(?<![\w/.-])(?:[\w.-]+/)+[\w-]+\.[A-Za-z]\w*\b')
_ERRCLS = re.compile(r'\b[A-Z]\w*(?:Error|Exception|Warning)\b')
_FAILED = re.compile(r'\b(?:FAILED|ERROR)\s+(\S+::\S+)')
_SANE_PATH = re.compile(r'^[\w.-]+(?:/[\w.-]+)*$')
# an exception RAISED in run_command output: at line start (optionally module-qualified, `ValueError: x`,
# `pydantic_core._pydantic_core.ValidationError: x`), after pytest's `E   ` prefix, or after `raise`
_EXC_NAME = r'(?:[A-Za-z_][\w]*\.)*([A-Z]\w*(?:Error|Exception))\b'
_EXC_RAISED = re.compile(rf'^[ \t]*(?:E[ \t]+)?{_EXC_NAME}|\braise[ \t]+{_EXC_NAME}', re.MULTILINE)


# keywords and lowercase builtins (`None`, `yield`, `input`, `list`); builtin exception classes stay facts
_PY_NAMES = {n.lower() for n in (*keyword.kwlist, *(b for b in dir(builtins) if b[:1].islower()))}


def _dedupe(cands: list[str], limit: int, drop: set[str] = frozenset(), contained: bool = False) -> list[str]:
    """Keep the first occurrence (case-insensitive); with `contained`, also skip a candidate that is a substring
    of an already chosen fact or contains one (`get_type_hints` vs `typing.get_type_hints`)."""
    out: list[str] = []
    for c in cands:
        c = c.strip().strip('.,;:\'"')
        k = c.lower()
        if len(c) < MIN_FACT_LEN or k in drop:
            continue
        if any(k == o.lower() or (contained and (k in o.lower() or o.lower() in k)) for o in out):
            continue
        out.append(c)
        if len(out) >= limit:
            break
    return out


def _is_camel(w: str) -> bool:
    """CamelCase: starts upper, has a lowercase letter and a second uppercase letter (HTTPParser, TypeError)."""
    return any(c.islower() for c in w) and sum(c.isupper() for c in w) >= 2


def issue_facts(problem: str, repo: str = '') -> list[str]:
    text = _URL.sub(' ', problem or '')
    no_fence = _FENCE.sub(' ', text)
    cands = [m.group(1) for m in _BACKTICK.finditer(no_fence)]
    dc = sorted([(m.start(), m.group(0)) for m in _DOTTED.finditer(text)]
                + [(m.start(), m.group(0)) for m in _CAPWORD.finditer(text) if _is_camel(m.group(0))])
    cands += [s for _, s in dc]
    cands += [m.group(0) for m in _PATH.finditer(text)]
    cands += [m.group(0) for m in _ERRCLS.finditer(text)]
    drop = {p.lower() for p in re.split(r'[/_\s]', repo or '') if p} | _PY_NAMES
    return _dedupe(cands, MAX_ISSUE_FACTS, drop, contained=True)


def _strings(obj: Any, skip_keys: set[str] = frozenset()) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() if k not in skip_keys for s in _strings(v, skip_keys)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v, skip_keys)]
    return []


def _norm_path(p: str) -> str:
    p = p.strip()
    for pre in ('/workspace/', './'):
        p = p.removeprefix(pre)
    return p


def tool_facts(window: list[Any]) -> tuple[list[str], dict[str, list[str]]]:
    paths, failing, excs, edits = [], [], [], []
    for ev in window:
        for part in (ev.content.parts if ev.content and ev.content.parts else []):
            fc = part.function_call
            if fc and fc.name in ('read_file', 'edit_file', 'write_file'):
                fp = (fc.args or {}).get('filepath')
                if isinstance(fp, str) and _SANE_PATH.match(_norm_path(fp)):
                    paths.append(_norm_path(fp))
                    if fc.name == 'edit_file':
                        edits.append(_norm_path(fp))
            fr = part.function_response
            if fr:
                out = '\n'.join(_strings(fr.response, _NON_OUTPUT_KEYS))
                failing += [m.group(1).rstrip(' -') for m in _FAILED.finditer(out)]
                if fr.name == 'run_command':  # not names that only appear in source read via read_file
                    excs += [m.group(1) or m.group(2) for m in _EXC_RAISED.finditer(out)]
    cats = {'last_edit_target': edits[-1:], 'failing_tests': _dedupe(failing, 99),
            'exceptions': _dedupe(excs, 99), 'paths': _dedupe(paths, 99)}
    order = list(cats['last_edit_target'])
    rr = [list(cats['failing_tests']), list(cats['exceptions']), list(cats['paths'])]
    while any(rr):
        for lst in rr:
            if lst:
                order.append(lst.pop(0))
    return _dedupe(order, MAX_TOOL_FACTS), cats


_ALNUM = re.compile(r'[^0-9A-Za-z]+')
COVERAGE_MIN_LINE = 8


def char_coverage(window: list[Any], arm_input: str) -> dict[str, dict[str, int]]:
    """EXPLORATORY (not a frozen P-E5 measure): line-level character coverage of (a) every tool result string
    (function_response values) and (b) edit_file old_string/new_string in the window. Text is reduced to
    [0-9A-Za-z] (so repr escaping in arm Y does not matter); a line with >= 8 such chars is covered if it is a
    substring of the arm's reduced summarizer input; shares are covered chars / all chars of counted lines."""
    hay = _ALNUM.sub('', arm_input)
    out = {'tool_results': {'covered': 0, 'total': 0}, 'edit_strings': {'covered': 0, 'total': 0}}
    for ev in window:
        for part in (ev.content.parts if ev.content and ev.content.parts else []):
            texts = []
            if part.function_response:
                texts += [('tool_results', x) for x in _strings(part.function_response.response)]
            fc = part.function_call
            if fc and fc.name == 'edit_file':
                texts += [('edit_strings', (fc.args or {}).get(k)) for k in ('old_string', 'new_string')]
            for kind, txt in texts:
                for line in (txt if isinstance(txt, str) else '').splitlines():
                    a = _ALNUM.sub('', line)
                    if len(a) >= COVERAGE_MIN_LINE:
                        out[kind]['total'] += len(a)
                        out[kind]['covered'] += len(a) if a in hay else 0
    return out


def present(fact: str, text: str) -> bool:
    return fact.lower() in (text or '').lower()


# -------------------------------------------------------------------------------------------- sampling

def load_tasks(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    return {r['instance_id']: r for r in read_jsonl(path) if 'instance_id' in r}


def repo_of(iid: str, tasks: dict[str, dict[str, Any]]) -> str:
    return (tasks.get(iid) or {}).get('repo') or iid.rsplit('_', 1)[0]


def stratified_sample(by_task: dict[str, list[str]], repos: dict[str, str], n: int = N_TASKS,
                      seed: int = SEED) -> list[tuple[str, str]]:
    """Pick n distinct tasks, allocated across repos by largest remainder (each repo >= 1 while n allows), then
    one task-run per task. Deterministic for a given seed and candidate set."""
    rng = random.Random(seed)
    ids = sorted(by_task)
    if len(ids) <= n:
        chosen = ids
    else:
        groups: dict[str, list[str]] = defaultdict(list)
        for i in ids:
            groups[repos[i]].append(i)
        rs = sorted(groups)
        alloc = {r: 0 for r in rs}
        for r in sorted(rs, key=lambda r: (-len(groups[r]), r))[:n]:
            alloc[r] = 1
        rest = n - sum(alloc.values())
        if rest > 0:
            quota = {r: rest * len(groups[r]) / len(ids) for r in rs}
            for r in rs:
                alloc[r] += int(quota[r])
            left = n - sum(alloc.values())
            for r in sorted(rs, key=lambda r: (-(quota[r] - int(quota[r])), r)):
                if left <= 0:
                    break
                if alloc[r] < len(groups[r]):
                    alloc[r] += 1
                    left -= 1
        for r in rs:  # cap at group size, push overflow to the largest groups
            over = alloc[r] - len(groups[r])
            if over > 0:
                alloc[r] -= over
                for r2 in sorted(rs, key=lambda x: -len(groups[x])):
                    room = len(groups[r2]) - alloc[r2]
                    take = min(room, over)
                    alloc[r2] += take
                    over -= take
        chosen = sorted(i for r in rs for i in rng.sample(groups[r], alloc[r]))
    return [(i, rng.choice(sorted(by_task[i]))) for i in chosen]


# -------------------------------------------------------------------------------------------- prepare

def cmd_prepare(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(Path(args.tasks) if args.tasks else None)
    cands: dict[tuple[str, str], Path] = {}
    for rd in args.runs:
        rd = Path(rd)
        run = rd.name
        for td in sorted((rd / 'tasks').glob('*')):
            if (td / 'compactions.jsonl').exists() and read_jsonl(td / 'compactions.jsonl'):
                cands[(td.name, run)] = td
    print(f'task-runs with a compaction: {len(cands)} ({len({t for t, _ in cands})} distinct tasks)')
    recs: dict[tuple[str, str], dict[str, Any]] = {}
    for key, td in sorted(cands.items()):
        recs[key] = reconstruct(td, args.agent_name)
    n_valid_all = sum(r['valid'] for r in recs.values())
    print(f'validity over all first compactions: {n_valid_all}/{len(recs)} '
          f'(exact {sum(bool(r.get("exact_match")) for r in recs.values())})')
    by_task: dict[str, list[str]] = defaultdict(list)
    for t, run in cands:
        by_task[t].append(run)
    repos = {t: repo_of(t, tasks) for t in by_task}
    sample = stratified_sample(by_task, repos, args.n_tasks, args.seed)
    facts_rows, req_rows, prep_rows = [], [], []
    for t, run in sample:
        r = recs[(t, run)]
        sid = f'{t}@{run}'
        row = {k: v for k, v in r.items() if not k.startswith('_') and k not in ('logged_summary', 'logged_summary_reasoning')}
        row.update({'sample_id': sid, 'instance_id': t, 'run': run, 'repo': repos[t]})
        prep_rows.append(row)
        if not r['valid']:
            print(f'  DROP {sid}: {r["reason"]}')
            continue
        task = tasks.get(t) or {}
        problem = task.get('problem_statement')
        problem_source = 'tasks.jsonl'
        if not problem:  # fallback: the issue text as the agent saw it
            first_user = next((e for e in r['_window'] if e.author == 'user'), None)
            problem = ''.join(p.text or '' for p in (first_user.content.parts if first_user else []))
            problem_source = 'first_user_message'
        ifs = issue_facts(problem, repos[t])
        tfs, cats = tool_facts(r['_window'])
        facts_rows.append({'sample_id': sid, 'instance_id': t, 'run': run, 'repo': repos[t],
                           'issue_facts': ifs, 'tool_facts': tfs, 'tool_fact_categories': cats,
                           'problem_source': problem_source})
        hist = {'T': r['_hist_T'], 'Y': format_220(r['_window'])}
        row['exploratory_coverage'] = {arm: char_coverage(r['_window'], build_prompt(hist[arm])) for arm in ARMS}
        for arm in ARMS:
            prompt = build_prompt(hist[arm])
            req_rows.append({'sample_id': sid, 'arm': arm, 'instance_id': t, 'run': run,
                             'messages': [{'role': 'user', 'content': prompt}], 'params': r['params'],
                             'history_chars': len(hist[arm]), 'prompt_chars': len(prompt),
                             'logged_prompt_tokens_T': r.get('logged_summary_prompt_tokens'),
                             'logged_completion_tokens_T': r.get('logged_summary_completion_tokens'),
                             'logged_latency_s_T': r.get('logged_summary_latency_s'),
                             'logged_summary': r.get('logged_summary', ''),
                             'logged_summary_reasoning': r.get('logged_summary_reasoning', '')})
    # estimated input tokens: arm T's logged prompt_tokens (same text), arm Y scaled by its char ratio
    for q in req_rows:
        tq = next(x for x in req_rows if x['sample_id'] == q['sample_id'] and x['arm'] == 'T')
        lt = q.get('logged_prompt_tokens_T')
        q['est_prompt_tokens'] = (round(lt * q['prompt_chars'] / tq['prompt_chars']) if lt
                                  else round(q['prompt_chars'] / 3.5))
    write_jsonl(out / 'facts.jsonl', facts_rows)
    write_jsonl(out / 'requests.jsonl', req_rows)
    n_valid = sum(r['valid'] for r in prep_rows)
    est = {arm: [q['est_prompt_tokens'] for q in req_rows if q['arm'] == arm] for arm in ARMS}
    summary = {
        'seed': args.seed, 'n_tasks_target': args.n_tasks, 'runs': [str(x) for x in args.runs],
        'candidates': len(cands), 'distinct_tasks': len(by_task),
        'validity_all': {'valid': n_valid_all, 'total': len(recs),
                         'invalid': {f'{t}@{run}': r['reason'] for (t, run), r in recs.items() if not r['valid']}},
        'validity_sample': {'valid': n_valid, 'total': len(prep_rows)},
        'sample': prep_rows,
        'fact_counts': {'issue': sum(len(f['issue_facts']) for f in facts_rows),
                        'tool': sum(len(f['tool_facts']) for f in facts_rows)},
        'est_prompt_tokens': {arm: {'mean': round(sum(v) / len(v)) if v else None, 'max': max(v, default=None),
                                    'total': sum(v)} for arm, v in est.items()},
        'repos': dict(Counter(r['repo'] for r in prep_rows)),
    }
    (out / 'prepare.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(f'sample: {len(prep_rows)} task-runs over repos {summary["repos"]}')
    print(f'VALIDITY (sample): {n_valid}/{len(prep_rows)} match the logged summarizer history (whitespace-insensitive); '
          f'exact {sum(bool(r.get("exact_match")) for r in prep_rows)}')
    print(f'facts: issue {summary["fact_counts"]["issue"]}, tool {summary["fact_counts"]["tool"]} '
          f'over {len(facts_rows)} samples')
    for arm in ARMS:
        e = summary['est_prompt_tokens'][arm]
        print(f'est. input tokens arm {arm}: mean {e["mean"]}, max {e["max"]}, total {e["total"]}')
    print(f'wrote {out}/facts.jsonl, {out}/requests.jsonl ({len(req_rows)} requests), {out}/prepare.json')
    return 0


# ------------------------------------------------------------------------------------------------- run

def _base(endpoint: str) -> str:
    e = endpoint.rstrip('/')
    return e if e.endswith('/v1') else e + '/v1'


def _http(url: str, body: dict[str, Any] | None, key: str | None, timeout: float) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method='POST' if data is not None else 'GET')
    req.add_header('Content-Type', 'application/json')
    if key:
        req.add_header('Authorization', f'Bearer {key}')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def request_body(model: str, req: dict[str, Any]) -> dict[str, Any]:
    params = dict(req.get('params') or {})
    extra = params.pop('extra_body', None) or {}
    body = {'model': model, 'messages': req['messages']}
    body.update({k: v for k, v in params.items() if k not in _DROP_PARAMS and v is not None})
    body.update(extra)  # the OpenAI client merges extra_body into the JSON body
    return body


def _call_key(r: dict[str, Any]) -> tuple[str, str, int]:
    return (r['sample_id'], r['arm'], int(r['k']))


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    reqs = read_jsonl(out / 'requests.jsonl')
    if not reqs:
        print(f'no {out}/requests.jsonl: run prepare first', file=sys.stderr)
        return 2
    key = os.environ.get('VLLM_API_KEY') or None
    base = _base(args.endpoint)
    model = args.model
    if not model:
        models = _http(base + '/models', None, key, 60).get('data') or []
        if not models:
            print('no model served at /v1/models', file=sys.stderr)
            return 2
        model = models[0]['id']
    print(f'endpoint {base}, model {model}, auth {"yes" if key else "no"}')
    done = {_call_key(r) for r in read_jsonl(out / 'calls.jsonl') if not r.get('error')}
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for q in reqs:
        by_sample[q['sample_id']][q['arm']] = q
    todo = [(sid, arm, k) for sid in by_sample for k in range(args.samples) for arm in ARMS
            if arm in by_sample[sid] and (sid, arm, k) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f'{len(done)} calls already done, {len(todo)} to go')
    n_err = 0
    with open(out / 'calls.jsonl', 'a', encoding='utf-8') as fh:
        for i, (sid, arm, k) in enumerate(todo, 1):
            q = by_sample[sid][arm]
            body = request_body(model, q)
            rec: dict[str, Any] = {'sample_id': sid, 'arm': arm, 'k': k, 'model': model, 't_start': time.time()}
            t0 = time.perf_counter()
            try:
                resp = _http(base + '/chat/completions', body, key, args.timeout)
                msg = (resp.get('choices') or [{}])[0].get('message') or {}
                rec.update({
                    'content': msg.get('content') or '',
                    'reasoning_content': msg.get('reasoning_content') or msg.get('reasoning') or '',
                    'finish_reason': (resp.get('choices') or [{}])[0].get('finish_reason'),
                    'usage': resp.get('usage'),
                })
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                detail = ''
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        detail = ' ' + e.read().decode()[:300]
                    except Exception:
                        pass
                rec['error'] = f'{type(e).__name__}: {e}{detail}'
                n_err += 1
            rec['latency_s'] = round(time.perf_counter() - t0, 3)
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fh.flush()
            u = rec.get('usage') or {}
            print(f'[{i}/{len(todo)}] {sid} {arm} k={k}: {rec["latency_s"]}s '
                  f'in={u.get("prompt_tokens")} out={u.get("completion_tokens")}'
                  + (f' ERROR {rec["error"][:200]}' if rec.get('error') else ''), flush=True)
    print(f'done; {n_err} errors (re-run the same command to retry them)')
    return 1 if n_err else 0


# ---------------------------------------------------------------------------------------------- report

def _ratio(x: int, n: int) -> dict[str, Any]:
    return {'x': x, 'n': n, 'pct': round(100 * x / n, 1) if n else None}


def _mean(v: list[float]) -> float | None:
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 1) if v else None


def compute(out: Path) -> dict[str, Any]:
    facts = {f['sample_id']: f for f in read_jsonl(out / 'facts.jsonl')}
    reqs = {(q['sample_id'], q['arm']): q for q in read_jsonl(out / 'requests.jsonl')}
    calls = [c for c in read_jsonl(out / 'calls.jsonl') if not c.get('error')]
    latest: dict[tuple[str, str, int], dict[str, Any]] = {}
    for c in calls:
        latest[_call_key(c)] = c
    calls = list(latest.values())
    prep = json.loads((out / 'prepare.json').read_text()) if (out / 'prepare.json').exists() else {}
    res: dict[str, Any] = {'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                           'validity_sample': prep.get('validity_sample'), 'validity_all': prep.get('validity_all'),
                           'n_samples': len(facts), 'fact_counts': prep.get('fact_counts'),
                           'est_prompt_tokens': prep.get('est_prompt_tokens'), 'arms': {}, 'per_sample': {}}

    def score(arm_inputs: dict[str, str], summaries: list[tuple[str, str, str]]) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for kind in ('issue', 'tool'):
            n = avail = 0
            for sid, f in facts.items():
                for fact in f[f'{kind}_facts']:
                    n += 1
                    avail += present(fact, arm_inputs.get(sid, ''))
            ret = ret_av = ret_incl = n_calls_facts = n_av_calls = unavail_ret = 0
            for sid, summ, reasoning in summaries:
                f = facts.get(sid)
                if not f:
                    continue
                for fact in f[f'{kind}_facts']:
                    a = present(fact, arm_inputs.get(sid, ''))
                    r = present(fact, summ)
                    n_calls_facts += 1
                    n_av_calls += a
                    ret += r
                    ret_av += r and a
                    unavail_ret += r and not a
                    ret_incl += r or present(fact, reasoning)
            d[kind] = {'available': _ratio(avail, n), 'retained': _ratio(ret, n_calls_facts),
                       'retained_given_available': _ratio(ret_av, n_av_calls),
                       'retained_unavailable': unavail_ret,
                       'retained_incl_reasoning': _ratio(ret_incl, n_calls_facts)}
        return d

    for arm in ARMS:
        inputs = {sid: reqs[(sid, arm)]['messages'][0]['content'] for sid in facts if (sid, arm) in reqs}
        ac = [c for c in calls if c['arm'] == arm and c['sample_id'] in facts]
        d = score(inputs, [(c['sample_id'], c.get('content', ''), c.get('reasoning_content', '')) for c in ac])
        d.update({
            'desc': ARM_DESC[arm], 'calls': len(ac),
            'prompt_tokens_mean': _mean([(c.get('usage') or {}).get('prompt_tokens') for c in ac]),
            'completion_tokens_mean': _mean([(c.get('usage') or {}).get('completion_tokens') for c in ac]),
            'latency_s_mean': _mean([c.get('latency_s') for c in ac]),
            'finish_reasons': dict(Counter(c.get('finish_reason') for c in ac)),
            'history_chars_mean': _mean([reqs[(sid, arm)]['history_chars'] for sid in facts if (sid, arm) in reqs]),
            'est_prompt_tokens_mean': _mean([reqs[(sid, arm)].get('est_prompt_tokens') for sid in facts if (sid, arm) in reqs]),
        })
        res['arms'][arm] = d
    # reference: arm T as it actually ran in ab1 (A100, one sample, the logged summary)
    inputs_t = {sid: reqs[(sid, 'T')]['messages'][0]['content'] for sid in facts if (sid, 'T') in reqs}
    logged = [(sid, reqs[(sid, 'T')].get('logged_summary', ''), reqs[(sid, 'T')].get('logged_summary_reasoning', ''))
              for sid in facts if (sid, 'T') in reqs]
    ref = score(inputs_t, logged)
    ref.update({'desc': 'arm T as logged in ab1 (A100, scorer flags, 1 sample/window)', 'calls': len(logged),
                **{f'{m}_mean': _mean([reqs[(sid, 'T')].get(f'logged_{m}_T') for sid in facts if (sid, 'T') in reqs])
                   for m in ('prompt_tokens', 'completion_tokens', 'latency_s')}})
    res['arms']['T_logged_ab1'] = ref
    for sid, f in facts.items():
        ps = {'issue_facts': f['issue_facts'], 'tool_facts': f['tool_facts']}
        for arm in ARMS:
            q = reqs.get((sid, arm))
            if not q:
                continue
            ac = sorted([c for c in calls if c['sample_id'] == sid and c['arm'] == arm], key=lambda c: c['k'])
            inp = q['messages'][0]['content']
            ps[arm] = {
                'available': {kind: [x for x in f[f'{kind}_facts'] if present(x, inp)] for kind in ('issue', 'tool')},
                'retained': [{kind: [x for x in f[f'{kind}_facts'] if present(x, c.get('content', ''))]
                              for kind in ('issue', 'tool')} for c in ac],
                'summaries': [c.get('content', '') for c in ac],
                'prompt_tokens': [(c.get('usage') or {}).get('prompt_tokens') for c in ac],
            }
        res['per_sample'][sid] = ps
    cov: dict[str, Any] = {'label': 'EXPLORATORY/descriptive, not a frozen P-E5 measure', 'method': char_coverage.__doc__}
    rows = [x for x in prep.get('sample', []) if x.get('exploratory_coverage')]
    for arm in ARMS:
        for kind in ('tool_results', 'edit_strings'):
            c = sum(x['exploratory_coverage'][arm][kind]['covered'] for x in rows)
            n = sum(x['exploratory_coverage'][arm][kind]['total'] for x in rows)
            cov.setdefault(arm, {})[kind] = _ratio(c, n)
    res['exploratory_char_coverage'] = cov
    return res


def _fmt(r: dict[str, Any] | None) -> str:
    if not r or not r.get('n'):
        return '–'
    return f'{r["x"]}/{r["n"]} ({r["pct"]:.0f}%)'


def _short(s: str, n: int = 600) -> str:
    s = re.sub(r'\s+', ' ', s or '').strip()
    return s if len(s) <= n else s[:n].rstrip() + ' …'


def render_markdown(res: dict[str, Any]) -> str:
    L = ['<!-- GENERATED by `python -m harness.analysis.pe5 report`; do not edit by hand, re-run the report. -->',
         '# P-E5 results: serializer replay (GENERATED DRAFT)', '',
         ('> **Generated draft.** Numbers come from `runs/pe5/{facts,requests,calls}.jsonl`; '
         'spec: docs/pe5.md. Review before quoting in the paper.'), '',
         f'Generated {res["generated_at"]}.', '']
    va, vs = res.get('validity_all') or {}, res.get('validity_sample') or {}
    L += ['## Validity check', '',
          (f'- Sampled windows whose rebuilt 1.36.1 serialization matches the logged summarizer request '
          f'(whitespace-insensitive): **{vs.get("valid")}/{vs.get("total")}**.'),
          f'- Over all first compactions in the input runs: {va.get("valid")}/{va.get("total")}.']
    for k, why in (va.get('invalid') or {}).items():
        L.append(f'  - dropped `{k}`: {why[:300]}')
    fc = res.get('fact_counts') or {}
    L += ['', (f'Samples: {res["n_samples"]}; facts frozen before the run: {fc.get("issue")} issue, '
          f'{fc.get("tool")} tool.'), '', '## Per arm', '',
          ('Availability = fact appears in the summarizer input; retention = fact appears in the summary (final '
          'content, not reasoning), pooled over calls. x/available counts only facts present in that arm\'s input.'), '',
          ('| Arm | Calls | Issue avail. | Issue retained | Issue ret./avail. | Tool avail. | Tool retained | '
          'Tool ret./avail. | In tok (mean) | Out tok (mean) | Latency s |'),
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for arm in (*ARMS, 'T_logged_ab1'):
        d = res['arms'].get(arm)
        if not d:
            continue
        name = {'T': 'T (1.36.1 text-only)', 'Y': 'Y (2.2.0 typed)', 'T_logged_ab1': 'T, ab1 logged (ref.)'}[arm]
        L.append(f'| {name} | {d["calls"]} | {_fmt(d["issue"]["available"])} | {_fmt(d["issue"]["retained"])} | '
                 f'{_fmt(d["issue"]["retained_given_available"])} | {_fmt(d["tool"]["available"])} | '
                 f'{_fmt(d["tool"]["retained"])} | {_fmt(d["tool"]["retained_given_available"])} | '
                 f'{d.get("prompt_tokens_mean") or "–"} | {d.get("completion_tokens_mean") or "–"} | '
                 f'{d.get("latency_s_mean") or "–"} |')
    est = res.get('est_prompt_tokens') or {}
    L += ['', 'Estimated input tokens before the run (arm T = logged prompt_tokens; arm Y scaled by prompt length): '
          + ', '.join(f'{a}: mean {e.get("mean")}, max {e.get("max")}' for a, e in est.items()) + '.', '']
    cov = res.get('exploratory_char_coverage') or {}
    if cov.get('T'):
        L += ['**EXPLORATORY (descriptive, not a frozen measure):** share of the window\'s characters that reach the '
              'summarizer input (line-level char coverage on alphanumerics, lines >= 8 chars, pooled over samples): '
              + '; '.join(f'{a}: tool results {_fmt(cov[a]["tool_results"])}, edit_file old/new strings '
                          f'{_fmt(cov[a]["edit_strings"])}' for a in ARMS if a in cov) + '.', '']
    for arm in ARMS:
        d = res['arms'].get(arm) or {}
        if d.get('calls'):
            L.append(f'- Arm {arm}: facts retained although absent from the input: issue '
                     f'{d["issue"]["retained_unavailable"]}, tool {d["tool"]["retained_unavailable"]}; retained incl. '
                     f'reasoning: issue {_fmt(d["issue"]["retained_incl_reasoning"])}, tool '
                     f'{_fmt(d["tool"]["retained_incl_reasoning"])}; finish reasons {d.get("finish_reasons")}.')
    # examples: largest tool-retention gap Y - T on the first sample
    ex = []
    for sid, ps in res['per_sample'].items():
        if ps.get('T', {}).get('summaries') and ps.get('Y', {}).get('summaries'):
            gap = len(ps['Y']['retained'][0]['tool']) - len(ps['T']['retained'][0]['tool'])
            ex.append((gap, sid))
    ex.sort(key=lambda x: (-x[0], x[1]))
    if ex:
        L += ['', '## Example summary pairs (shortened, first sample per arm)', '']
        for _, sid in ex[:3]:
            ps = res['per_sample'][sid]
            L += [f'### `{sid}`', '', f'- Issue facts: {", ".join(f"`{x}`" for x in ps["issue_facts"]) or "–"}',
                  f'- Tool facts: {", ".join(f"`{x}`" for x in ps["tool_facts"]) or "–"}']
            for arm in ARMS:
                p = ps[arm]
                kept = p['retained'][0]
                L += [(f'- **{arm}** (tool facts available {len(p["available"]["tool"])}/{len(ps["tool_facts"])}, '
                      f'retained issue {len(kept["issue"])}/{len(ps["issue_facts"])}, tool '
                      f'{len(kept["tool"])}/{len(ps["tool_facts"])}): {_short(p["summaries"][0])}')]
            L.append('')
    else:
        L += ['', '_No completed calls yet: run `python -m harness.analysis.pe5 run` first._', '']
    return '\n'.join(L) + '\n'


def cmd_report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    res = compute(out)
    (out / 'results.json').write_text(json.dumps(res, indent=2, ensure_ascii=False) + '\n')
    doc = Path(args.doc) if args.doc else out / 'pe5-results.md'
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(render_markdown(res), encoding='utf-8')
    for arm, d in res['arms'].items():
        print(f'{arm}: calls {d["calls"]}, issue retained {_fmt(d["issue"]["retained"])}, '
              f'tool avail {_fmt(d["tool"]["available"])}, tool retained {_fmt(d["tool"]["retained"])}, '
              f'in tok {d.get("prompt_tokens_mean")}')
    print(f'wrote {out}/results.json and {doc}')
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare', help='rebuild windows, validity check, freeze facts, write requests')
    p.add_argument('--runs', nargs='+', required=True, help='runner output dirs (runs/<name>, holding tasks/<id>/)')
    p.add_argument('--tasks', default='data/kaggle/tasks.jsonl', help='tasks.jsonl with problem_statement and repo')
    p.add_argument('--out', default='runs/pe5')
    p.add_argument('--n-tasks', type=int, default=N_TASKS)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--agent-name', help='event author of the agent (default: detected from the logged history)')
    p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser('run', help='send the requests (sequential, resumable)')
    p.add_argument('--endpoint', required=True, help='OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1')
    p.add_argument('--out', default='runs/pe5')
    p.add_argument('--samples', type=int, default=2, help='samples per arm per window')
    p.add_argument('--model', help='model id (default: first id from /v1/models)')
    p.add_argument('--timeout', type=float, default=900.0)
    p.add_argument('--limit', type=int, default=0, help='send at most N calls this invocation (smoke test)')
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser('report', help='results.json and the generated markdown draft')
    p.add_argument('--out', default='runs/pe5')
    p.add_argument('--doc', default=str(DEFAULT_DOC) if DEFAULT_DOC else None,
                   help='markdown draft path (default: <out>/pe5-results.md)')
    p.set_defaults(fn=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
