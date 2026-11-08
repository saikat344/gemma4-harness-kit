"""Reliability analyzer for runner output: how often does the agent fail mechanically, and does it matter?

    python -m harness.analysis.reliability runs/ab1/v0 runs/ab1/v1 [--grades FILE] [--out runs/ab1/reliability]

Reads runs/<name>/ directories written by harness/runner (docs/runner.md section 5) plus their
grades.jsonl, and reports per task and in aggregate (pooled and per run):

  identical consecutive tool calls (esp. repeated after an error), calls ADK rejected (missing/invalid
  args, unknown tool) and parser-error signatures, edit_file failures by cause, A-B-A edit oscillation,
  runaway generations (output-token tail, calls at max_output_tokens), time spent in failed/retried
  turns, empty patch, compactions and whether the problem statement survived them.

Every failure mode is a per-task boolean flag (thresholds are CLI options). Each flag is then correlated with
resolved/unresolved: tasks flagged, resolve rate flagged vs not, share of unresolved tasks affected, Fisher p.
Definitions: docs/reliability.md section 2.

Primary source is tasks/<id>/llm_calls.jsonl (what the model emitted and what ADK returned to it, crash-safe,
survives killed workers); trace_legacy.json is only the fallback when llm_calls.jsonl is missing.
Outputs: markdown (stdout, or <out>.md / <out>/reliability.md) and a JSON summary (<out>.json).
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------------------------------


@dataclass
class Thresholds:
    loop_min_run: int = 3  # a run of >= N identical consecutive calls = retry_loop
    aba_sim: float = 0.8  # difflib ratio for "same change" (A-B-A)
    aba_min_reverts: int = 2  # reverts needed for aba_oscillation (>=1 is aba_revert_any)
    aba_min_chars: int = 3  # ignore changes smaller than this (chars) when looking for reverts
    runaway_frac: float = 0.9  # a call with completion_tokens >= frac * max_output_tokens "hit the cap"
    tail_tokens: int = 4096  # a call with >= N completion tokens is in the "long tail"
    default_cap: int = 8192  # max_output_tokens if the request params do not say
    edit_fail_heavy_n: int = 3  # edit_file failures >= N ...
    edit_fail_heavy_rate: float = 0.5  # ... or rate >= this (with >= 4 edit_file calls)
    waste_frac: float = 0.2  # failed+retried turns >= frac of agent time = time_wasted_high


# ---------------------------------------------------------------------------------------------------
# tool schema
# ---------------------------------------------------------------------------------------------------

DEFAULT_TOOL_PARAMS: dict[str, tuple[list[str], list[str]]] = {  # name -> (all params, required)
    'edit_file': (['filepath', 'old_string', 'new_string', 'allow_multiple'], ['filepath', 'old_string', 'new_string']),
    'write_file': (['filepath', 'content'], ['filepath', 'content']),
    'read_file': (['filepath', 'start_line', 'end_line'], ['filepath']),
    'run_command': (['command'], ['command']),
    'submit_patch': ([], []),
    'get_status': ([], []),
}
EDIT_TOOLS = ('edit_file', 'write_file')
REJECTED = ('rejected_missing', 'rejected_unknown_tool', 'rejected_other')
# kinds that mean "the harness/tool did not do what the model asked" (a failing pytest run is NOT one)
FAILED = REJECTED + ('edit_fail', 'write_fail', 'budget', 'timeout', 'other_error')
SPECIAL_TOKENS = ('<|"|>', '<tool_call|>', '<|tool_call>', '<|channel>', '<channel|>', '<|tool_response>', '<turn|>')
REVERT_CMD = re.compile(r'git\s+(checkout\s+(--\s|\.|HEAD)|restore|stash|reset\s+--hard|revert|apply\s+(-R|--reverse))'
                        r'|patch\s+(-R|--reverse)')
PROBLEM_MARKER = 'Problem Statement:\n'


# ---------------------------------------------------------------------------------------------------
# result classification
# ---------------------------------------------------------------------------------------------------


def parse_json(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except Exception:
        return text


def edit_cause(msg: str) -> str:
    m = msg.lower()
    if 'old_string not found' in m or 'failed to replace' in m:
        return 'old_string_not_found'
    if 'occurs' in m and 'times' in m or 'expected 1 occurrence' in m or 'flexibly matches' in m:
        return 'multiple_matches'
    if 'old_string cannot be empty' in m:
        return 'empty_old_string'
    if 'no changes to apply' in m:
        return 'identical_old_new'
    if 'is empty. use write_file' in m:
        return 'file_empty'
    if 'path traversal' in m or 'invalid file path' in m:
        return 'bad_path'
    if 'no such file' in m or 'not found in workspace' in m or 'filenotfound' in m or 'could not find' in m:
        return 'file_not_found'
    return 'other'


def classify_result(tool: str, text: str | None) -> tuple[str, str, str, str | None]:
    """-> (kind, error_type, message, cause). kinds: ok, rejected_*, edit_fail, write_fail, cmd_error,
    timeout, budget, other_error, no_result."""
    if text is None:
        return 'no_result', '', '', None
    obj = parse_json(text)
    if isinstance(obj, dict):
        inner = obj.get('result') if isinstance(obj.get('result'), dict) else obj
        status = inner.get('status')
        if 'error' in inner and status is None:  # ADK FunctionTool short-circuit: {"error": "..."}
            msg = str(inner['error'])
            if 'mandatory input parameters' in msg:
                return 'rejected_missing', 'MissingArguments', msg, None
            if 'not found' in msg and 'Available tools' in msg:
                return 'rejected_unknown_tool', 'UnknownTool', msg, None
            return 'rejected_other', 'ToolInvocationError', msg, None
        if status == 'error':
            et = str(inner.get('error_type') or inner.get('error_code') or 'Error')
            msg = str(inner.get('error_message') or inner.get('error') or '')
            if et == 'ValueError' and 'not found' in msg and 'Available tools' in msg:
                return 'rejected_unknown_tool', 'UnknownTool', msg, None
            if et == 'INVALID_ARGUMENTS' or inner.get('error_code') == 'INVALID_ARGUMENTS':
                return 'rejected_other', 'InvalidArguments', msg, None
            if et == 'BudgetExceeded':
                return 'budget', et, msg, None
            if et == 'TimeoutExceeded':
                return 'timeout', et, msg, None
            if et == 'FileEditError' or tool == 'edit_file':
                return 'edit_fail', et, msg, edit_cause(msg)
            if tool == 'write_file':
                return 'write_fail', et, msg, None
            if et == 'CommandError':
                return 'cmd_error', et, msg[:300], None
            if et in ('TypeError', 'ValidationError', 'KeyError'):
                return 'rejected_other', et, msg, None
            return 'other_error', et, msg, None
        return 'ok', '', '', None
    if isinstance(obj, str) and obj.lower().startswith('error'):
        return 'other_error', 'Error', obj[:300], None
    return 'ok', '', '', None


def canon_args(args: Any) -> str:
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return repr(args)


# ---------------------------------------------------------------------------------------------------
# loading one task
# ---------------------------------------------------------------------------------------------------


@dataclass
class Call:
    turn: int
    tool: str
    args: Any
    raw_args: str
    call_id: str | None
    result: str | None = None
    kind: str = 'no_result'
    err_type: str = ''
    err_msg: str = ''
    cause: str | None = None
    sig: str = ''
    signatures: list[str] = field(default_factory=list)  # parser/format signatures on the call itself
    llm_index: int | None = None

    @property
    def failed(self) -> bool:
        return self.kind in FAILED

    @property
    def is_error(self) -> bool:
        return self.kind not in ('ok', 'no_result')


@dataclass
class Turn:
    index: int
    llm_index: int | None
    calls: list[Call]
    latency_s: float = 0.0
    gap_s: float = 0.0
    completion_tokens: int = 0
    finish_reason: str | None = None

    @property
    def total_s(self) -> float:
        return self.latency_s + self.gap_s


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def tool_schema_from_request(tools: Any) -> dict[str, tuple[list[str], list[str]]]:
    """OpenAI `tools` -> {name: (all params, required)}; falls back to DEFAULT_TOOL_PARAMS per name."""
    out: dict[str, tuple[list[str], list[str]]] = {}
    for t in tools or []:
        fn = (t or {}).get('function') or {}
        name = fn.get('name')
        if not name:
            continue
        params = fn.get('parameters') or {}
        out[name] = (list((params.get('properties') or {}).keys()), list(params.get('required') or []))
    return out


def _msg_text(m: dict[str, Any]) -> str:
    c = m.get('content')
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return ''.join(p.get('text', '') for p in c if isinstance(p, dict))
    return ''


def call_signatures(c: Call, schema: dict[str, tuple[list[str], list[str]]]) -> list[str]:
    """Parser/format signatures visible in the call as the model's server returned it."""
    sigs: list[str] = []
    allp, req = schema.get(c.tool) or DEFAULT_TOOL_PARAMS.get(c.tool, (None, None))
    if c.args is None:
        return ['invalid_json_arguments']
    if not isinstance(c.args, dict):
        return ['non_object_arguments']
    for k in c.args:
        if '<|' in k or '|>' in k or '"' in k or ',' in k or ':' in k or ' ' in k:
            sigs.append('mangled_key')
            break
    if allp is not None:
        unknown = [k for k in c.args if k not in allp]
        if unknown and 'mangled_key' not in sigs:
            sigs.append('unknown_key')
        if any(r not in c.args for r in req):
            sigs.append('missing_required')
    for v in c.args.values():
        if isinstance(v, str) and any(t in v for t in SPECIAL_TOKENS):
            sigs.append('special_token_in_value')
            break
    return sigs


def attach_results(calls: list[Call], later: list[dict[str, Any]]) -> None:
    """Fill Call.result from the tool messages of the next agent request(s): by tool_call_id, else by position
    (the tool messages that follow the last assistant message carrying this many tool calls)."""
    for r2 in later:
        msgs = (r2.get('request') or {}).get('messages') or []
        by_id = {m.get('tool_call_id'): m for m in msgs if m.get('role') == 'tool' and m.get('tool_call_id')}
        for c in calls:
            if c.result is None and c.call_id in by_id:
                c.result = _msg_text(by_id[c.call_id])
        if all(c.result is not None for c in calls):
            return
    if later and any(c.result is None for c in calls):
        msgs = (later[0].get('request') or {}).get('messages') or []
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if m.get('role') == 'assistant' and len(m.get('tool_calls') or []) == len(calls):
                following = [x for x in msgs[i + 1:] if x.get('role') == 'tool'][: len(calls)]
                if len(following) == len(calls):
                    for c, x in zip(calls, following):
                        if c.result is None:
                            c.result = _msg_text(x)
                break


def load_task_calls(task_dir: Path) -> dict[str, Any]:
    """Build turns/calls/LLM-call stats for one task from llm_calls.jsonl (fallback: trace_legacy.json)."""
    recs = read_jsonl(task_dir / 'llm_calls.jsonl')
    out: dict[str, Any] = {'source': 'llm_calls', 'turns': [], 'agent_calls': [], 'llm_errors': 0, 'recs': recs,
                           'unparsed_text': 0, 'truncated_tool_text': 0, 'schema': {}, 'cap': None}
    if not recs:
        return _from_trace(task_dir, out)
    agent = [r for r in recs if r.get('kind') != 'summarizer']
    for r in recs:
        if r.get('error'):
            out['llm_errors'] += 1
    ok_agent = [r for r in agent if not r.get('error') and r.get('response')]
    out['agent_calls'] = ok_agent
    # tool schema + output cap from the first request
    if ok_agent:
        req0 = ok_agent[0].get('request') or {}
        out['schema'] = tool_schema_from_request(req0.get('tools'))
        params = req0.get('params') or {}
        eb = params.get('extra_body') or {}
        out['cap'] = (params.get('max_completion_tokens') or params.get('max_tokens') or
                      eb.get('max_completion_tokens') or eb.get('max_tokens'))
    turns: list[Turn] = []
    tool_turn_no = 0
    all_by_pos = recs
    pos_of = {id(r): i for i, r in enumerate(all_by_pos)}
    for ai, r in enumerate(ok_agent):
        resp = r.get('response') or {}
        choice = (resp.get('choices') or [{}])[0]
        msg = choice.get('message') or {}
        usage = r.get('usage') or resp.get('usage') or {}
        content = msg.get('content') or ''
        if isinstance(content, str) and '<|tool_call>' in content:
            out['unparsed_text'] += 1
            if r.get('finish_reason') in ('length', 'max_tokens'):
                out['truncated_tool_text'] += 1
        tcs = msg.get('tool_calls') or []
        calls: list[Call] = []
        for tc in tcs:
            fn = tc.get('function') or {}
            raw = fn.get('arguments')
            raw_s = raw if isinstance(raw, str) else json.dumps(raw)
            try:
                args = json.loads(raw_s) if raw_s not in (None, '') else {}
            except Exception:
                args = None
            c = Call(turn=tool_turn_no, tool=str(fn.get('name') or '?'), args=args, raw_args=raw_s,
                     call_id=tc.get('id'), llm_index=r.get('call_index'))
            c.sig = f'{c.tool}:{canon_args(args) if args is not None else raw_s}'
            c.signatures = call_signatures(c, out['schema'])
            calls.append(c)
        # gap to the next record of any kind
        gap = 0.0
        i = pos_of[id(r)]
        if i + 1 < len(all_by_pos):
            nxt = all_by_pos[i + 1]
            gap = max(0.0, float(nxt.get('t_start', 0) or 0) - (float(r.get('t_start', 0) or 0)
                                                                 + float(r.get('latency_s', 0) or 0)))
        if calls:
            t = Turn(index=tool_turn_no, llm_index=r.get('call_index'), calls=calls,
                     latency_s=float(r.get('latency_s') or 0), gap_s=gap,
                     completion_tokens=int(usage.get('completion_tokens') or 0), finish_reason=r.get('finish_reason'))
            turns.append(t)
            tool_turn_no += 1
            attach_results(calls, ok_agent[ai + 1: ai + 3])
    out['turns'] = turns
    fill_from_trace(task_dir, [c for t in turns for c in t.calls])
    for t in turns:
        for c in t.calls:
            c.kind, c.err_type, c.err_msg, c.cause = classify_result(c.tool, c.result)
            if c.result is None:  # never fed back to the model (last turn / killed worker): infer from the call itself
                if c.tool not in out['schema'] and c.tool not in DEFAULT_TOOL_PARAMS:
                    c.kind, c.err_type = 'rejected_unknown_tool', 'inferred'
                elif 'missing_required' in c.signatures:
                    c.kind, c.err_type = 'rejected_missing', 'inferred'
    return out


def fill_from_trace(task_dir: Path, calls: list[Call]) -> None:
    """Results the model never saw again (last turn) are still in trace_legacy.json. `calls` is every call of the
    task in order (observed ones included). Match by stable call id when both sides carry one; otherwise the k-th
    call with a given tool + args is the k-th trace call with that tool + args, so repeated identical calls never
    inherit an earlier call's outcome (audit F13)."""
    if not any(c.result is None for c in calls):
        return
    p = task_dir / 'trace_legacy.json'
    if not p.exists():
        return
    try:
        entries = json.loads(p.read_text(encoding='utf-8')).get('entries', [])
    except Exception:
        return

    def eid(e: dict[str, Any]) -> str | None:
        v = e.get('call_id', e.get('id'))
        return v if isinstance(v, str) and v else None

    # pair each trace response with its call: by id when present, else the oldest pending call of that tool
    pending: list[tuple[str, str, str | None]] = []
    pairs: list[tuple[str, str, str | None, Any]] = []  # (tool, args, id, result), in call order of completion
    for e in entries:
        if not isinstance(e, dict):
            continue
        if e.get('type') == 'tool_call':
            pending.append((str(e.get('tool')), canon_args(e.get('args') if isinstance(e.get('args'), dict) else None),
                            eid(e)))
        elif e.get('type') == 'tool_response':
            tool, rid = str(e.get('tool')), eid(e)
            idx = next((i for i, (_t, _a, cid) in enumerate(pending) if rid and cid == rid), None)
            if idx is None:
                idx = next((i for i, (t, _a, _c) in enumerate(pending) if t == tool), None)
            if idx is not None:
                t, a, cid = pending.pop(idx)
                pairs.append((t, a, cid, e.get('result')))
    by_id = {cid: res for _t, _a, cid, res in pairs if cid}
    by_key: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for t, a, _cid, res in pairs:
        by_key[(t, a)].append(res)
    seen: Counter = Counter()
    for c in calls:
        key = (c.tool, canon_args(c.args))
        k = seen[key]
        seen[key] += 1
        if c.result is not None:
            continue
        if c.call_id and c.call_id in by_id:
            c.result = by_id[c.call_id]
        elif k < len(by_key.get(key, ())):
            c.result = by_key[key][k]


def _from_trace(task_dir: Path, out: dict[str, Any]) -> dict[str, Any]:
    """Fallback when there is no llm_calls.jsonl: calls and results from trace_legacy.json (no tokens)."""
    p = task_dir / 'trace_legacy.json'
    if not p.exists():
        return out
    try:
        entries = json.loads(p.read_text(encoding='utf-8')).get('entries', [])
    except Exception:
        return out
    out['source'] = 'trace_legacy'
    turns: list[Turn] = []
    pending: list[Call] = []
    cur: Turn | None = None
    last_resp_t = 0.0
    in_call_run = False
    for e in entries:
        t = e.get('type')
        el = float(e.get('elapsed_s', e.get('elapsed', 0)) or 0)
        if t == 'tool_call':
            args = e.get('args') if isinstance(e.get('args'), dict) else None
            if not in_call_run or cur is None:
                cur = Turn(index=len(turns), llm_index=None, calls=[], latency_s=max(0.0, el - last_resp_t))
                turns.append(cur)
            in_call_run = True
            c = Call(turn=cur.index, tool=str(e.get('tool', '?')), args=args, raw_args=json.dumps(args),
                     call_id=None)
            c.sig = f'{c.tool}:{canon_args(args)}'
            c.signatures = call_signatures(c, {})
            cur.calls.append(c)
            pending.append(c)
        elif t == 'tool_response':
            in_call_run = False
            for i, c in enumerate(pending):
                if c.tool == e.get('tool'):
                    c.result = e.get('result')
                    c.kind, c.err_type, c.err_msg, c.cause = classify_result(c.tool, c.result)
                    pending.pop(i)
                    break
            last_resp_t = el
        elif t in ('continuation_nudge', 'task_prompt', 'compaction'):
            in_call_run = False
            last_resp_t = el
    out['turns'] = turns
    return out


# ---------------------------------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------------------------------


def _norm(s: str, cap: int = 3000) -> str:
    s = '\n'.join(line.rstrip() for line in s.splitlines())
    if len(s) > cap:
        s = s[: cap // 2] + s[-cap // 2:]
    return s


SHORT_EXACT = 25  # strings shorter than this must match exactly: a ratio of 0.8 on 5 chars is one char of noise


def sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    a, b = _norm(a), _norm(b)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if max(len(a), len(b)) < SHORT_EXACT:
        return 0.5 * difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    if sm.real_quick_ratio() < 0.5 or sm.quick_ratio() < 0.5:
        return sm.quick_ratio() * 0.5
    return sm.ratio()


def change_of(old: str, new: str) -> tuple[str, str]:
    """Line-level change between two text blocks: (removed text, added text), context excluded."""
    a, b = old.splitlines(keepends=True), new.splitlines(keepends=True)
    rem: list[str] = []
    add: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ('replace', 'delete'):
            rem.extend(a[i1:i2])
        if tag in ('replace', 'insert'):
            add.extend(b[j1:j2])
    return ''.join(rem), ''.join(add)


def change_from_unified_diff(diff: str) -> tuple[str, str]:
    rem, add = [], []
    for ln in diff.splitlines(keepends=True):
        if ln.startswith(('---', '+++', '@@', 'diff ', 'index ')):
            continue
        if ln.startswith('-'):
            rem.append(ln[1:])
        elif ln.startswith('+'):
            add.append(ln[1:])
    return ''.join(rem), ''.join(add)


@dataclass
class Edit:
    call_pos: int
    path: str
    removed: str
    added: str
    source: str  # diff | args | write_file


def collect_edits(calls: list[Call]) -> list[Edit]:
    """Successful edits as (removed, added) changes. Prefer the unified diff in the tool result; fall back to the args."""
    edits: list[Edit] = []
    last_content: dict[str, str] = {}
    for pos, c in enumerate(calls):
        if c.kind != 'ok' or not isinstance(c.args, dict):
            continue
        path = str(c.args.get('filepath', ''))
        if c.tool == 'edit_file':
            res = parse_json(c.result)
            inner = res.get('result') if isinstance(res, dict) and isinstance(res.get('result'), dict) else res
            diff = inner.get('diff') if isinstance(inner, dict) else None
            if isinstance(diff, str) and diff.strip() and not inner.get('is_truncated'):
                rem, add = change_from_unified_diff(diff)
                src = 'diff'
            else:
                rem, add = change_of(str(c.args.get('old_string', '')), str(c.args.get('new_string', '')))
                src = 'args'
            edits.append(Edit(pos, path, rem, add, src))
        elif c.tool == 'write_file':
            content = str(c.args.get('content', ''))
            if path in last_content and last_content[path] != content:
                rem, add = change_of(last_content[path], content)
                edits.append(Edit(pos, path, rem, add, 'write_file'))
            last_content[path] = content
    return edits


def find_oscillation(edits: list[Edit], th: Thresholds) -> dict[str, Any]:
    """A-B-A: edit j undoes an earlier edit k on the same file (j's removed text ~ k's added text and j's added text ~
    k's removed text, and that direction is closer than 'same direction'). reapplies = k's change applied again after
    being reverted."""
    reverts: list[tuple[int, int]] = []
    reapplies: list[tuple[int, int]] = []
    by_file: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(edits):
        by_file[e.path].append(i)
    for path, idxs in by_file.items():
        open_idx: list[int] = []  # edits not yet reverted, oldest first
        reverted: list[int] = []
        for j in idxs:
            ej = edits[j]
            if max(len(ej.removed), len(ej.added)) < th.aba_min_chars:
                continue
            match = None
            for k in reversed(open_idx):
                ek = edits[k]
                if max(len(ek.removed), len(ek.added)) < th.aba_min_chars:
                    continue
                cross = sim(ej.removed, ek.added) + sim(ej.added, ek.removed)
                same = sim(ej.removed, ek.removed) + sim(ej.added, ek.added)
                if cross / 2 >= th.aba_sim and cross > same + 0.1:
                    match = k
                    break
            if match is not None:
                open_idx.remove(match)
                reverted.append(match)
                reverts.append((match, j))
                continue
            # re-apply of a previously reverted change?
            for k in reverted:
                ek = edits[k]
                same = sim(ej.removed, ek.removed) + sim(ej.added, ek.added)
                cross = sim(ej.removed, ek.added) + sim(ej.added, ek.removed)
                if same / 2 >= th.aba_sim and same > cross + 0.1:
                    reapplies.append((k, j))
                    break
            open_idx.append(j)
    return {
        'reverts': len(reverts), 'reapplies': len(reapplies),
        'revert_pairs': [{'file': edits[k].path, 'apply_call': edits[k].call_pos, 'revert_call': edits[j].call_pos}
                         for k, j in reverts[:10]],
        'files': sorted({edits[k].path for k, _ in reverts}),
    }


def find_repeats(calls: list[Call]) -> dict[str, Any]:
    consecutive = after_error = max_run = 0
    run = 1
    repeat_of_failed = 0
    seen_failed: set[str] = set()
    longest: dict[str, Any] | None = None
    repeat_idx: set[int] = set()
    run_start = 0
    for i, c in enumerate(calls):
        if i > 0 and c.sig == calls[i - 1].sig:
            consecutive += 1
            repeat_idx.add(i)
            run += 1
            if calls[i - 1].is_error:
                after_error += 1
        else:
            run = 1
            run_start = i
        if run > max_run:
            max_run = run
            longest = {'tool': c.tool, 'run': run, 'start_call': run_start, 'kind': calls[run_start].kind,
                       'error': ' '.join((calls[run_start].err_msg or calls[run_start].err_type).split())[:160],
                       'args_preview': ' '.join((c.raw_args or '').split())[:160]}
        if c.sig in seen_failed:
            repeat_of_failed += 1
            repeat_idx.add(i)
        if c.is_error:
            seen_failed.add(c.sig)
    return {'consecutive': consecutive, 'after_error': after_error, 'max_run': max_run,
            'repeat_of_failed': repeat_of_failed, 'repeat_idx': sorted(repeat_idx), 'longest': longest}


# ---------------------------------------------------------------------------------------------------
# per-task analysis
# ---------------------------------------------------------------------------------------------------


def problem_probe(agent_calls: list[dict[str, Any]], task_text: str | None) -> str | None:
    if task_text:
        return ' '.join(task_text.split())[:120]
    for r in agent_calls[:1]:
        for m in (r.get('request') or {}).get('messages') or []:
            t = _msg_text(m)
            if PROBLEM_MARKER in t:
                first_par = t.split(PROBLEM_MARKER, 1)[1].lstrip('\n').split('\n\n')[0]
                return ' '.join(first_par.split())[:120] or None
    return None


def contains_probe(messages: list[dict[str, Any]], probe: str) -> list[str]:
    roles = []
    for m in messages:
        if probe in ' '.join(_msg_text(m).split()):
            roles.append(str(m.get('role')))
    return roles


def compaction_info(task_dir: Path, data: dict[str, Any], probe: str | None) -> dict[str, Any]:
    comps = read_jsonl(task_dir / 'compactions.jsonl')
    info: dict[str, Any] = {'count': len(comps), 'problem_after': [], 'lost_problem': 0, 'unknown': 0,
                            'first_after_llm_call': comps[0].get('after_llm_call_index') if comps else None}
    agent = data['agent_calls']
    for c in comps:
        idx = c.get('after_llm_call_index') or 0
        nxt = next((r for r in agent if (r.get('call_index') or 0) > idx), None)
        if nxt is None or probe is None:
            info['unknown'] += 1
            info['problem_after'].append(None)
            continue
        roles = contains_probe((nxt.get('request') or {}).get('messages') or [], probe)
        info['problem_after'].append(roles)
        if not roles:
            info['lost_problem'] += 1
    return info


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(math.ceil(p / 100 * len(xs))) - 1))
    return xs[k]


def analyze_task(run: str, run_dir: Path, iid: str, th: Thresholds, grades: dict[str, bool],
                 problem_texts: dict[str, str]) -> dict[str, Any]:
    td = run_dir / 'tasks' / iid
    try:
        result = json.loads((td / 'result.json').read_text())
    except Exception:
        result = {}
    data = load_task_calls(td)
    turns: list[Turn] = data['turns']
    calls = [c for t in turns for c in t.calls]
    n_calls = len(calls)
    kinds = Counter(c.kind for c in calls)
    by_tool = Counter(c.tool for c in calls)

    rep = find_repeats(calls)
    rejected = [c for c in calls if c.kind in REJECTED]
    sig_counter: Counter = Counter()
    for c in calls:
        for s in c.signatures:
            sig_counter[s] += 1
    parser_sig = sig_counter['mangled_key'] + sig_counter['special_token_in_value'] + data['unparsed_text']
    edit_calls = [c for c in calls if c.tool == 'edit_file']
    edit_fail = [c for c in edit_calls if c.kind == 'edit_fail']
    edit_causes = Counter(c.cause or 'other' for c in edit_fail)
    edits = collect_edits(calls)
    osc = find_oscillation(edits, th)
    revert_cmds = sum(1 for c in calls if c.tool == 'run_command' and isinstance(c.args, dict)
                      and REVERT_CMD.search(str(c.args.get('command', ''))))

    # generation tail
    agent = data['agent_calls']
    comp_tokens = [int((r.get('usage') or {}).get('completion_tokens') or 0) for r in agent]
    cap = int(data['cap'] or th.default_cap)
    n_length = sum(1 for r in agent if r.get('finish_reason') in ('length', 'max_tokens'))
    n_cap = sum(1 for x in comp_tokens if x >= th.runaway_frac * cap)
    n_tail = sum(1 for x in comp_tokens if x >= th.tail_tokens)
    runaway_time = sum(float(r.get('latency_s') or 0) for r, x in zip(agent, comp_tokens)
                       if x >= th.runaway_frac * cap or r.get('finish_reason') in ('length', 'max_tokens'))

    # time accounting at turn level
    agent_s = float((result.get('timings') or {}).get('agent_s') or 0.0)
    if not agent_s and turns:
        agent_s = sum(t.total_s for t in turns)
    repeat_idx = set(rep['repeat_idx'])
    waste = {'rejected_s': 0.0, 'retry_s': 0.0, 'edit_fail_s': 0.0, 'budget_timeout_s': 0.0, 'other_failed_s': 0.0}
    wasted_turns = 0
    pos = 0
    for t in turns:
        idxs = range(pos, pos + len(t.calls))
        pos += len(t.calls)
        cs = t.calls
        if all(c.failed or (i in repeat_idx and c.is_error) for c, i in zip(cs, idxs)) or \
                all(i in repeat_idx for i in idxs):
            wasted_turns += 1
            if any(c.kind in REJECTED for c in cs):
                waste['rejected_s'] += t.total_s
            elif any(i in repeat_idx for i in idxs):
                waste['retry_s'] += t.total_s
            elif any(c.kind in ('edit_fail', 'write_fail') for c in cs):
                waste['edit_fail_s'] += t.total_s
            elif any(c.kind in ('budget', 'timeout') for c in cs):
                waste['budget_timeout_s'] += t.total_s
            else:
                waste['other_failed_s'] += t.total_s
    failed_s = sum(waste.values())

    # outcome
    patch_chars = int(result.get('patch_chars') or 0)
    empty_patch = bool(result.get('empty_patch')) if 'empty_patch' in result else None
    probe = problem_probe(agent, problem_texts.get(iid))
    comp = compaction_info(td, data, probe)
    resolved = grades.get(iid)

    m = {
        'run': run, 'instance_id': iid, 'resolved': resolved, 'source': data['source'],
        'status': result.get('status'), 'termination': result.get('termination'),
        'patch_chars': patch_chars, 'empty_patch': empty_patch, 'patch_source': result.get('patch_source'),
        'nudges': result.get('nudges'), 'agent_s': round(agent_s, 1),
        'turns': len(turns), 'tool_calls': n_calls, 'by_tool': dict(by_tool), 'kinds': dict(kinds),
        'llm_calls': len(agent), 'llm_errors': data['llm_errors'],
        'repeat': {k: v for k, v in rep.items() if k != 'repeat_idx'},
        'rejected': {'total': len(rejected), 'by_kind': dict(Counter(c.kind for c in rejected)),
                     'missing_params': dict(Counter(name for c in rejected for blk in
                                                    re.findall(r'present:\n(.*?)\nYou could', c.err_msg, re.S)
                                                    for name in blk.split('\n') if name))},
        'parser_signatures': {**dict(sig_counter), 'unparsed_tool_text': data['unparsed_text'],
                              'truncated_tool_text': data['truncated_tool_text']},
        'edit_file': {'calls': len(edit_calls), 'failed': len(edit_fail), 'causes': dict(edit_causes),
                      'rejected_by_adk': sum(1 for c in edit_calls if c.kind in REJECTED),
                      'rate': round(len(edit_fail) / len(edit_calls), 3) if edit_calls else None},
        'oscillation': {**osc, 'edits': len(edits), 'revert_cmds': revert_cmds},
        'generation': {'cap': cap, 'calls': len(comp_tokens), 'max_tokens': max(comp_tokens, default=0),
                       'p50': pct(comp_tokens, 50), 'p90': pct(comp_tokens, 90), 'p99': pct(comp_tokens, 99),
                       'n_length_finish': n_length, 'n_at_cap': n_cap, 'n_tail': n_tail,
                       'runaway_time_s': round(runaway_time, 1)},
        'time': {**{k: round(v, 1) for k, v in waste.items()}, 'failed_total_s': round(failed_s, 1),
                 'wasted_turns': wasted_turns,
                 'share_of_agent': round(failed_s / agent_s, 3) if agent_s else None},
        'compaction': comp,
    }

    flags: list[str] = []
    if rep['consecutive'] >= 1:
        flags.append('identical_repeat')
    if rep['max_run'] >= th.loop_min_run:
        flags.append('retry_loop')
    if rep['after_error'] >= 1:
        flags.append('identical_after_error')
    if rejected:
        flags.append('adk_rejected')
    if parser_sig:
        flags.append('parser_signature')
    if edit_fail:
        flags.append('edit_fail')
    if len(edit_fail) >= th.edit_fail_heavy_n or (len(edit_calls) >= 4 and len(edit_fail) / len(edit_calls)
                                                   >= th.edit_fail_heavy_rate):
        flags.append('edit_fail_heavy')
    if osc['reverts'] >= 1:
        flags.append('aba_revert_any')
    if osc['reverts'] >= th.aba_min_reverts or osc['reapplies'] >= 1:
        flags.append('aba_oscillation')
    if n_length or n_cap:
        flags.append('runaway_generation')
    if n_tail:
        flags.append('long_tail_generation')
    if agent_s and failed_s / agent_s >= th.waste_frac:
        flags.append('time_wasted_high')
    if kinds['budget']:
        flags.append('tool_budget_hit')
    if (result.get('nudges') or 0) >= 1:
        flags.append('nudged')
    if empty_patch:
        flags.append('empty_patch')
    if result.get('termination') in ('time_budget', 'tool_budget', 'turn_budget'):
        flags.append('budget_termination')
    if result.get('status') in ('worker_timeout', 'killed', 'infra_error'):
        flags.append('worker_failure')
    if comp['count']:
        flags.append('compaction')
    if comp['lost_problem']:
        flags.append('compaction_lost_problem')
    m['flags'] = flags
    return m


MODE_DOC = {
    'identical_repeat': 'a tool call identical to the previous one (>=1)',
    'retry_loop': 'run of identical consecutive calls >= --loop-min-run',
    'identical_after_error': 'an identical call right after the same call errored',
    'adk_rejected': 'ADK rejected a call (missing/invalid args, unknown tool)',
    'parser_signature': 'mangled arg keys / special tokens in args / raw tool-call tag text in content',
    'edit_fail': 'any edit_file failure',
    'edit_fail_heavy': 'edit_file failures >= --edit-fail-n or rate >= --edit-fail-rate',
    'aba_revert_any': 'one edit undid an earlier edit on the same file',
    'aba_oscillation': 'reverts >= --aba-min-reverts, or a reverted change re-applied',
    'runaway_generation': 'a call hit max_output_tokens (finish=length or >= frac of cap)',
    'long_tail_generation': 'a call with >= --tail-tokens completion tokens',
    'time_wasted_high': 'failed/retried turns >= --waste-frac of agent time',
    'tool_budget_hit': 'a BudgetExceeded tool response',
    'nudged': 'the harness had to nudge the model (text-only/truncated turns)',
    'empty_patch': 'final patch empty (NO_PATCH)',
    'budget_termination': 'ended by time/tool/turn budget',
    'worker_failure': 'worker killed / timed out / infra error',
    'compaction': 'ADK compaction fired at least once',
    'compaction_lost_problem': 'issue opening not quoted verbatim in the first call after a compaction (B2 probe, not content loss)',
}


# ---------------------------------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------------------------------


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a,b],[c,d]] (a=flagged&resolved, b=flagged&unresolved, ...)."""
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2
    if n == 0:
        return 1.0

    def pmf(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)

    lo, hi = max(0, k - n2), min(k, n1)
    p0 = pmf(a)
    return min(1.0, sum(p for x in range(lo, hi + 1) if (p := pmf(x)) <= p0 * (1 + 1e-9)))


def mode_stats(tasks: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    graded = [t for t in tasks if t['resolved'] is not None]
    flagged = [t for t in tasks if mode in t['flags']]
    g_flag = [t for t in graded if mode in t['flags']]
    g_not = [t for t in graded if mode not in t['flags']]
    res_f = sum(1 for t in g_flag if t['resolved'])
    res_n = sum(1 for t in g_not if t['resolved'])
    unres = [t for t in graded if not t['resolved']]
    res = [t for t in graded if t['resolved']]
    unres_aff = sum(1 for t in unres if mode in t['flags'])
    res_aff = sum(1 for t in res if mode in t['flags'])

    def r(x: int, n: int) -> float | None:
        return round(x / n, 3) if n else None

    return {
        'mode': mode, 'tasks_flagged': len(flagged), 'tasks': len(tasks),
        'rate': r(len(flagged), len(tasks)),
        'graded': len(graded), 'flagged_graded': len(g_flag),
        'flagged_resolved': res_f, 'flagged_unresolved': len(g_flag) - res_f,
        'resolve_rate_flagged': r(res_f, len(g_flag)), 'resolve_rate_unflagged': r(res_n, len(g_not)),
        'unresolved_total': len(unres), 'unresolved_affected': unres_aff,
        'share_of_unresolved_affected': r(unres_aff, len(unres)),
        'share_of_resolved_affected': r(res_aff, len(res)),
        'fisher_p': round(fisher_exact(res_f, len(g_flag) - res_f, res_n, len(g_not) - res_n), 4)
        if g_flag and g_not else None,
    }


def event_totals(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    def tot(f) -> int:
        return int(sum(f(t) for t in tasks))

    ec: Counter = Counter()
    for t in tasks:
        ec.update(t['edit_file']['causes'])
    n_calls = tot(lambda t: t['tool_calls'])
    return {
        'tasks': len(tasks), 'tool_calls': n_calls, 'turns': tot(lambda t: t['turns']),
        'identical_consecutive_calls': tot(lambda t: t['repeat']['consecutive']),
        'identical_after_error': tot(lambda t: t['repeat']['after_error']),
        'repeat_of_failed_calls': tot(lambda t: t['repeat']['repeat_of_failed']),
        'longest_identical_run': max((t['repeat']['max_run'] for t in tasks), default=0),
        'adk_rejected_calls': tot(lambda t: t['rejected']['total']),
        'mangled_key_calls': tot(lambda t: t['parser_signatures'].get('mangled_key', 0)),
        'unparsed_tool_text': tot(lambda t: t['parser_signatures'].get('unparsed_tool_text', 0)),
        'edit_file_calls': tot(lambda t: t['edit_file']['calls']),
        'edit_file_failed': tot(lambda t: t['edit_file']['failed']),
        'edit_file_rejected_by_adk': tot(lambda t: t['edit_file']['rejected_by_adk']),
        'edit_file_fail_causes': dict(ec),
        'aba_reverts': tot(lambda t: t['oscillation']['reverts']),
        'aba_reapplies': tot(lambda t: t['oscillation']['reapplies']),
        'revert_commands': tot(lambda t: t['oscillation']['revert_cmds']),
        'calls_at_cap': tot(lambda t: t['generation']['n_at_cap']),
        'calls_finish_length': tot(lambda t: t['generation']['n_length_finish']),
        'agent_s': round(sum(t['agent_s'] for t in tasks), 1),
        'failed_retried_s': round(sum(t['time']['failed_total_s'] for t in tasks), 1),
        'compactions': tot(lambda t: t['compaction']['count']),
        'compactions_lost_problem': tot(lambda t: t['compaction']['lost_problem']),
        'compactions_issue_not_verbatim': tot(lambda t: t['compaction']['lost_problem']),  # alias, same count
        'empty_patches': sum(1 for t in tasks if t['empty_patch']),
    }


def summarize(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [t for t in tasks if t['resolved'] is not None]
    return {
        'tasks': len(tasks), 'graded': len(graded),
        'resolved': sum(1 for t in graded if t['resolved']),
        'resolve_rate': round(sum(1 for t in graded if t['resolved']) / len(graded), 3) if graded else None,
        'totals': event_totals(tasks),
        'modes': [mode_stats(tasks, m) for m in MODE_DOC],
        'by_outcome': {
            label: event_totals([t for t in graded if bool(t['resolved']) == want])
            for label, want in (('resolved', True), ('unresolved', False))
        },
    }


# ---------------------------------------------------------------------------------------------------
# run discovery
# ---------------------------------------------------------------------------------------------------


def find_runs(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if (p / 'run.json').exists() or (p / 'tasks').is_dir():
            out.append(p)
        else:
            subs = sorted(s for s in p.glob('*') if (s / 'run.json').exists() or (s / 'tasks').is_dir())
            out.extend(subs)
    return out


def load_problem_texts(run_dir: Path) -> dict[str, str]:
    try:
        manifest = json.loads((run_dir / 'run.json').read_text())
        tp = Path((manifest.get('data') or {}).get('tasks_path', ''))
    except Exception:
        return {}
    if not tp.is_file():
        return {}
    out = {}
    for line in tp.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            row = json.loads(line)
            out[row['instance_id']] = row.get('problem_statement') or ''
        except Exception:
            continue
    return out


def load_grades(run_dir: Path, explicit: str | None) -> dict[str, bool]:
    try:
        from harness.runner.report import load_grades as _lg
    except Exception:  # pragma: no cover - analyzer should still work without the runner package importable
        _lg = None
    for p in ([Path(explicit)] if explicit else []) + [run_dir / 'grades.jsonl']:
        if p.exists():
            if _lg:
                return _lg(p)
            rows = read_jsonl(p)
            return {r['instance_id']: bool(r.get('resolved', r.get('passed'))) for r in rows if 'instance_id' in r}
    return {}


def analyze_runs(run_dirs: list[Path], th: Thresholds, grades_arg: str | None = None,
                 use_task_text: bool = True) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    notes: list[str] = []
    per_run_meta: dict[str, Any] = {}
    for rd in run_dirs:
        name = rd.name
        grades = load_grades(rd, grades_arg)
        texts = load_problem_texts(rd) if use_task_text else {}
        ids = []
        try:
            ids = json.loads((rd / 'run.json').read_text()).get('task_ids') or []
        except Exception:
            pass
        if not ids:
            ids = sorted(p.name for p in (rd / 'tasks').glob('*') if p.is_dir())
        n0 = len(tasks)
        for iid in ids:
            if not (rd / 'tasks' / iid).is_dir():
                continue
            tasks.append(analyze_task(name, rd, iid, th, grades, texts))
        if not grades:
            notes.append(f'{name}: no grades.jsonl; correlations skipped for this run')
        per_run_meta[name] = {'path': str(rd), 'tasks': len(tasks) - n0, 'graded': sum(1 for t in tasks[n0:] if t['resolved'] is not None)}
    runs = sorted({t['run'] for t in tasks})
    summary = {
        'thresholds': asdict(th), 'inputs': per_run_meta, 'notes': notes,
        'pooled': summarize(tasks),
        'per_run': {r: summarize([t for t in tasks if t['run'] == r]) for r in runs},
        'mode_definitions': MODE_DOC,
        'tasks': tasks,
    }
    return summary


# ---------------------------------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------------------------------


def _f(x: Any, nd: int = 2) -> str:
    if x is None:
        return '-'
    if isinstance(x, float):
        return f'{x:.{nd}f}'.rstrip('0').rstrip('.') if nd else str(round(x))
    return str(x)


def md_modes(modes: list[dict[str, Any]]) -> list[str]:
    L = ['| failure mode | tasks flagged | resolve rate flagged / not | unresolved tasks affected | resolved tasks affected | Fisher p |',
         '|---|---|---|---|---|---|']
    for m in modes:
        L.append(f"| `{m['mode']}` | {m['tasks_flagged']}/{m['tasks']} ({_pc(m['rate'])}) "
                 f"| {_pc(m['resolve_rate_flagged'])} ({m['flagged_resolved']}/{m['flagged_graded']}) / "
                 f"{_pc(m['resolve_rate_unflagged'])} "
                 f"| {m['unresolved_affected']}/{m['unresolved_total']} ({_pc(m['share_of_unresolved_affected'])}) "
                 f"| {_pc(m['share_of_resolved_affected'])} | {_f(m['fisher_p'], 3)} |")
    return L


def _pc(x: float | None) -> str:
    return '-' if x is None else f'{round(100 * x)}%'


def md_totals(tot: dict[str, Any]) -> list[str]:
    keys = [('tool_calls', 'tool calls'), ('turns', 'turns with tool calls'),
            ('identical_consecutive_calls', 'identical consecutive calls'),
            ('identical_after_error', '  of which right after an error'),
            ('repeat_of_failed_calls', 'calls repeating an earlier failed call'),
            ('longest_identical_run', 'longest identical run'),
            ('adk_rejected_calls', 'ADK-rejected calls'), ('mangled_key_calls', 'calls with mangled arg keys'),
            ('unparsed_tool_text', 'raw tool-call tag text in message content'),
            ('edit_file_calls', 'edit_file calls'), ('edit_file_failed', 'edit_file failures (tool ran, edit not applied)'),
            ('edit_file_rejected_by_adk', 'edit_file calls rejected by ADK before the tool ran'),
            ('aba_reverts', 'A-B-A reverts'), ('aba_reapplies', 'A-B-A re-applies'),
            ('revert_commands', 'git checkout/restore/stash style commands'),
            ('calls_at_cap', 'calls at max_output_tokens'), ('calls_finish_length', 'calls with finish_reason=length'),
            ('agent_s', 'agent seconds'), ('failed_retried_s', 'seconds in failed/retried turns'),
            ('compactions', 'compactions'), ('compactions_lost_problem', 'issue opening not quoted verbatim after compaction'),
            ('empty_patches', 'empty patches')]
    L = ['| metric | value |', '|---|---|']
    for k, label in keys:
        L.append(f'| {label} | {tot.get(k)} |')
    if tot.get('edit_file_fail_causes'):
        causes = ', '.join(f'{k}={v}' for k, v in sorted(tot['edit_file_fail_causes'].items(), key=lambda kv: -kv[1]))
        L.append(f'| edit_file failure causes | {causes} |')
    return L


def render_markdown(summary: dict[str, Any], max_tasks: int = 40, show_all: bool = False) -> str:
    L: list[str] = ['# Reliability report', '']
    th = summary['thresholds']
    L.append('Thresholds: ' + ', '.join(f'{k}={v}' for k, v in th.items()))
    L.append('')
    L.append('| run | tasks | graded | resolved | resolve rate |')
    L.append('|---|---|---|---|---|')
    for r, s in summary['per_run'].items():
        L.append(f"| {r} | {s['tasks']} | {s['graded']} | {s['resolved']} | {_pc(s['resolve_rate'])} |")
    for n in summary['notes']:
        L.append(f'\n> {n}')
    pooled = summary['pooled']
    L += ['', f"## Aggregate (pooled over {len(summary['per_run'])} run(s), {pooled['tasks']} tasks)", '']
    L += md_totals(pooled['totals'])
    L += ['', '### Failure modes vs outcome', '']
    L += md_modes(pooled['modes'])
    bo = pooled['by_outcome']
    if bo['resolved']['tasks'] or bo['unresolved']['tasks']:
        L += ['', '### Event counts by outcome', '', '| event | resolved tasks | unresolved tasks |', '|---|---|---|']
        for k in ('tool_calls', 'identical_consecutive_calls', 'identical_after_error', 'adk_rejected_calls',
                  'mangled_key_calls', 'edit_file_failed', 'aba_reverts', 'calls_at_cap', 'agent_s',
                  'failed_retried_s', 'compactions', 'compactions_lost_problem'):
            L.append(f"| {k} | {bo['resolved'].get(k, 0)} | {bo['unresolved'].get(k, 0)} |")
        L.append(f"| tasks | {bo['resolved']['tasks']} | {bo['unresolved']['tasks']} |")
    if len(summary['per_run']) > 1:
        L += ['', '## Per run', '']
        names = list(summary['per_run'])
        L.append('| failure mode | ' + ' | '.join(f'{n}: flagged (resolve rate)' for n in names) + ' |')
        L.append('|---|' + '---|' * len(names))
        for mode in MODE_DOC:
            cells = []
            for n in names:
                ms = next(m for m in summary['per_run'][n]['modes'] if m['mode'] == mode)
                cells.append(f"{ms['tasks_flagged']}/{ms['tasks']} ({_pc(ms['resolve_rate_flagged'])})")
            if any(not c.startswith('0/') for c in cells):
                L.append(f'| `{mode}` | ' + ' | '.join(cells) + ' |')
    tasks = summary['tasks']
    shown = tasks if show_all else sorted(
        [t for t in tasks if set(t['flags']) - {'compaction', 'identical_repeat', 'nudged'}],
        key=lambda t: (-t['repeat']['max_run'], -t['time']['failed_total_s']))[:max_tasks]
    L += ['', f'## Tasks ({len(shown)} of {len(tasks)} shown; `--all-tasks` for every task)', '',
          '| run | task | res | calls | maxrun | rej | editfail | revert | cap | waste s (share) | comp (lost) | flags |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for t in shown:
        res = {True: 'yes', False: 'no', None: '?'}[t['resolved']]
        L.append(f"| {t['run']} | {t['instance_id']} | {res} | {t['tool_calls']} | {t['repeat']['max_run']} | "
                 f"{t['rejected']['total']} | {t['edit_file']['failed']}/{t['edit_file']['calls']} | "
                 f"{t['oscillation']['reverts']} | {t['generation']['n_at_cap']} | "
                 f"{t['time']['failed_total_s']} ({_pc(t['time']['share_of_agent'])}) | "
                 f"{t['compaction']['count']} ({t['compaction']['lost_problem']}) | {', '.join(t['flags'])} |")
    loops = sorted([t for t in tasks if t['repeat']['longest'] and t['repeat']['max_run'] >= th['loop_min_run']],
                   key=lambda t: -t['repeat']['max_run'])[:8]
    if loops:
        L += ['', '## Longest identical-call runs', '']
        for t in loops:
            lg = t['repeat']['longest']
            L.append(f"- {t['run']}/{t['instance_id']}: `{lg['tool']}` x{lg['run']} from call {lg['start_call']} "
                     f"({lg['kind']}: {lg['error'] or 'no error'}) args `{lg['args_preview']}`")
    osc = sorted([t for t in tasks if t['oscillation']['reverts']], key=lambda t: -t['oscillation']['reverts'])[:8]
    if osc:
        L += ['', '## A-B-A examples', '']
        for t in osc:
            o = t['oscillation']
            L.append(f"- {t['run']}/{t['instance_id']}: {o['reverts']} reverts, {o['reapplies']} re-applies, files "
                     f"{', '.join(o['files'])}; pairs (apply call -> revert call) "
                     + ', '.join(f"{p['apply_call']}->{p['revert_call']}" for p in o['revert_pairs'][:6]))
    L += ['', '## Mode definitions', '']
    for k, v in MODE_DOC.items():
        L.append(f'- `{k}`: {v}')
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='runs/<name>/ dirs, or a parent dir holding several (runs/ab1)')
    ap.add_argument('--grades', help='grades.jsonl to use for every run (default: <run>/grades.jsonl)')
    ap.add_argument('--out', help='write <out>.md and <out>.json (or <out>/reliability.{md,json} if it is a dir)')
    ap.add_argument('--all-tasks', action='store_true', help='list every task in the markdown table')
    ap.add_argument('--max-tasks', type=int, default=40)
    ap.add_argument('--json-only', action='store_true')
    ap.add_argument('--no-task-text', action='store_true',
                    help="don't read tasks.jsonl for the problem-statement probe (use the first prompt instead)")
    d = Thresholds()
    ap.add_argument('--loop-min-run', type=int, default=d.loop_min_run)
    ap.add_argument('--aba-sim', type=float, default=d.aba_sim)
    ap.add_argument('--aba-min-reverts', type=int, default=d.aba_min_reverts)
    ap.add_argument('--aba-min-chars', type=int, default=d.aba_min_chars)
    ap.add_argument('--runaway-frac', type=float, default=d.runaway_frac)
    ap.add_argument('--tail-tokens', type=int, default=d.tail_tokens)
    ap.add_argument('--default-cap', type=int, default=d.default_cap)
    ap.add_argument('--edit-fail-n', type=int, default=d.edit_fail_heavy_n)
    ap.add_argument('--edit-fail-rate', type=float, default=d.edit_fail_heavy_rate)
    ap.add_argument('--waste-frac', type=float, default=d.waste_frac)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    th = Thresholds(loop_min_run=a.loop_min_run, aba_sim=a.aba_sim, aba_min_reverts=a.aba_min_reverts,
                    aba_min_chars=a.aba_min_chars, runaway_frac=a.runaway_frac, tail_tokens=a.tail_tokens,
                    default_cap=a.default_cap, edit_fail_heavy_n=a.edit_fail_n,
                    edit_fail_heavy_rate=a.edit_fail_rate, waste_frac=a.waste_frac)
    runs = find_runs(a.runs)
    if not runs:
        print('ERROR: no run directories found (expected runs/<name>/ with run.json or tasks/)', file=sys.stderr)
        return 2
    summary = analyze_runs(runs, th, a.grades, use_task_text=not a.no_task_text)
    md = render_markdown(summary, a.max_tasks, a.all_tasks)
    if a.out:
        out = Path(a.out)
        if out.suffix == '' and (out.is_dir() or not out.exists()):
            out.mkdir(parents=True, exist_ok=True)
            md_path, js_path = out / 'reliability.md', out / 'reliability.json'
        else:
            md_path, js_path = out.with_suffix('.md'), out.with_suffix('.json')
        md_path.write_text(md, encoding='utf-8')
        js_path.write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
        print(f'wrote {md_path} and {js_path}', file=sys.stderr)
    if a.json_only:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(md)
    return 0


if __name__ == '__main__':
    sys.exit(main())
