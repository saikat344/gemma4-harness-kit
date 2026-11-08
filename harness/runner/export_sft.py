"""Turn a run's llm_calls.jsonl files into chat-format SFT samples (OpenAI messages + tools).

Limitation: `--only-resolved` admits by instance id, without binding the grade to this run or to its patch. Use one
grades file per run, and treat the output as for inspection unless you add that binding yourself.

Each agent LLM call is (request.messages, request.tools) -> response.choices[0].message. Consecutive calls
share history, so a call whose request (plus its response) is an exact prefix of the next call's request is
dropped: one sample per maximal conversation segment (segments break at compaction, where ADK rewrites the
history). Every assistant turn inside a sample is a training target; a trainer may mask non-final turns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.runner.report import load_grades


def _assistant_from_response(resp: dict[str, Any]) -> dict[str, Any] | None:
    try:
        msg = resp['choices'][0]['message']
    except (KeyError, IndexError, TypeError):
        return None
    out: dict[str, Any] = {'role': 'assistant', 'content': msg.get('content')}
    reasoning = msg.get('reasoning_content') or msg.get('reasoning')
    if reasoning:
        out['reasoning_content'] = reasoning
    if msg.get('tool_calls'):
        out['tool_calls'] = [
            {'id': tc.get('id'), 'type': 'function',
             'function': {'name': tc['function']['name'], 'arguments': tc['function']['arguments']}}
            for tc in msg['tool_calls']
        ]
    return out


def task_samples(calls_path: Path, include_summarizer: bool = False) -> list[dict[str, Any]]:
    calls = []
    for line in calls_path.read_text(encoding='utf-8').splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get('error') or not rec.get('response'):
            continue
        if rec.get('kind') == 'summarizer' and not include_summarizer:
            continue
        calls.append(rec)
    calls.sort(key=lambda r: r['call_index'])
    samples = []
    for i, rec in enumerate(calls):
        msgs = rec['request']['messages']
        nxt = calls[i + 1]['request']['messages'] if i + 1 < len(calls) else None
        if nxt is not None and len(nxt) > len(msgs) and nxt[: len(msgs)] == msgs:
            continue  # subsumed by the next call's history
        asst = _assistant_from_response(rec['response'])
        if asst is None:
            continue
        samples.append({
            'messages': [*msgs, asst],
            'tools': rec['request'].get('tools'),
            'meta': {'call_index': rec['call_index'], 'kind': rec.get('kind'),
                     'finish_reason': rec.get('finish_reason'), 'usage': rec.get('usage'),
                     'params': {k: v for k, v in (rec['request'].get('params') or {}).items()
                                if k in ('extra_body', 'temperature', 'top_p', 'max_tokens', 'max_completion_tokens', 'seed')}},
        })
    return samples


def export(run_dir: Path, out: Path, grades_path: str | None = None, only_resolved: bool = False,
           include_summarizer: bool = False) -> int:
    manifest = json.loads((run_dir / 'run.json').read_text())
    grades = load_grades(grades_path) if grades_path else {}
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        for iid in manifest['task_ids']:
            calls = run_dir / 'tasks' / iid / 'llm_calls.jsonl'
            if not calls.exists():
                continue
            resolved = grades.get(iid)
            if only_resolved and not resolved:
                continue
            for s in task_samples(calls, include_summarizer):
                s['meta'].update({'run': manifest['name'], 'instance_id': iid, 'resolved': resolved,
                                  'submission_sha256': manifest.get('submission_sha256')})
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
                n += 1
    return n
