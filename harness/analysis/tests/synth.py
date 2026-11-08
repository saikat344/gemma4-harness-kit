"""Synthetic runner output (runs/<name>/tasks/<id>/...) for analyzer tests: realistic llm_calls.jsonl,
compactions.jsonl, result.json, grades.jsonl, written without Docker or a model."""

from __future__ import annotations

import copy
import difflib
import json
from pathlib import Path
from typing import Any

TOOLS = [
    {'type': 'function', 'function': {'name': n, 'description': n, 'parameters': {
        'type': 'object', 'properties': {p: {'type': 'string'} for p in params}, 'required': req}}}
    for n, params, req in [
        ('edit_file', ['filepath', 'old_string', 'new_string', 'allow_multiple'], ['filepath', 'old_string', 'new_string']),
        ('write_file', ['filepath', 'content'], ['filepath', 'content']),
        ('read_file', ['filepath', 'start_line', 'end_line'], ['filepath']),
        ('run_command', ['command'], ['command']),
        ('submit_patch', [], []),
    ]
]
ADK_MISSING = ('Invoking `{tool}()` failed as the following mandatory input parameters are not present:\n{params}\n'
               'You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters.')


def ok_edit(path: str, old: str, new: str) -> str:
    diff = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), path, path))
    return json.dumps({'status': 'ok', 'filepath': path, 'occurrences': 1, 'strategy': 'exact', 'diff': diff,
                       'is_truncated': False})


def edit_err(msg: str) -> str:
    return json.dumps({'status': 'error', 'error_type': 'FileEditError', 'error_message': msg})


def adk_missing(tool: str, *params: str) -> str:
    return json.dumps({'error': ADK_MISSING.format(tool=tool, params='\n'.join(params))})


def unknown_tool(name: str) -> str:
    return json.dumps({'status': 'error', 'error_type': 'ValueError', 'error': (
        f"Tool '{name}' not found.\nAvailable tools: run_command, read_file, edit_file\n\nPossible causes:\n  1. x")})


def ok(**kw: Any) -> str:
    return json.dumps({'status': 'ok', **kw})


NOT_FOUND = 'Failed to replace: old_string not found. Ensure you\'re not escaping content incorrectly and check whitespace.'
MULTI = 'Expected 1 occurrence but found 3. If you intended to replace multiple occurrences, set \'allow_multiple\' to true.'


class TaskBuilder:
    def __init__(self, run_dir: Path, iid: str, *, problem: str = 'Fix the frobnicator so that it handles empty input.',
                 cap: int | None = 8192, instruction_has_problem: bool = False, tools: list | None = ...):
        self.dir = run_dir / 'tasks' / iid
        self.dir.mkdir(parents=True, exist_ok=True)
        self.iid = iid
        self.problem = problem
        self.cap = cap
        self.t = 1000.0
        self.n = 0
        self.agent_s = 0.0
        self.recs: list[dict[str, Any]] = []
        self.comps: list[dict[str, Any]] = []
        self.tools = TOOLS if tools is ... else tools
        sys_msg = 'You are a SWE agent.' + (f'\nTask: {problem}' if instruction_has_problem else '')
        self.messages: list[dict[str, Any]] = [
            {'role': 'system', 'content': sys_msg},
            {'role': 'user', 'content': f'You are evaluating repo x.\n\nProblem Statement:\n{problem}\n\n## Instructions'},
        ]
        self.call_no = 0
        self.nudges = 0

    def _params(self) -> dict[str, Any]:
        p: dict[str, Any] = {'extra_body': {'chat_template_kwargs': {'enable_thinking': True}}}
        if self.cap:
            p['max_completion_tokens'] = self.cap
        return p

    def turn(self, calls: list[tuple[str, Any, str | None]], *, tokens: int = 120, latency: float = 2.0,
             gap: float = 1.0, finish: str = 'tool_calls', content: str | None = None) -> None:
        """calls: (tool, args dict | raw-json-string, result text). One agent LLM call + tool results."""
        self.n += 1
        tcs = []
        for i, (name, args, _res) in enumerate(calls):
            raw = args if isinstance(args, str) else json.dumps(args)
            tcs.append({'id': f'call_{self.n}_{i}', 'type': 'function', 'function': {'name': name, 'arguments': raw}})
        msg: dict[str, Any] = {'role': 'assistant', 'content': content}
        if tcs:
            msg['tool_calls'] = tcs
        rec = {'call_index': self.n, 'kind': 'agent', 't_start': self.t, 'latency_s': latency,
               'request': {'model': 'openai/m', 'messages': copy.deepcopy(self.messages), 'tools': self.tools,
                           'params': self._params()},
               'response': {'choices': [{'index': 0, 'message': msg, 'finish_reason': finish}],
                            'usage': {'prompt_tokens': 2000, 'completion_tokens': tokens}},
               'usage': {'prompt_tokens': 2000, 'completion_tokens': tokens}, 'finish_reason': finish}
        self.recs.append(rec)
        self.t += latency + gap
        self.agent_s += latency + gap
        self.messages.append(msg)
        for tc, (_n, _a, res) in zip(tcs, calls):
            self.messages.append({'role': 'tool', 'tool_call_id': tc['id'], 'content': res if res is not None else ''})

    def compaction(self, keep_problem: bool = False) -> None:
        """ADK compaction: a summarizer call, then history collapses to system + summary + last 2 messages."""
        self.n += 1
        self.recs.append({'call_index': self.n, 'kind': 'summarizer', 't_start': self.t, 'latency_s': 3.0,
                          'request': {'model': 'openai/m', 'messages': [{'role': 'user', 'content':
                                      'The following is a conversation history between a user and an AI agent ...'}],
                                      'tools': None, 'params': {}},
                          'response': {'choices': [{'message': {'role': 'assistant', 'content': 'Summary'},
                                                    'finish_reason': 'stop'}]}, 'finish_reason': 'stop'})
        self.t += 3.0
        self.comps.append({'t': self.t, 'after_llm_call_index': self.n, 'summary': 'Summary', 'events_in_window': 9})
        tail = self.messages[-2:]
        sys_msg = dict(self.messages[0])
        user = {'role': 'user', 'content': 'Summary of earlier work: the agent read files.'}
        if keep_problem:
            user['content'] += f'\nProblem Statement:\n{self.problem}'
        self.messages = [sys_msg, user] + tail

    def text_turn(self, content: str, *, tokens: int = 50, finish: str = 'stop') -> None:
        self.turn([], tokens=tokens, finish=finish, content=content)

    def finish(self, *, status: str = 'done', empty_patch: bool = False, termination: str = 'submitted',
               patch_chars: int | None = None, extra: dict | None = None, final_text: bool = True) -> None:
        last = self.recs[-1] if self.recs else None
        if final_text and last and last['kind'] == 'agent' and \
                last['response']['choices'][0]['message'].get('tool_calls'):
            self.text_turn('Done.', tokens=20, finish='stop')  # the model sees the last tool results, then answers
        with open(self.dir / 'llm_calls.jsonl', 'w') as f:
            for r in self.recs:
                f.write(json.dumps(r) + '\n')
        if self.comps:
            with open(self.dir / 'compactions.jsonl', 'w') as f:
                for c in self.comps:
                    f.write(json.dumps(c) + '\n')
        res = {'status': status, 'instance_id': self.iid, 'empty_patch': empty_patch, 'termination': termination,
               'patch_chars': 0 if empty_patch else (patch_chars if patch_chars is not None else 500),
               'nudges': self.nudges, 'timings': {'agent_s': round(self.agent_s, 1)}, **(extra or {})}
        (self.dir / 'result.json').write_text(json.dumps(res))


def write_run(root: Path, name: str, tasks: list[str], grades: dict[str, bool] | None = None) -> Path:
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    (run / 'run.json').write_text(json.dumps({'name': name, 'task_ids': tasks, 'data': {}}))
    if grades is not None:
        (run / 'grades.jsonl').write_text(''.join(json.dumps({'instance_id': k, 'passed': v}) + '\n'
                                                  for k, v in grades.items()))
    return run
