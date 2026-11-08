"""Mock OpenAI-compatible server that plays a scripted Gemma-style agent (no GPU needed).

    python -m harness.runner.mock_openai --port 8011 --tasks data/kaggle/tasks.jsonl --mode oracle

It answers /health, /v1/models and /v1/chat/completions the way vLLM does *after* the gemma4 tool and
reasoning parsers: `message.tool_calls` (OpenAI format), `message.reasoning_content`, usage.

Modes (per request; the step is derived from the conversation, so the server is stateless):
  oracle  identify the task from the problem statement in the prompt, then: run_command + read_file
          (parallel), a malformed edit_file (missing new_string), an edit_file whose old_string does not
          match, one edit_file per gold-patch hunk (write_file for new files), submit_patch, final text.
          The resulting patch equals the gold patch, so a grader should resolve it.
  empty   only ever answers with text; exercises the nudge loop and NO_PATCH.
  probe   first tool call prints git status/diff and the installed starlette, then submit_patch (env-fix tests).
  rawtool emits an unparsed `<|tool_call>` in content with finish_reason=length (truncation nudge path).
  reliability  scripted failures for harness/analysis/reliability.py (docs/reliability.md): read; the same edit_file
          with mangled arg keys (`<|"|>,filepath`, as in report 747168) three times in a row (ADK rejects each with
          MissingArguments); an unknown tool; the same non-matching edit_file twice; the gold hunk applied, reverted,
          re-applied, reverted and applied again (A-B-A); one turn that reports 8192 completion tokens (runaway);
          then submit_patch. Needs a gold hunk, so it is for tasks with a non-new first file.
  replay  replay a fixture JSON (`--fixture`, a `parsed_tool_calls` list of turns): a read, then the fixture's recorded
          (post-parser) tool-call turns, then submit_patch. `$FILE`/`$OLD`/`$NEW` in a fixture are filled from the
          task's first gold hunk.
  loop    only the identical mangled edit_file, forever (a pure retry loop; ends on the turn/time budget).
  late_edit  for time checkpoints (harness/runner/checkpoints.py): a command, one edit per gold hunk, submit_patch,
          then write_file of a non-test scratch file AFTER the submission, then final text. So the working tree
          ends up differing from the (graded) submission.
Requests without tools that contain ADK's summarizer prompt get a short summary (compaction path).
Model ids: the base model id (--model-id) plus any --lora names; the request's `model` is echoed.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SUMMARIZER_MARKER = 'The following is a conversation history between a user and an AI agent'


# --------------------------------------------------------------------------------------------------
# gold patch -> edit steps
# --------------------------------------------------------------------------------------------------


_HUNK_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')


def parse_patch(patch: str) -> list[dict[str, Any]]:
    """Split a unified diff (with or without `diff --git` headers) into files:
    {path, new_file, deleted, hunks: [(old_text, new_text)]}. Hunk bodies are consumed by their @@ counts,
    so removed lines that start with '--' are not mistaken for file headers."""
    lines = patch.splitlines(keepends=True)
    files: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('diff --git '):
            m = re.match(r'diff --git a/(\S+) b/(\S+)', line)
            cur = {'path': m.group(2) if m else '', 'new_file': False, 'deleted': False, 'hunks': []}
            files.append(cur)
        elif line.startswith('--- ') and i + 1 < len(lines) and lines[i + 1].startswith('+++ '):
            old_p = line[4:].strip().split('\t')[0]
            new_p = lines[i + 1][4:].strip().split('\t')[0]
            path = new_p[2:] if new_p.startswith('b/') else (old_p[2:] if old_p.startswith('a/') else new_p)
            if cur is None or cur['hunks'] or (cur['path'] and cur['path'] != path):
                cur = {'path': path, 'new_file': False, 'deleted': False, 'hunks': []}
                files.append(cur)
            cur['path'] = path if new_p != '/dev/null' else old_p[2:]
            cur['new_file'] = cur['new_file'] or old_p == '/dev/null'
            cur['deleted'] = cur['deleted'] or new_p == '/dev/null'
            i += 1
        elif cur is not None and line.startswith('new file mode'):
            cur['new_file'] = True
        elif cur is not None and line.startswith('deleted file mode'):
            cur['deleted'] = True
        elif cur is not None and (m := _HUNK_RE.match(line)):
            n_old = int(m.group(2) or 1)
            n_new = int(m.group(4) or 1)
            old: list[str] = []
            new: list[str] = []
            i += 1
            while i < len(lines) and (n_old > 0 or n_new > 0 or lines[i].startswith('\\')):
                ln = lines[i]
                tag, body = ln[:1], ln[1:]
                if tag == '\\':  # "\ No newline at end of file"
                    for side in (old, new):
                        if side and side[-1].endswith('\n'):
                            side[-1] = side[-1][:-1]
                elif tag == '-':
                    old.append(body)
                    n_old -= 1
                elif tag == '+':
                    new.append(body)
                    n_new -= 1
                else:  # context (' ' or an empty line)
                    old.append(body if tag == ' ' else '\n')
                    new.append(body if tag == ' ' else '\n')
                    n_old -= 1
                    n_new -= 1
                i += 1
            cur['hunks'].append((''.join(old), ''.join(new)))
            continue
        i += 1
    return files


def oracle_plan(task: dict[str, Any]) -> list[list[tuple[str, dict[str, Any]]]]:
    """List of assistant turns; each turn is a list of (tool_name, args) calls."""
    files = parse_patch(task['patch'])
    first = next((f['path'] for f in files if not f['new_file']), files[0]['path'] if files else 'README.md')
    plan: list[list[tuple[str, dict[str, Any]]]] = [
        [('run_command', {'command': 'git status --short && ls'}),
         ('read_file', {'filepath': first, 'start_line': 1, 'end_line': 30})],
        [('edit_file', {'filepath': first, 'old_string': 'x'})],  # malformed: missing new_string
        [('edit_file', {'filepath': first, 'old_string': '<<this text is not in the file>>',
                        'new_string': 'y'})],
    ]
    for f in files:
        if f['deleted']:
            plan.append([('run_command', {'command': f'git rm -q {f["path"]}'})])
        elif f['new_file']:
            content = ''.join(n for _, n in f['hunks'])
            plan.append([('write_file', {'filepath': f['path'], 'content': content})])
        else:
            for old, new in f['hunks']:
                plan.append([('edit_file', {'filepath': f['path'], 'old_string': old, 'new_string': new})])
    plan.append([('get_status', {})])
    plan.append([('submit_patch', {})])
    return plan


MANGLED_ARG = '<|"|>,filepath'  # what vLLM's gemma4 parser produced in report 747168 (key swallowed a stray delimiter)


def _first_hunk(task: dict[str, Any]) -> tuple[str, str, str]:
    files = parse_patch(task['patch'])
    f = next((f for f in files if not f['new_file'] and not f['deleted'] and f['hunks']), None)
    if f is None:
        raise ValueError('reliability mode needs a task whose gold patch edits an existing file')
    old, new = f['hunks'][0]
    return f['path'], old, new


def reliability_plan(task: dict[str, Any]) -> list[list[tuple[str, dict[str, Any]]]]:
    path, old, new = _first_hunk(task)
    mangled = ('edit_file', {MANGLED_ARG: path, 'old_string': old, 'new_string': new})
    missing = ('edit_file', {'filepath': path, 'old_string': '<<this text is not in the file>>', 'new_string': 'y'})
    fwd = ('edit_file', {'filepath': path, 'old_string': old, 'new_string': new})
    back = ('edit_file', {'filepath': path, 'old_string': new, 'new_string': old})
    return [
        [('run_command', {'command': 'git status --short && ls'}), ('read_file', {'filepath': path, 'start_line': 1, 'end_line': 30})],
        [mangled], [mangled], [mangled],
        [('edit_files', {'filepath': path, 'old_string': old, 'new_string': new})],
        [missing], [missing],
        [fwd], [back], [fwd], [back], [fwd],
        [('submit_patch', {})],
    ]


def loop_plan(task: dict[str, Any]) -> list[list[tuple[str, dict[str, Any]]]]:
    path, old, new = _first_hunk(task)
    return [[('edit_file', {MANGLED_ARG: path, 'old_string': old, 'new_string': new})]] * 40




LATE_EDIT_FILE = 'zz_after_submit_scratch.py'


def late_edit_plan(task: dict[str, Any]) -> list[list[tuple[str, dict[str, Any]]]]:
    plan: list[list[tuple[str, dict[str, Any]]]] = [[('run_command', {'command': 'git status --short'})]]
    for f in parse_patch(task['patch']):
        if f['deleted']:
            plan.append([('run_command', {'command': f'git rm -q {f["path"]}'})])
        elif f['new_file']:
            plan.append([('write_file', {'filepath': f['path'], 'content': ''.join(n for _, n in f['hunks'])})])
        else:
            for old, new in f['hunks']:
                plan.append([('edit_file', {'filepath': f['path'], 'old_string': old, 'new_string': new})])
    plan.append([('submit_patch', {})])
    plan.append([('write_file', {'filepath': LATE_EDIT_FILE, 'content': 'x = 1  # written after submit_patch\n'})])
    return plan


def replay_plan(task: dict[str, Any], fixture: dict[str, Any]) -> list[list[tuple[str, dict[str, Any]]]]:
    """A replay fixture: `parsed_tool_calls` is a list of turns, each a list of
    {name, arguments}; placeholders `$FILE`, `$OLD`, `$NEW` (in keys and string values) come from the first gold hunk."""
    path, old, new = _first_hunk(task)
    subs = {'$FILE': path, '$OLD': old, '$NEW': new}

    def fill(x: Any) -> Any:
        if isinstance(x, str):
            return subs.get(x, x)
        if isinstance(x, dict):
            return {fill(k): fill(v) for k, v in x.items()}
        if isinstance(x, list):
            return [fill(v) for v in x]
        return x

    plan: list[list[tuple[str, dict[str, Any]]]] = [[('read_file', {'filepath': path, 'start_line': 1, 'end_line': 30})]]
    for turn in fixture.get('parsed_tool_calls') or []:
        plan.append([(c['name'], fill(c.get('arguments') or {})) for c in turn])
    plan.append([('submit_patch', {})])
    return plan


PROBE_COMMAND = (
    'echo GIT_STATUS_BEGIN; git status --porcelain; echo GIT_STATUS_END; '
    'echo HEAD_DIFF_BEGIN; git diff HEAD --stat; git diff --binary _swegemma_baseline --stat; echo HEAD_DIFF_END; '
    "python -c \"import importlib.metadata as m; print('STARLETTE_DIST=' + m.version('starlette'))\"; "
    "python -c \"import starlette; print('STARLETTE_IMPORT=' + starlette.__version__)\""
)


def probe_plan(task: dict[str, Any]) -> list[list[tuple[str, dict[str, Any]]]]:
    """Environment probe (mode 'probe'): the FIRST tool call reports git cleanliness and the installed starlette,
    then submit_patch with no edits. Used to check what the agent sees at the start (tests: env fixes)."""
    return [[('run_command', {'command': PROBE_COMMAND})], [('submit_patch', {})]]


# --------------------------------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------------------------------


class MockState:
    def __init__(self, tasks_path: Path | None, mode: str, model_id: str, loras: list[str], delay: float,
                 inflate_prompt_tokens: int = 0) -> None:
        self.mode = mode
        self.inflate = inflate_prompt_tokens
        self.progress: dict[str, int] = {}
        self.last_task: dict[str, Any] | None = None
        self.model_id = model_id
        self.loras = loras
        self.delay = delay
        self.tasks: list[dict[str, Any]] = []
        if tasks_path:
            self.tasks = [json.loads(ln) for ln in tasks_path.read_text().splitlines() if ln.strip()]
        self.fixture: dict[str, Any] | None = None  # mode 'replay' (see replay_plan)
        self.requests = 0
        self.lock = threading.Lock()
        self.log: list[dict[str, Any]] = []

    def find_task(self, text: str) -> dict[str, Any] | None:
        for t in self.tasks:
            ps = (t.get('problem_statement') or '').strip()
            if ps and ps[:300] in text:
                return t
        return None


def _text_of(msg: dict[str, Any]) -> str:
    c = msg.get('content')
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return ''.join(p.get('text', '') for p in c if isinstance(p, dict))
    return ''


def _approx_tokens(obj: Any) -> int:
    return max(1, len(json.dumps(obj, default=str)) // 4)


def respond(state: MockState, body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get('messages') or []
    tools = body.get('tools') or []
    all_text = '\n'.join(_text_of(m) for m in messages)
    content: str | None = None
    tool_calls: list[dict[str, Any]] = []
    reasoning = None
    finish = 'stop'

    if not tools and SUMMARIZER_MARKER in all_text:
        content = 'Summary: the agent inspected the repository and is applying the fix.'
    elif state.mode == 'empty':
        reasoning = 'I am not sure what to do.'
        content = 'I could not find the problem.'
    elif state.mode == 'rawtool':
        reasoning = 'Let me read the file.'
        content = '<|tool_call>call:read_file{filepath:<|"|>README.md'
        finish = 'length'
    else:
        task = state.find_task(all_text)
        if task is None and state.last_task is not None and tools:
            task = state.last_task  # after compaction the mock's fixed summary no longer quotes the problem statement
        if task is not None:
            state.last_task = task
        # step = assistant turns with tool calls so far; after compaction the history is shorter, so a
        # per-task high-water mark (reset when a fresh session starts) keeps the script moving forward.
        step = sum(1 for m in messages if m.get('role') == 'assistant' and m.get('tool_calls'))
        key = task['instance_id'] if task else None
        tool_names = [t.get('function', {}).get('name') for t in tools]
        with state.lock:
            if key and step == 0 and 'Summary: the agent' not in all_text:
                state.progress[key] = 0
            step = max(step, state.progress.get(key, 0)) if key else step
        if task is None and tools:
            # generic Gemma-like behaviour for unknown prompts (health check): read a file, then answer
            names = [t.get('function', {}).get('name') for t in tools]
            if not any(m.get('role') == 'tool' for m in messages) and 'read_file' in names:
                reasoning = 'I should read the file first.'
                tool_calls.append({'id': f'chatcmpl-tool-{uuid.uuid4().hex[:16]}', 'type': 'function',
                                   'function': {'name': 'read_file',
                                                'arguments': json.dumps({'filepath': 'pkg/headers.py'})}})
                finish = 'tool_calls'
            else:
                reasoning = 'The fix is clear.'
                content = 'Mock: done.'
        elif task is None:
            content = 'Mock: task not recognised from the prompt.'
        else:
            plan = (probe_plan(task) if state.mode == 'probe' else reliability_plan(task) if state.mode == 'reliability'
                    else loop_plan(task) if state.mode == 'loop'
                    else late_edit_plan(task) if state.mode == 'late_edit'
                    else replay_plan(task, state.fixture or {}) if state.mode == 'replay'
                    else oracle_plan(task))
            if step < len(plan):
                reasoning = f'Step {step + 1}/{len(plan)} for {task["instance_id"]}.'
                for name, args in plan[step]:
                    tool_calls.append({
                        'id': f'chatcmpl-tool-{uuid.uuid4().hex[:16]}',
                        'type': 'function',
                        'function': {'name': name, 'arguments': json.dumps(args)},
                    })
                finish = 'tool_calls'
                with state.lock:
                    state.progress[key] = step + 1
            else:
                content = 'I applied the fix and submitted the patch.'
    msg: dict[str, Any] = {'role': 'assistant', 'content': content}
    thinking_on = (body.get('chat_template_kwargs') or {}).get('enable_thinking', False)
    if reasoning and thinking_on:
        msg['reasoning_content'] = reasoning
    if tool_calls:
        msg['tool_calls'] = tool_calls
    prompt_tokens = _approx_tokens(messages) + _approx_tokens(tools) + (state.inflate if tools else 0)
    completion_tokens = _approx_tokens(msg)
    if state.mode == 'reliability' and tool_calls and step == 3:  # one runaway-looking turn (cap 8192)
        completion_tokens = 8192
    return {
        'id': f'chatcmpl-{uuid.uuid4().hex}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': body.get('model', state.model_id),
        'choices': [{'index': 0, 'message': msg, 'finish_reason': finish, 'logprobs': None}],
        'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
                  'total_tokens': prompt_tokens + completion_tokens,
                  'completion_tokens_details': {'reasoning_tokens': len(reasoning or '') // 4}},
    }


def make_handler(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            return

        def _send(self, code: int, obj: Any) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip('/') == '/health':
                self._send(200, {})
            elif self.path.rstrip('/') == '/v1/models':
                data = [{'id': state.model_id, 'object': 'model', 'root': state.model_id, 'parent': None,
                         'max_model_len': 32768}]
                data += [{'id': n, 'object': 'model', 'root': f'/adapters/{n}', 'parent': state.model_id}
                         for n in state.loras]
                self._send(200, {'object': 'list', 'data': data})
            else:
                self._send(404, {'error': 'not found'})

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.rstrip('/').endswith('/chat/completions'):
                self._send(404, {'error': 'not found'})
                return
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            model = body.get('model')
            if model not in [state.model_id, *state.loras]:
                self._send(404, {'error': {'message': f'The model `{model}` does not exist.', 'code': 404}})
                return
            if state.delay:
                time.sleep(state.delay)
            resp = respond(state, body)
            with state.lock:
                state.requests += 1
                state.log.append({'model': model, 'n_messages': len(body.get('messages') or []),
                                  'has_tools': bool(body.get('tools')),
                                  'extra': {k: body.get(k) for k in ('chat_template_kwargs', 'thinking_token_budget',
                                                                      'temperature', 'top_p', 'max_tokens',
                                                                      'max_completion_tokens') if k in body}})
            if body.get('stream'):
                self._send_stream(resp)
            else:
                self._send(200, resp)

        def _send_stream(self, resp: dict[str, Any]) -> None:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            msg = resp['choices'][0]['message']
            delta = {'role': 'assistant', 'content': msg.get('content') or ''}
            if msg.get('reasoning_content'):
                delta['reasoning_content'] = msg['reasoning_content']
            if msg.get('tool_calls'):
                delta['tool_calls'] = [{**tc, 'index': i} for i, tc in enumerate(msg['tool_calls'])]
            chunk = {**resp, 'object': 'chat.completion.chunk',
                     'choices': [{'index': 0, 'delta': delta, 'finish_reason': resp['choices'][0]['finish_reason']}]}
            self.wfile.write(f'data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n'.encode())

    return Handler


def serve(port: int, state: MockState, host: str = '127.0.0.1') -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


MODES = ['oracle', 'empty', 'rawtool', 'probe', 'reliability', 'loop', 'replay', 'late_edit']


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8011)
    ap.add_argument('--tasks', type=Path, default=Path('data/kaggle/tasks.jsonl'))
    ap.add_argument('--mode', choices=MODES, default='oracle')
    ap.add_argument('--model-id', default='/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
    ap.add_argument('--lora', action='append', default=[])
    ap.add_argument('--delay', type=float, default=0.0, help='seconds per completion (simulate decode time)')
    ap.add_argument('--inflate-prompt-tokens', type=int, default=0,
                    help='add N to reported prompt_tokens (>14336 triggers ADK token-threshold compaction)')
    ap.add_argument('--fixture', type=Path, help='fixture JSON for --mode replay (see replay_plan)')
    a = ap.parse_args()
    state = MockState(a.tasks if a.tasks.exists() else None, a.mode, a.model_id, a.lora, a.delay,
                      a.inflate_prompt_tokens)
    if a.fixture:
        state.fixture = json.loads(a.fixture.read_text())
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(state))
    print(f'mock OpenAI server on http://{a.host}:{a.port}/v1 mode={a.mode} model={a.model_id} loras={a.lora}')
    httpd.serve_forever()


if __name__ == '__main__':
    main()
