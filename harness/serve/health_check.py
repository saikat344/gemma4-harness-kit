"""Health check for a live vLLM endpoint: tool calling, reasoning parsing, LoRA routing, context, speed.

    python -m harness.serve.health_check --endpoint http://127.0.0.1:8000/v1 [--long] [--bench 2] [--json out.json]

Stdlib only (runs from any venv). Requests mirror what the runner sends through ADK LiteLlm: OpenAI
`tools` with swegemma's tool schemas, `chat_template_kwargs.enable_thinking` and `thinking_token_budget`
in the body, temperature/top_p from the sample submission. Exit code 0 only if every required check passes.

Checks
  models       /health, /v1/models: one base model, LoRA modules listed, max_model_len == 32768
  tool_call    thinking ON: message.tool_calls parsed (read_file, JSON args with filepath), no raw
               `<|tool_call>` / `<|channel>` markup leaking into content, reasoning_content non-empty
  tool_turn    multi-turn: assistant tool call (+ reasoning mirrored as `reasoning`, as adk_submission's
               litellm patch does) + tool result -> next turn parses (tool call or text)
  no_think     enable_thinking=false: no reasoning_content
  think_budget thinking_token_budget=64 respected (reasoning tokens <= ~64 + slack) [warn only]
  loras        each LoRA id answers a short completion
  long         (--long) ~28k-token prompt + 512 output fits in 32k; reports prefill seconds
  bench        (--bench N) N parallel requests, decode tokens/s per request and aggregate
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'read_file',
        'description': 'Read the contents of a file from the repository workspace.',
        'parameters': {'type': 'object', 'properties': {
            'filepath': {'type': 'string', 'description': 'Relative path within the workspace directory.'},
            'start_line': {'type': 'integer', 'description': 'Optional starting line number (1-indexed).'},
            'end_line': {'type': 'integer', 'description': 'Optional ending line number (1-indexed).'}},
            'required': ['filepath']}}},
    {'type': 'function', 'function': {
        'name': 'edit_file',
        'description': 'Replace a contiguous block of text in an existing file.',
        'parameters': {'type': 'object', 'properties': {
            'filepath': {'type': 'string'}, 'old_string': {'type': 'string'}, 'new_string': {'type': 'string'},
            'allow_multiple': {'type': 'boolean'}},
            'required': ['filepath', 'old_string', 'new_string']}}},
    {'type': 'function', 'function': {
        'name': 'submit_patch',
        'description': 'Capture the current working tree modifications as the agent submission.',
        'parameters': {'type': 'object', 'properties': {}}}},
]
SYSTEM = 'You are a software engineer working in /workspace. Use the tools. Be brief.'
USER = ('The function `parse_header` in `pkg/headers.py` crashes on empty input. '
        'Start by reading that file with the read_file tool.')
LEAK_MARKERS = ('<|tool_call>', '<tool_call|>', '<|channel>', '<channel|>', '<|"|>')


class Check:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: Any = '', required: bool = True) -> None:
        self.results.append({'check': name, 'ok': bool(ok), 'required': required, 'detail': detail})
        tag = 'PASS' if ok else ('FAIL' if required else 'WARN')
        print(f'[{tag}] {name}: {detail if isinstance(detail, str) else json.dumps(detail)[:400]}', flush=True)

    @property
    def ok(self) -> bool:
        return all(r['ok'] for r in self.results if r['required'])


def http(url: str, payload: dict | None = None, timeout: float = 600, key: str = 'EMPTY') -> tuple[int, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method='POST' if data else 'GET')
    req.add_header('Authorization', f'Bearer {key}')
    if data:
        req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode()
            return r.status, (json.loads(body) if body.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors='replace')[:2000]


def chat(ep: str, model: str, messages: list, *, tools: list | None = TOOLS, thinking: bool = True,
         budget: int | None = 1024, max_tokens: int = 2048, key: str = 'EMPTY') -> tuple[int, Any, float]:
    body: dict[str, Any] = {'model': model, 'messages': messages, 'temperature': 0.2, 'top_p': 0.95,
                            'max_completion_tokens': max_tokens,
                            'chat_template_kwargs': {'enable_thinking': thinking}}
    if tools:
        body['tools'] = tools
    if thinking and budget:
        body['thinking_token_budget'] = budget
    t0 = time.perf_counter()
    code, resp = http(ep + '/chat/completions', body, key=key)
    return code, resp, time.perf_counter() - t0


def _msg(resp: Any) -> dict[str, Any]:
    return (resp.get('choices') or [{}])[0].get('message') or {} if isinstance(resp, dict) else {}


def _reasoning(msg: dict[str, Any]) -> str:
    return msg.get('reasoning_content') or msg.get('reasoning') or ''


def run(args: argparse.Namespace) -> Check:
    ep = args.endpoint.rstrip('/')
    c = Check()
    root = ep[:-3] if ep.endswith('/v1') else ep
    code, _ = http(root + '/health', timeout=10)
    c.add('health', code == 200, f'HTTP {code}')
    code, models = http(ep + '/models', timeout=10, key=args.api_key)
    if code != 200:
        c.add('models', False, f'HTTP {code}: {models}')
        return c
    data = models.get('data', [])
    base = [m for m in data if not m.get('parent')]
    loras = [m['id'] for m in data if m.get('parent')]
    mml = base[0].get('max_model_len') if base else None
    c.add('models', len(base) == 1, {'base': [m['id'] for m in base], 'loras': loras, 'max_model_len': mml})
    c.add('max_model_len_32768', mml == 32768, f'max_model_len={mml} (scorer 32768)', required=not args.allow_short)
    model = args.model or (base[0]['id'] if base else '')

    # 1. tool call with thinking on
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': USER}]
    code, resp, dt = chat(ep, model, msgs, key=args.api_key)
    msg = _msg(resp)
    tcs = msg.get('tool_calls') or []
    content = msg.get('content') or ''
    reasoning = _reasoning(msg)
    args_ok = False
    if tcs:
        try:
            a = json.loads(tcs[0]['function']['arguments'])
            args_ok = isinstance(a, dict) and isinstance(a.get('filepath'), str)
        except Exception:
            args_ok = False
    usage = resp.get('usage', {}) if isinstance(resp, dict) else {}
    c.add('tool_call', code == 200 and bool(tcs) and tcs[0]['function']['name'] == 'read_file' and args_ok,
          {'http': code, 'tool_calls': [t['function'] for t in tcs][:2], 'finish': (resp.get('choices') or [{}])[0].get('finish_reason') if isinstance(resp, dict) else None,
           'seconds': round(dt, 2), 'usage': usage} if code == 200 else f'HTTP {code}: {resp}')
    leaks = [m for m in LEAK_MARKERS if m in content or m in reasoning]
    c.add('no_markup_leak', not leaks, f'leaked markers: {leaks}' if leaks else 'content/reasoning clean')
    c.add('reasoning_parsed', bool(reasoning.strip()), f'{len(reasoning)} chars reasoning_content')

    # 2. multi-turn after a tool result
    if tcs:
        asst = {'role': 'assistant', 'content': content or None, 'tool_calls': tcs[:1]}
        if reasoning:
            asst['reasoning'] = reasoning  # adk_submission install_litellm_reasoning_patch mirrors this
        tool_msg = {'role': 'tool', 'tool_call_id': tcs[0].get('id', 'call_0'),
                    'content': json.dumps({'status': 'ok', 'filepath': 'pkg/headers.py',
                                           'content': 'def parse_header(value):\n    return value.split(";")[0].strip()\n'})}
        code2, resp2, dt2 = chat(ep, model, [*msgs, asst, tool_msg], key=args.api_key)
        m2 = _msg(resp2)
        ok2 = code2 == 200 and (bool(m2.get('tool_calls')) or bool((m2.get('content') or '').strip()))
        c.add('tool_turn', ok2, {'http': code2, 'tool_calls': [t['function']['name'] for t in m2.get('tool_calls') or []],
                                 'content': (m2.get('content') or '')[:120], 'seconds': round(dt2, 2)}
              if code2 == 200 else f'HTTP {code2}: {resp2}')

    # 3. thinking off
    code, resp, _ = chat(ep, model, msgs, thinking=False, key=args.api_key)
    m3 = _msg(resp)
    c.add('no_think', code == 200 and not _reasoning(m3).strip(),
          {'http': code, 'reasoning_chars': len(_reasoning(m3)), 'tool_calls': bool(m3.get('tool_calls'))})

    # 4. thinking budget (warn only)
    code, resp, _ = chat(ep, model, [{'role': 'user', 'content': 'Think carefully: how many primes are below 200? '
                                      'Then answer with just the number.'}], tools=None, budget=64, key=args.api_key)
    if code == 200:
        u = resp.get('usage') or {}
        rt = ((u.get('completion_tokens_details') or {}).get('reasoning_tokens'))
        r_chars = len(_reasoning(_msg(resp)))
        est = rt if rt is not None else r_chars // 4
        c.add('think_budget', est <= 64 + 32, f'reasoning ~{est} tokens with thinking_token_budget=64', required=False)
    else:
        c.add('think_budget', False, f'HTTP {code}: {resp}', required=False)

    # 5. LoRAs
    for lora in loras:
        code, resp, dt = chat(ep, lora, msgs, key=args.api_key, max_tokens=256)
        c.add(f'lora:{lora}', code == 200, f'HTTP {code} in {dt:.1f}s' if code == 200 else f'HTTP {code}: {resp}')

    # 6. long context
    if args.long:
        filler = ('def f_{i}(x):\n    return x + {i}\n\n')
        text = ''.join(filler.format(i=i) for i in range(1450))  # ~28k tokens: measured 18.8 tokens/function with vLLM /tokenize (2026-11-02)
        lm = [{'role': 'user', 'content': 'Here is a file:\n' + text + '\nWhat does f_7 return for x=1? Answer briefly.'}]
        code, resp, dt = chat(ep, model, lm, tools=None, thinking=False, max_tokens=512, key=args.api_key)
        pt = (resp.get('usage') or {}).get('prompt_tokens') if code == 200 else None
        c.add('long_context', code == 200, {'http': code, 'prompt_tokens': pt, 'seconds': round(dt, 1)}
              if code == 200 else f'HTTP {code}: {str(resp)[:300]}')

    # 7. throughput
    if args.bench:
        def one(_: int) -> tuple[int, int, float]:
            code, resp, dt = chat(ep, model, [{'role': 'user', 'content': 'Write a 400-word explanation of '
                                              'Python context managers.'}], tools=None, thinking=False,
                                  max_tokens=600, key=args.api_key)
            return code, ((resp.get('usage') or {}).get('completion_tokens') or 0) if code == 200 else 0, dt
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(args.bench) as pool:
            res = list(pool.map(one, range(args.bench)))
        wall = time.perf_counter() - t0
        toks = sum(r[1] for r in res)
        c.add('bench', all(r[0] == 200 for r in res),
              {'parallel': args.bench, 'per_request_tok_s': [round(r[1] / r[2], 1) for r in res if r[2]],
               'aggregate_tok_s': round(toks / wall, 1)}, required=False)
    return c


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--endpoint', default='http://127.0.0.1:8000/v1')
    ap.add_argument('--api-key', default=os.environ.get('LOCAL_API_KEY', 'EMPTY'),
                    help='default $LOCAL_API_KEY (same variable as the runner)')
    ap.add_argument('--model', help='base model id (default: discovered)')
    ap.add_argument('--long', action='store_true', help='also send a ~28k-token prompt')
    ap.add_argument('--bench', type=int, default=0, help='N parallel decode requests')
    ap.add_argument('--allow-short', action='store_true', help='do not fail when max_model_len != 32768')
    ap.add_argument('--json', help='write results to this file')
    args = ap.parse_args(argv)
    c = run(args)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'endpoint': args.endpoint, 'ok': c.ok, 'results': c.results}, f, indent=2)
    print('HEALTH', 'OK' if c.ok else 'FAILED')
    return 0 if c.ok else 1


if __name__ == '__main__':
    sys.exit(main())
