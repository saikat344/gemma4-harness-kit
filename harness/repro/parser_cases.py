"""Deterministic, CPU-only round-trip harness for vLLM 0.19.1's gemma4 tool-call and reasoning parsers.

Purpose (docs/reliability.md section 1, reproduction R1): community report 747168 says vLLM's gemma4 tool
parser mangles `edit_file` calls whose `old_string` holds `\"\"\"` docstrings: argument keys come out like
`<|"|>,filepath`, ADK rejects the call ("mandatory input parameters are not present") and the model retries
the identical call. This script feeds the parser synthetic raw model outputs in Gemma 4's native tool-call
format and checks that the parsed tool name and arguments round-trip exactly.

Native format (vllm/tool_parsers/gemma4_tool_parser.py, 0.19.1):

    <|tool_call>call:func_name{key:<|"|>string value<|"|>,num:42,flag:true}<tool_call|>

Strings are wrapped in the special token `<|"|>` (id 52); keys are bare; calls are concatenated with no
separator. Thinking is `<|channel>thought\\n...<channel|>`. The server runs the reasoning parser first and
the tool parser on the remaining content (vllm/entrypoints/openai/engine/serving.py).

Run on the Windows box inside the vLLM venv (needs a real `import vllm` and the tokenizer):

    ~/venvs/vllm/bin/python -m harness.repro.parser_cases --tokenizer <.../gemma-4-31b-it-qat-w4a16-ct/2> \\
        --json runs/repro/parser_cases.json --md runs/repro/parser_cases.md

Mac / CI (cannot import vllm):
    --vllm-src data/vllm-src/0.19.1   run the REAL parser source with stubbed vllm imports (fake tokenizer)
    --fake                            run a tiny built-in fake parser (tests of this harness only)
Both print "NEEDS WINDOWS RUN": only the real package plus the real tokenizer settles the question.

Case families: `canonical` cases are what the model is supposed to emit and MUST round-trip (a failure is a
parser bug, exit code 1). `glitch` cases simulate model-side format slips (stray delimiter, missing end tag,
...) and show what the parser does with them; those are expected to mis-parse (hypotheses H1-H5).
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import re
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

TOOL_CALL_START = '<|tool_call>'
TOOL_CALL_END = '<tool_call|>'
STR_DELIM = '<|"|>'
THINK_START = '<|channel>'
THINK_END = '<channel|>'

# Tool signatures of the swegemma tools (swegemma/tools/workspace.py, execution.py). ADK's FunctionTool
# rejects a call only when a REQUIRED parameter is absent; unknown keys are silently dropped.
TOOL_REQUIRED: dict[str, list[str]] = {
    'edit_file': ['filepath', 'old_string', 'new_string'],
    'write_file': ['filepath', 'content'],
    'run_command': ['command'],
}
TOOL_PARAMS: dict[str, list[str]] = {
    'edit_file': ['filepath', 'old_string', 'new_string', 'allow_multiple'],
    'write_file': ['filepath', 'content'],
    'run_command': ['command'],
}

Call = tuple[str, dict[str, Any]]


# --------------------------------------------------------------------------------------------------
# Rendering Gemma's native format
# --------------------------------------------------------------------------------------------------


def fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if v is None:
        return 'null'
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return f'{STR_DELIM}{v}{STR_DELIM}'
    if isinstance(v, dict):
        return '{' + ','.join(f'{k}:{fmt_value(x)}' for k, x in v.items()) + '}'
    if isinstance(v, (list, tuple)):
        return '[' + ','.join(fmt_value(x) for x in v) + ']'
    raise TypeError(type(v))


def render_call(name: str, args: dict[str, Any]) -> str:
    body = ','.join(f'{k}:{fmt_value(v)}' for k, v in args.items())
    return f'{TOOL_CALL_START}call:{name}{{{body}}}{TOOL_CALL_END}'


# --------------------------------------------------------------------------------------------------
# Test strings
# --------------------------------------------------------------------------------------------------

DOC3D_OLD = 'def f(x):\n    """Return x doubled.\n\n    Args:\n        x: a number\n    """\n    return x * 2\n'
DOC3D_NEW = 'def f(x):\n    """Return x tripled.\n\n    Args:\n        x: a number\n    """\n    return x * 3\n'
DOC3S_OLD = "def g():\n    '''Single-quoted docstring.'''\n    return 1\n"
DOC3S_NEW = "def g():\n    '''Single-quoted docstring, changed.'''\n    return 2\n"
DOC_END_OLD = '    """Docstring that ends the old_string."""'  # old_string ends with `"""` (H1 trigger)
DOC_END_NEW = '    """Docstring that ends the new_string, edited."""'
DOC_START = '"""Module docstring at the very start of old_string.\n\nMore lines.\n"""\nimport os\n'
DOC_ONLY = '""""""'
DOC_MIX = 'x = """a "quoted" \'word\' inside"""\ny = \'\'\'it\'s "fine"\'\'\'\n'
BS_PATH = 'C:\\Users\\dev\\new\\table.txt'
BS_REGEX = "pat = re.compile(r'\\d+\\s*\\.\\w+\\\\')\n"
BS_NEWLINE_TEXT = "print('line1\\nline2\\ttabbed')  # literal backslash-n in the source\nreal newline next\n"
UNICODE = 'h\u00e9llo w\u00f6rld \u2014 \u65e5\u672c\u8a9e \U0001f680 e\u0301 \u200b|\u202e|'
JSON_LIKE = '{"a": 1, "b": [1, 2, {"c": "d"}], "e": null, "f": true}'
BRACES = 'd = {"k": {"n": [1, {2: 3}]}}\nf"{x!r:>{width}}"\n{}\n{{}}\n}{ }}}\n'
MARKERS = 'a <| b |> c <|" d "|> e <||> <| |> |> <| <|"|'
LONG_13K = ''.join(f'    value_{i} = compute({i}, "arg_{i}")  # {"x" * 40}\n' for i in range(200))  # ~13 KB
LONG_DOCS = ('def h():\n    """Doc.\n\n    More.\n    """\n    pass\n\n') * 250  # ~10 KB of docstrings
LONG_LINE = 'a' * 60_000
LONG_200K = ''.join(f'line {i}: {"abc " * 12}\n' for i in range(4000))  # ~200 KB


def _edit(old: str, new: str, path: str = 'pkg/mod.py', **extra: Any) -> dict[str, Any]:
    return {'filepath': path, 'old_string': old, 'new_string': new, **extra}


def _write(content: str, path: str = 'pkg/new_mod.py') -> dict[str, Any]:
    return {'filepath': path, 'content': content}


def _cmd(command: str) -> dict[str, Any]:
    return {'command': command}


@dataclass
class Case:
    id: str
    family: str  # canonical | glitch
    group: str  # quoting, escapes, unicode, json, markers, long, multi, thinking, terminator, delimiter, ...
    calls: list[Call]  # what the parser SHOULD return (names + args)
    raw: str | None = None  # full model output; default: thinking + content + rendered calls
    thinking: str | None = None  # text inside <|channel>thought\n ... <channel|>
    text_before: str | None = None  # plain content before the first call
    expect: str = 'roundtrip'  # roundtrip | mangled (documented limitation / hypothesis)
    note: str = ''
    streaming: bool = True

    def build_raw(self) -> str:
        if self.raw is not None:
            return self.raw
        out = ''
        if self.thinking is not None:
            out += f'{THINK_START}thought\n{self.thinking}{THINK_END}'
        if self.text_before:
            out += self.text_before
        out += ''.join(render_call(n, a) for n, a in self.calls)
        return out

    @property
    def expected_reasoning(self) -> str | None:
        return self.thinking

    @property
    def expected_content(self) -> str | None:
        return self.text_before.strip() if self.text_before and self.text_before.strip() else None


def _tools_for(prefix: str, group: str, variants: dict[str, tuple[str, str]], **kw: Any) -> list[Case]:
    """For one string family make the edit_file / write_file / run_command trio. `variants[key]=(old,new)`."""
    cases: list[Case] = []
    for key, (old, new) in variants.items():
        cases.append(Case(f'{prefix}.{key}.edit_file', 'canonical', group, [('edit_file', _edit(old, new))], **kw))
        cases.append(Case(f'{prefix}.{key}.write_file', 'canonical', group, [('write_file', _write(new))], **kw))
        cases.append(Case(f'{prefix}.{key}.run_command', 'canonical', group,
                          [('run_command', _cmd(f"python - <<'EOF'\n{new}\nEOF"))], **kw))
    return cases


def build_cases(long_chars: int | None = None) -> list[Case]:
    cs: list[Case] = []

    # ---- docstrings --------------------------------------------------------------------------------
    cs += _tools_for('doc3dq', 'quoting', {
        'body': (DOC3D_OLD, DOC3D_NEW),
        'ends_with_quotes': (DOC_END_OLD, DOC_END_NEW),
        'starts_with_quotes': (DOC_START, DOC_START.replace('Module', 'Edited module')),
        'only_quotes': (DOC_ONLY, DOC_ONLY),
    })
    cs += _tools_for('doc3sq', 'quoting', {
        'body': (DOC3S_OLD, DOC3S_NEW),
        'ends_with_quotes': ("    '''ends here'''", "    '''ends here, edited'''"),
    })
    cs += _tools_for('quotes_mixed', 'quoting', {
        'both_in_triple': (DOC_MIX, DOC_MIX.replace('fine', 'good')),
        'dq_inside': ('say("hi") + \'x\'', 'say("bye") + \'y\''),
        'ends_with_dq': ('msg = "done"', 'msg = "finished"'),
        'starts_with_dq': ('"leading" quote', '"leading" quote!'),
        'single_dq_char': ('"', '""'),
        'single_sq_char': ("'", "''"),
    })
    # ---- escapes -----------------------------------------------------------------------------------
    cs += _tools_for('escapes', 'escapes', {
        'backslash_path': (BS_PATH, BS_PATH + '2'),
        'regex_backslashes': (BS_REGEX, BS_REGEX.replace('d+', 'd*')),
        'literal_backslash_n': (BS_NEWLINE_TEXT, BS_NEWLINE_TEXT + '# edited\n'),
        'trailing_backslash': ('x = "a" \\', 'x = "b" \\'),
        'backslash_quote': ('s = "say \\"hi\\""', 's = "say \\"bye\\""'),
        'crlf_tabs': ('a\r\n\tb\r\n', 'a\r\n\tc\r\n'),
    })
    # ---- unicode -----------------------------------------------------------------------------------
    cs += _tools_for('unicode', 'unicode', {
        'mixed': (UNICODE, UNICODE + '!'),
        'emoji_only': ('\U0001f680\U0001f680', '\U0001f4a5'),
    })
    # ---- JSON-looking, braces, special-looking sequences -------------------------------------------
    cs += _tools_for('jsonlike', 'json', {
        'json_text': (JSON_LIKE, JSON_LIKE.replace('1', '2')),
        'braces': (BRACES, BRACES + '# changed\n'),
        'only_braces': ('{}', '{ }'),
        'close_brace_end': ('x = {', 'x = {}'),
        'brace_then_tag_text': ('}<tool_call', '}<tool_call>'),
    })
    cs += _tools_for('markers', 'markers', {
        'pipe_angle_text': (MARKERS, MARKERS + ' extra |>'),
        'shift_ops': ('x = a << b | c >> d', 'x = a << b |> c'),
        'html_like': ('<div class="a">|</div>', '<div class="b">|</div>'),
    })
    # numbers/booleans that must stay strings, plus real typed args
    cs.append(Case('typed.string_lookalikes.edit_file', 'canonical', 'json', [
        ('edit_file', _edit('42', 'true')), ('edit_file', _edit('null', '3.5'))]))
    cs.append(Case('typed.allow_multiple_true.edit_file', 'canonical', 'json',
                   [('edit_file', _edit(DOC3D_OLD, DOC3D_NEW, allow_multiple=True))]))
    cs.append(Case('typed.allow_multiple_false.edit_file', 'canonical', 'json',
                   [('edit_file', _edit(DOC3D_OLD, DOC3D_NEW, allow_multiple=False))]))
    cs.append(Case('empty.new_string.edit_file', 'canonical', 'json', [('edit_file', _edit(DOC3D_OLD, ''))]))
    cs.append(Case('empty.content.write_file', 'canonical', 'json', [('write_file', _write(''))]))
    cs.append(Case('whitespace.only.edit_file', 'canonical', 'json', [('edit_file', _edit('  \n\t', ' '))]))
    cs.append(Case('whitespace.leading_trailing.run_command', 'canonical', 'json',
                   [('run_command', _cmd('  ls -la  \n'))]))
    cs.append(Case('keyorder.alphabetical.edit_file', 'canonical', 'json', [('edit_file', {
        'allow_multiple': False, 'filepath': 'pkg/mod.py', 'new_string': DOC3D_NEW, 'old_string': DOC3D_OLD})]))
    cs.append(Case('keyorder.old_first.edit_file', 'canonical', 'json', [('edit_file', {
        'old_string': DOC3D_OLD, 'new_string': DOC3D_NEW, 'filepath': 'pkg/mod.py'})]))

    # ---- long strings ------------------------------------------------------------------------------
    big = long_chars if long_chars else None
    long_13k = LONG_13K if not big else (LONG_13K * (big // len(LONG_13K) + 1))[:big]
    cs += _tools_for('long', 'long', {
        'code_13k': (long_13k, long_13k.replace('value_', 'val_')),
        'docstrings_10k': (LONG_DOCS, LONG_DOCS.replace('Doc.', 'Doc!')),
        'one_line_60k': (LONG_LINE, LONG_LINE + 'b'),
    })
    cs.append(Case('long.text_200k.write_file', 'canonical', 'long', [('write_file', _write(LONG_200K))]))
    cs.append(Case('long.multi_big.three_calls', 'canonical', 'long', [
        ('edit_file', _edit(LONG_13K, LONG_13K + '#')), ('write_file', _write(LONG_DOCS)),
        ('run_command', _cmd(LONG_13K[:5000]))]))

    # ---- multiple tool calls in one turn -------------------------------------------------------------
    two = [('edit_file', _edit(DOC3D_OLD, DOC3D_NEW)), ('edit_file', _edit(DOC3S_OLD, DOC3S_NEW, 'pkg/other.py'))]
    cs.append(Case('multi.two_edits_docstrings', 'canonical', 'multi', two))
    cs.append(Case('multi.edit_write_run', 'canonical', 'multi', [
        ('edit_file', _edit(DOC3D_OLD, DOC3D_NEW)), ('write_file', _write(DOC3S_NEW)),
        ('run_command', _cmd('python -m pytest -x -q tests/test_mod.py'))]))
    cs.append(Case('multi.same_call_twice', 'canonical', 'multi', [two[0], two[0]]))
    cs.append(Case('multi.five_calls', 'canonical', 'multi', [('run_command', _cmd(f'echo {i}')) for i in range(5)]))
    cs.append(Case('multi.text_then_two_calls', 'canonical', 'multi', two,
                   text_before='I will apply both edits now.\n'))
    cs.append(Case('multi.docstring_then_quote_edge', 'canonical', 'multi', [
        ('edit_file', _edit(DOC_END_OLD, DOC_END_NEW)), ('edit_file', _edit('"', "'", 'pkg/q.py'))]))

    # ---- tool call after thinking text ---------------------------------------------------------------
    edit = ('edit_file', _edit(DOC3D_OLD, DOC3D_NEW))
    cs.append(Case('thinking.then_edit', 'canonical', 'thinking', [edit],
                   thinking='The docstring says doubled but the code should triple.'))
    cs.append(Case('thinking.with_docstring_text', 'canonical', 'thinking', [edit],
                   thinking='I will replace """Return x doubled.""" with """Return x tripled."""; braces {} and <| |> too.'))
    cs.append(Case('thinking.looks_like_call', 'canonical', 'thinking', [edit],
                   thinking='Plan: call:edit_file{filepath:"pkg/mod.py"} then verify.'))
    cs.append(Case('thinking.text_then_edit', 'canonical', 'thinking', [edit],
                   thinking='Need to edit.', text_before='Applying the edit.\n'))
    cs.append(Case('thinking.then_three_calls', 'canonical', 'thinking', [
        edit, ('write_file', _write(DOC3S_NEW)), ('run_command', _cmd('pytest -q'))], thinking='Three steps.'))
    cs.append(Case('thinking.empty', 'canonical', 'thinking', [edit], thinking=''))
    cs.append(Case('thinking.multiline_long', 'canonical', 'thinking', [edit],
                   thinking='\n'.join(f'step {i}: consider """x"""' for i in range(300))))
    cs.append(Case('thinking.no_start_token', 'canonical', 'thinking', [edit],
                   raw=f'thought\nplanning{THINK_END}' + render_call(*edit), thinking='planning',
                   note='some prompts may put <|channel> in the prompt; the parser still splits on <channel|>'))

    # ---- glitches: model-side format slips (expected to mis-parse; see docs/reliability.md H1-H5) ------
    old_dq, new_dq = DOC_END_OLD, DOC_END_NEW
    e_args = _edit(old_dq, new_dq)

    def call_raw(body: str) -> str:
        return f'{TOOL_CALL_START}call:edit_file{{{body}}}{TOOL_CALL_END}'

    D = STR_DELIM
    cs.append(Case(
        'glitch.H1.extra_delim_after_trailing_quote', 'glitch', 'delimiter', [('edit_file', e_args)],
        raw=call_raw(f'old_string:{D}{old_dq[:-1]}{D}{D},filepath:{D}pkg/mod.py{D},new_string:{D}{new_dq}{D}'),
        expect='mangled',
        note='R1 signature: value ends in `""` + `"` merged into the closing delimiter, giving `<|"|><|"|>` '
             '-> next key is `<|"|>,filepath`'))
    cs.append(Case(
        'glitch.H1.extra_delim_old_string_first', 'glitch', 'delimiter', [('edit_file', e_args)],
        raw=call_raw(f'filepath:{D}pkg/mod.py{D},old_string:{D}{old_dq[:-1]}{D}{D},new_string:{D}{new_dq}{D}'),
        expect='mangled', note='same slip, filepath first: then new_string is lost instead'))
    cs.append(Case(
        'glitch.H2.inner_quote_as_delim', 'glitch', 'delimiter', [('edit_file', e_args)],
        raw=call_raw(f'filepath:{D}pkg/mod.py{D},old_string:{D}    {D}""Docstring that ends the old_string.""{D}'
                     f',new_string:{D}{new_dq}{D}'),
        expect='mangled', note='odd number of delimiters: a `"` of the docstring is emitted as `<|"|>`'))
    cs.append(Case(
        'glitch.H2.plain_quote_closes', 'glitch', 'delimiter', [('edit_file', e_args)],
        raw=call_raw(f'filepath:{D}pkg/mod.py{D},old_string:{D}{old_dq}",new_string:{D}{new_dq}{D}'),
        expect='mangled', note='closing delimiter replaced by a plain `"`: string runs on into the next keys'))
    cs.append(Case(
        'glitch.H2.json_escaped_quotes', 'glitch', 'delimiter', [('edit_file', e_args)],
        raw=call_raw(f'filepath:{D}pkg/mod.py{D},old_string:{D}' + old_dq.replace('"', '\\"') +
                     f'{D},new_string:{D}{new_dq}{D}'),
        expect='mangled', note='model JSON-escapes the docstring quotes: backslashes survive into the value'))
    cs.append(Case(
        'glitch.H2.json_style_args', 'glitch', 'delimiter', [('edit_file', e_args)],
        raw=call_raw('"filepath":"pkg/mod.py","old_string":"' + old_dq.replace('"', '\\"') + '","new_string":"x"'),
        expect='mangled', note='JSON object inside the braces instead of key:<|"|>v<|"|>'))
    cs.append(Case(
        'glitch.H3.turn_end_instead_of_tool_call_end', 'glitch', 'terminator', [('edit_file', e_args)],
        raw=f'{TOOL_CALL_START}call:edit_file{{filepath:{D}pkg/mod.py{D},old_string:{D}{old_dq}{D},'
            f'new_string:{D}{new_dq}{D}}}<turn|>',
        expect='mangled', note='gemma4_utils.parse_tool_calls tolerates <turn|>; the server regex does not'))
    cs.append(Case(
        'glitch.H3.tool_response_instead_of_end', 'glitch', 'terminator', [('edit_file', e_args)],
        raw=f'{TOOL_CALL_START}call:edit_file{{filepath:{D}pkg/mod.py{D},old_string:{D}{old_dq}{D},'
            f'new_string:{D}{new_dq}{D}}}<|tool_response>',
        expect='mangled', note='model skips <tool_call|> and emits <|tool_response> directly'))
    cs.append(Case(
        'glitch.H3.whitespace_before_end', 'glitch', 'terminator', [('edit_file', e_args)],
        raw=f'{TOOL_CALL_START}call:edit_file{{filepath:{D}pkg/mod.py{D},old_string:{D}{old_dq}{D},'
            f'new_string:{D}{new_dq}{D}}}\n{TOOL_CALL_END}',
        expect='mangled', note='newline between `}` and <tool_call|> defeats the regex'))
    cs.append(Case(
        'glitch.H3.space_before_brace', 'glitch', 'terminator', [('edit_file', e_args)],
        raw=f'{TOOL_CALL_START}call:edit_file {{filepath:{D}pkg/mod.py{D},old_string:{D}{old_dq}{D},'
            f'new_string:{D}{new_dq}{D}}}{TOOL_CALL_END}',
        expect='mangled', note='`call:name {` with a space does not match the regex'))
    cs.append(Case(
        'glitch.H5.truncated_by_max_tokens', 'glitch', 'terminator', [('write_file', _write(LONG_13K))],
        raw=f'{TOOL_CALL_START}call:write_file{{filepath:{D}pkg/new_mod.py{D},content:{D}' + LONG_13K[:5000],
        expect='mangled', note='finish_reason=length mid-string: no end tag, content keeps raw <|tool_call> text '
                               '(the runner counts these as unparsed_tool_call_text and nudges)'))
    cs.append(Case(
        'glitch.H4.call_inside_unterminated_thinking', 'glitch', 'thinking', [edit],
        raw=f'{THINK_START}thought\nI will edit now.' + render_call(*edit), thinking=None,
        expect='mangled', note='tool call emitted before <channel|>: reasoning parser swallows everything'))
    cs.append(Case(
        'glitch.collision.delim_in_value', 'glitch', 'collision',
        [('edit_file', _edit('x = "<|"|>"', 'x = "<|"|>!"'))], expect='mangled',
        note='a value that contains the literal string-delimiter token cannot be represented'))
    cs.append(Case(
        'glitch.collision.end_tag_in_value', 'glitch', 'collision',
        [('edit_file', _edit('s = "}<tool_call|>"', 's = "}<tool_call|>!"'))], expect='mangled',
        note='`}<tool_call|>` inside a value: non-greedy regex ends the call early'))
    cs.append(Case(
        'glitch.collision.start_tag_in_value', 'glitch', 'collision',
        [('edit_file', _edit('s = "<|tool_call>call:x{}"', 's = "<|tool_call>call:y{}"'))],
        note='a source file that mentions the literal tool-call tags: passes non-streaming (lucky), streaming mangles'))
    return cs


# --------------------------------------------------------------------------------------------------
# Parser environment
# --------------------------------------------------------------------------------------------------


@dataclass
class Env:
    tool_cls: type
    reasoning_cls: type | None
    tokenizer: Any
    info: dict[str, Any]
    real_tokenizer: bool
    needs_windows_run: bool
    split: Callable[[str], list[tuple[str, int]]] = field(default=lambda s: [])

    def new_tool_parser(self) -> Any:
        return self.tool_cls(self.tokenizer, None)

    def new_reasoning_parser(self) -> Any:
        return self.reasoning_cls(self.tokenizer) if self.reasoning_cls else None


class EnvError(RuntimeError):
    pass


def default_tokenizer_dir() -> Path | None:
    cands = [os.environ.get('GEMMA4_TOKENIZER_DIR'), os.environ.get('GEMMA4_MODEL_DIR')]
    cands += [str(p) for p in Path('~/models').expanduser().glob('**/gemma-4-31b-it-qat-w4a16-ct/2')]
    for c in cands:
        if c and (Path(c) / 'tokenizer_config.json').exists():
            return Path(c)
    return None


def _load_real_tokenizer(path: Path) -> Any:
    try:
        from vllm.tokenizers import get_tokenizer

        return get_tokenizer(str(path))
    except Exception as e:  # fall back to transformers (same underlying tokenizer files)
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(str(path))
        except Exception as e2:
            raise EnvError(f'cannot load tokenizer from {path}: vllm get_tokenizer: {e!r}; '
                           f'transformers: {e2!r}') from e2


def _real_splitter(tok: Any) -> Callable[[str], list[tuple[str, int]]]:
    """raw text -> [(delta_text, token_id)], an incremental detokenizer in the style of vLLM's
    (windowed decode, holds back incomplete UTF-8, skip_special_tokens=False)."""

    def split(raw: str) -> list[tuple[str, int]]:
        ids = tok.encode(raw, add_special_tokens=False)
        out: list[tuple[str, int]] = []
        prefix_off = read_off = 0
        for i in range(len(ids)):
            prefix_text = tok.decode(ids[prefix_off:read_off], skip_special_tokens=False)
            new_text = tok.decode(ids[prefix_off:i + 1], skip_special_tokens=False)
            if len(new_text) > len(prefix_text) and not new_text.endswith('\ufffd'):
                out.append((new_text[len(prefix_text):], ids[i]))
                prefix_off, read_off = read_off, i + 1
            else:
                out.append(('', ids[i]))
        return out

    return split


def load_env(args: argparse.Namespace) -> Env:
    if args.fake:
        from harness.repro.vllm_stub import FakeTokenizer

        tok = FakeTokenizer(chunk=args.chunk)
        return Env(FakeToolParser, FakeReasoningParser, tok, {'source': 'built-in FAKE parser (harness self-test)'},
                   False, True, lambda s: list(zip(tok.pieces(s), tok.token_ids(s))))
    if args.vllm_src:
        from harness.repro.vllm_stub import FakeTokenizer, load_gemma4_parsers

        tcls, rcls, info = load_gemma4_parsers(args.vllm_src)
        if args.tokenizer:
            tok = _load_real_tokenizer(Path(args.tokenizer))
            return Env(tcls, rcls, tok, info, True, True, _real_splitter(tok))
        tok = FakeTokenizer(chunk=args.chunk)
        return Env(tcls, rcls, tok, info, False, True, lambda s: list(zip(tok.pieces(s), tok.token_ids(s))))
    try:
        import vllm  # noqa: F401
        from vllm.reasoning import ReasoningParserManager
        from vllm.tool_parsers import ToolParserManager
    except Exception as e:
        raise EnvError(
            f'cannot import vllm ({type(e).__name__}: {e}).\n'
            '  This harness must run in the vLLM venv on the Windows/WSL box:\n'
            '    ~/venvs/vllm/bin/python -m harness.repro.parser_cases --tokenizer <.../gemma-4-31b-it-qat-w4a16-ct/2>\n'
            '  On the Mac use --vllm-src data/vllm-src/0.19.1 (real parser source, stubbed imports, fake tokenizer)\n'
            '  or --fake (tiny built-in parser). Both are marked NEEDS WINDOWS RUN.') from e
    tdir = Path(args.tokenizer) if args.tokenizer else default_tokenizer_dir()
    if tdir is None or not tdir.exists():
        raise EnvError('--tokenizer <dir> is required (a directory with tokenizer.json / tokenizer_config.json, e.g. '
                       '~/models/.../gemma-4-31b-it-qat-w4a16-ct/2); none given and no default found '
                       '(set GEMMA4_TOKENIZER_DIR to avoid the flag)')
    tok = _load_real_tokenizer(tdir)
    tcls = ToolParserManager.get_tool_parser('gemma4')
    rcls = ReasoningParserManager.get_reasoning_parser('gemma4')
    info = {'source': 'real vllm package', 'vllm_version': getattr(vllm, '__version__', '?'),
            'tool_parser': f'{tcls.__module__}.{tcls.__name__}',
            'reasoning_parser': f'{rcls.__module__}.{rcls.__name__}', 'tokenizer_dir': str(tdir)}
    return Env(tcls, rcls, tok, info, True, False, _real_splitter(tok))


# --------------------------------------------------------------------------------------------------
# Tiny fake parser (harness self-test only; deliberately simple, NOT vLLM's code)
# --------------------------------------------------------------------------------------------------


class FakeToolParser:
    """Same wire format and regex as vLLM's, with a minimal argument reader (strings, bare scalars)."""

    _re = re.compile(r'<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>', re.DOTALL)

    def __init__(self, tokenizer: Any, tools: Any = None) -> None:
        self.tok = tokenizer
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []

    @staticmethod
    def parse_args(s: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        i, n = 0, len(s)
        while i < n:
            while i < n and s[i] in ' ,\n\t':
                i += 1
            j = s.find(':', i)
            if j < 0:
                break
            key, i = s[i:j].strip(), j + 1
            if s.startswith(STR_DELIM, i):
                e = s.find(STR_DELIM, i + len(STR_DELIM))
                if e < 0:
                    out[key] = s[i + len(STR_DELIM):]
                    break
                out[key] = s[i + len(STR_DELIM):e]
                i = e + len(STR_DELIM)
            else:
                e = i
                while e < n and s[e] != ',':
                    e += 1
                v = s[i:e].strip()
                out[key] = True if v == 'true' else False if v == 'false' else int(v) if v.lstrip('-').isdigit() else v
                i = e
        return out

    def extract_tool_calls(self, model_output: str, request: Any) -> Any:
        ms = self._re.findall(model_output)
        calls = [types.SimpleNamespace(function=types.SimpleNamespace(
            name=n, arguments=json.dumps(self.parse_args(a), ensure_ascii=False))) for n, a in ms]
        content = None
        if calls:
            c = model_output[:model_output.find(TOOL_CALL_START)].strip()
            content = c or None
        else:
            content = model_output
        return types.SimpleNamespace(tools_called=bool(calls), tool_calls=calls, content=content)

    def extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str,
                                     previous_token_ids: Any, current_token_ids: Any, delta_token_ids: Any,
                                     request: Any) -> Any:
        # Non-incremental: emit the whole call once, on the delta that closes it.
        if TOOL_CALL_END in delta_text or (TOOL_CALL_END in current_text and TOOL_CALL_END not in previous_text):
            before = len(self._re.findall(previous_text))
            ms = self._re.findall(current_text)
            if len(ms) > before:
                tcs = [types.SimpleNamespace(index=before + k, id=f'c{k}', function={'name': n, 'arguments':
                       json.dumps(self.parse_args(a), ensure_ascii=False)}) for k, (n, a) in enumerate(ms[before:])]
                return types.SimpleNamespace(content=None, tool_calls=tcs)
        if TOOL_CALL_START not in current_text and delta_text:
            return types.SimpleNamespace(content=delta_text, tool_calls=[])
        return None


class FakeReasoningParser:
    def __init__(self, tokenizer: Any, *a: Any, **k: Any) -> None:
        pass

    def extract_reasoning(self, model_output: str, request: Any) -> tuple[str | None, str | None]:
        if THINK_START not in model_output and THINK_END not in model_output:
            return None, model_output
        out = model_output.partition(THINK_START)
        out = out[2] if out[1] else out[0]
        if THINK_END not in out:
            return out.removeprefix('thought\n'), None
        reasoning, _, content = out.partition(THINK_END)
        return reasoning.removeprefix('thought\n'), content or None


# --------------------------------------------------------------------------------------------------
# Running a case
# --------------------------------------------------------------------------------------------------


class _LogCatcher(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if record.exc_info and record.exc_info[1] is not None:
            msg += f' [{type(record.exc_info[1]).__name__}: {record.exc_info[1]}]'
        self.records.append(msg)


REQUEST = types.SimpleNamespace(tools=None, tool_choice='auto', skip_special_tokens=False)


def _canon(args: Any) -> str:
    return json.dumps(args, sort_keys=True, ensure_ascii=False)


@dataclass
class Parsed:
    reasoning: str | None
    content: str | None
    calls: list[Call]
    tools_called: bool
    raw_args: list[str]
    logged_errors: list[str]


def run_nonstreaming(env: Env, raw: str) -> Parsed:
    """Mirror vllm serving: reasoning parser first, tool parser on the remaining content."""
    catcher = _LogCatcher()
    root = logging.getLogger()
    root.addHandler(catcher)
    try:
        reasoning, content = None, raw
        rp = env.new_reasoning_parser()
        if rp is not None:
            reasoning, content = rp.extract_reasoning(raw, REQUEST)
        tp = env.new_tool_parser()
        info = tp.extract_tool_calls(content if content is not None else '', REQUEST)
        calls: list[Call] = []
        raw_args: list[str] = []
        for tc in info.tool_calls:
            raw_args.append(tc.function.arguments)
            try:
                calls.append((tc.function.name, json.loads(tc.function.arguments)))
            except Exception:
                calls.append((tc.function.name, {'__unparseable_json__': tc.function.arguments}))
        left = info.content if info.tools_called else content
        return Parsed(reasoning, left.strip() if isinstance(left, str) and left.strip() else None, calls,
                      bool(info.tools_called), raw_args, catcher.records)
    finally:
        root.removeHandler(catcher)


def run_streaming(env: Env, raw: str) -> Parsed:
    """Approximation of the streaming glue: reasoning split non-streaming, then the tool parser fed token deltas.
    (serving.py's end-of-stream remainder flush is not replicated; the scorer's ADK call is non-streaming.)"""
    catcher = _LogCatcher()
    root = logging.getLogger()
    root.addHandler(catcher)
    try:
        reasoning, content = None, raw
        rp = env.new_reasoning_parser()
        if rp is not None:
            reasoning, content = rp.extract_reasoning(raw, REQUEST)
        content = content or ''
        tp = env.new_tool_parser()
        prev, prev_ids = '', []
        text_out = ''
        tcs: dict[int, dict[str, Any]] = {}
        for delta, tid in env.split(content):
            cur = prev + delta
            cur_ids = prev_ids + [tid]
            msg = tp.extract_tool_calls_streaming(prev, cur, delta, prev_ids, cur_ids, [tid], REQUEST)
            prev, prev_ids = cur, cur_ids
            if msg is None:
                continue
            if getattr(msg, 'content', None):
                text_out += msg.content
            for tc in getattr(msg, 'tool_calls', None) or []:
                slot = tcs.setdefault(tc.index, {'name': None, 'args': ''})
                fn = tc.function
                name = fn.get('name') if isinstance(fn, dict) else getattr(fn, 'name', None)
                arg = fn.get('arguments') if isinstance(fn, dict) else getattr(fn, 'arguments', None)
                if name:
                    slot['name'] = name
                if arg:
                    slot['args'] += arg
        calls: list[Call] = []
        raw_args = []
        for idx in sorted(tcs):
            raw_args.append(tcs[idx]['args'])
            try:
                calls.append((tcs[idx]['name'], json.loads(tcs[idx]['args'] or '{}')))
            except Exception:
                calls.append((tcs[idx]['name'], {'__unparseable_json__': tcs[idx]['args']}))
        return Parsed(reasoning, text_out.strip() or None, calls, bool(tcs), raw_args, catcher.records)
    finally:
        root.removeHandler(catcher)


def adk_verdict(name: str, args: dict[str, Any]) -> str:
    """What ADK's FunctionTool does with these args (adk 1.36.1 function_tool.py run_async)."""
    if name not in TOOL_REQUIRED:
        return f"REJECT: unknown tool '{name}'"
    params = set(TOOL_PARAMS[name])
    kept = {k for k in args if k in params}
    missing = [r for r in TOOL_REQUIRED[name] if r not in kept]
    extra = [k for k in args if k not in params]
    if missing:
        return 'REJECT: mandatory input parameters are not present: ' + ', '.join(missing)
    if extra:
        return 'ok (silently drops unknown keys: ' + ', '.join(repr(k) for k in extra) + ')'
    return 'ok'


def short_diff(expected: Any, actual: Any, limit: int = 18) -> str:
    a = json.dumps(expected, indent=1, sort_keys=True, ensure_ascii=False).splitlines()
    b = json.dumps(actual, indent=1, sort_keys=True, ensure_ascii=False).splitlines()
    d = list(difflib.unified_diff(a, b, 'expected', 'parsed', lineterm='', n=0))
    clip = lambda s: s if len(s) <= 160 else s[:157] + '...'  # noqa: E731
    out = [clip(x) for x in d[:limit]]
    if len(d) > limit:
        out.append(f'... (+{len(d) - limit} more diff lines)')
    return '\n'.join(out)


def judge(case: Case, got: Parsed, check_text: bool = True) -> tuple[str, str]:
    """-> (status, detail). status: OK | MANGLED | NOT_PARSED | EXCEPTION."""
    exp_calls = [{'name': n, 'arguments': a} for n, a in case.calls]
    got_calls = [{'name': n, 'arguments': a} for n, a in got.calls]
    if got.logged_errors:
        return 'EXCEPTION', 'parser logged: ' + ' | '.join(got.logged_errors)[:300]
    if not got.tools_called and exp_calls:
        raw_hint = ' (content still holds raw <|tool_call> text)' if got.content and TOOL_CALL_START in got.content else ''
        reason = ' (swallowed by reasoning parser)' if got.reasoning and TOOL_CALL_START in got.reasoning else ''
        return 'NOT_PARSED', f'tools_called=False{raw_hint}{reason}'
    if _canon(exp_calls) != _canon(got_calls):
        return 'MANGLED', short_diff(exp_calls, got_calls)
    if check_text:
        if case.thinking is not None and (got.reasoning or '') != case.thinking:
            return 'MANGLED', short_diff({'reasoning': case.thinking}, {'reasoning': got.reasoning})
        if (case.expected_content or None) != (got.content or None):
            return 'MANGLED', short_diff({'content': case.expected_content}, {'content': got.content})
    return 'OK', ''


TOOL_DEFS = [{'type': 'function', 'function': {'name': n, 'description': n, 'parameters': {
    'type': 'object', 'properties': {p: {'type': 'string'} for p in TOOL_PARAMS[n]}, 'required': TOOL_REQUIRED[n]}}}
    for n in TOOL_PARAMS]


def template_check(env: Env, case: Case) -> tuple[str, str]:
    """Render the case's calls as an assistant tool-call turn with the model's own chat template (what vLLM feeds back
    as history on the next request), cut out the <|tool_call>...<tool_call|> spans, and parse them again. Catches a
    template that escapes/mangles strings differently from what the parser expects. Real tokenizer only."""
    tok = env.tokenizer
    msgs: list[dict[str, Any]] = [{'role': 'user', 'content': 'edit the file'}, {
        'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': f'call_{i}', 'type': 'function', 'function': {'name': n, 'arguments': a}}
            for i, (n, a) in enumerate(case.calls)]}]
    msgs += [{'role': 'tool', 'tool_call_id': f'call_{i}', 'name': n, 'content': 'ok'}
             for i, (n, _a) in enumerate(case.calls)]
    try:
        text = tok.apply_chat_template(msgs, tools=TOOL_DEFS, tokenize=False, add_generation_prompt=False)
    except Exception as e:
        return 'EXCEPTION', f'apply_chat_template: {type(e).__name__}: {e}'[:300]
    spans = re.findall(re.escape(TOOL_CALL_START) + r'.*?' + re.escape(TOOL_CALL_END), text, re.S)
    if len(spans) != len(case.calls):
        return 'NOT_PARSED', f'template rendered {len(spans)} tool-call spans, expected {len(case.calls)}'
    try:
        got = run_nonstreaming(env, ''.join(spans))
    except Exception as e:
        return 'EXCEPTION', f'{type(e).__name__}: {e}'
    status, detail = judge(Case('tpl', 'canonical', case.group, case.calls), got, check_text=False)
    if status != 'OK':
        detail = 'rendered: ' + ''.join(spans)[:400].replace('\n', '\\n') + '\n' + detail
    return status, detail


def tok_roundtrip(env: Env, raw: str) -> str:
    if not env.real_tokenizer:
        return 'n/a'
    try:
        ids = env.tokenizer.encode(raw, add_special_tokens=False)
        back = env.tokenizer.decode(ids, skip_special_tokens=False)
        return 'ok' if back == raw else 'DIFF'
    except Exception as e:
        return f'EXC {type(e).__name__}'


def run_case(env: Env, case: Case, streaming: bool = True, stream_max_chars: int = 60000,
             template: bool = False) -> dict[str, Any]:
    raw = case.build_raw()
    row: dict[str, Any] = {
        'id': case.id, 'family': case.family, 'group': case.group, 'tools': [n for n, _ in case.calls],
        'expect': case.expect, 'raw_chars': len(raw), 'note': case.note, 'tokenizer_roundtrip': tok_roundtrip(env, raw),
    }
    t0 = time.perf_counter()
    try:
        got = run_nonstreaming(env, raw)
        status, detail = judge(case, got)
        row.update(status=status, detail=detail, parsed=[{'name': n, 'arguments': a} for n, a in got.calls],
                   parsed_reasoning=got.reasoning, parsed_content=got.content)
        verdicts = [adk_verdict(n, a) for n, a in got.calls]
        row['adk'] = verdicts
    except Exception as e:  # escaped the parser's own try/except
        row.update(status='EXCEPTION', detail=f'{type(e).__name__}: {e}', parsed=[], adk=[])
    row['ms'] = round((time.perf_counter() - t0) * 1000, 2)
    if streaming and case.streaming and len(raw) > stream_max_chars:
        row['stream'] = 'skipped(size)'
        row['stream_detail'] = (f'raw is {len(raw)} chars > --stream-max-chars {stream_max_chars}: the streaming '
                                'parser re-parses the whole argument string on every token (quadratic)')
    elif streaming and case.streaming:
        try:
            sgot = run_streaming(env, raw)
            sstat, sdetail = judge(case, sgot, check_text=False)
            row['stream'] = sstat
            row['stream_detail'] = sdetail
            if sstat == 'OK' and row['status'] != 'OK':
                row['stream'] = 'OK(!)'
        except Exception as e:
            row['stream'] = 'EXCEPTION'
            row['stream_detail'] = f'{type(e).__name__}: {e}'
    if template and case.thinking is None and case.raw is None and len(case.build_raw()) <= stream_max_chars:
        row['template'], row['template_detail'] = template_check(env, case)
    return row


def classify(row: dict[str, Any]) -> str:
    """regression = a canonical case that failed; surprise = a predicted failure that passed; else 'as expected'."""
    ok = row['status'] == 'OK'
    if row['expect'] == 'roundtrip':
        return 'as expected' if ok else 'REGRESSION'
    return 'SURPRISE(passed)' if ok else 'as expected'


# --------------------------------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------------------------------


def render_markdown(env: Env, rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    L: list[str] = []
    L.append('# gemma4 parser round-trip report')
    L.append('')
    if env.needs_windows_run:
        L.append('> **NEEDS WINDOWS RUN**: this report used '
                 f'{env.info.get("source")}; only the real vllm package with the real tokenizer settles R1.')
        L.append('')
    for k, v in {**env.info, **meta}.items():
        L.append(f'- {k}: `{v}`')
    L.append('')
    c = {}
    for r in rows:
        c.setdefault(r['family'], {}).setdefault(r['status'], 0)
        c[r['family']][r['status']] += 1
    L.append('## Summary')
    L.append('')
    L.append('| family | ' + ' | '.join(['OK', 'MANGLED', 'NOT_PARSED', 'EXCEPTION']) + ' | total |')
    L.append('|---|---|---|---|---|---|')
    for fam, d in c.items():
        L.append(f'| {fam} | ' + ' | '.join(str(d.get(s, 0)) for s in ('OK', 'MANGLED', 'NOT_PARSED', 'EXCEPTION'))
                 + f' | {sum(d.values())} |')
    reg = [r for r in rows if classify(r) == 'REGRESSION']
    sur = [r for r in rows if classify(r).startswith('SURPRISE')]
    L.append('')
    L.append(f'- canonical cases that FAILED to round-trip (parser bugs): **{len(reg)}**')
    L.append(f'- glitch cases that unexpectedly parsed fine: **{len(sur)}**')
    sm = [r for r in rows if r.get('stream') not in (None, 'OK', 'OK(!)') and not str(r.get('stream')).startswith('skipped')
          and r['family'] == 'canonical']
    L.append(f'- canonical cases whose STREAMING path failed (informational; scorer is non-streaming): **{len(sm)}**')
    tp = [r for r in rows if r.get('template') not in (None, 'OK')]
    if any('template' in r for r in rows):
        L.append(f'- cases whose chat-template history round trip FAILED (template -> parser): **{len(tp)}**')
    L.append('')
    L.append('## Table')
    L.append('')
    L.append('| case | tools | chars | parse | stream | tok | tpl | ADK | ms | verdict |')
    L.append('|---|---|---:|---|---|---|---|---|---|---|')
    for r in rows:
        adk = '; '.join(sorted(set(a.split(':')[0] for a in r.get('adk', [])))) or '-'
        if any(a.startswith('REJECT') for a in r.get('adk', [])):
            adk = 'REJECT'
        L.append(f"| {r['id']} | {','.join(dict.fromkeys(r['tools']))} | {r['raw_chars']} | {r['status']} | "
                 f"{r.get('stream', '-')} | {r['tokenizer_roundtrip']} | {r.get('template', '-')} | {adk} | {r['ms']} | "
                 f"{classify(r)} |")
    bad = [r for r in rows if r['status'] != 'OK' or r.get('template') not in (None, 'OK')
           or (r.get('stream') not in (None, 'OK', 'OK(!)') and not str(r.get('stream')).startswith('skipped'))]
    if bad:
        L.append('')
        L.append('## Details (non-OK cases)')
        for r in bad:
            L.append('')
            L.append(f"### {r['id']}  ({r['status']}, expected {r['expect']})")
            if r['note']:
                L.append(f"_{r['note']}_")
            if r.get('detail'):
                L.append('```diff\n' + r['detail'] + '\n```')
            rej = [a for a in r.get('adk', []) if a != 'ok']
            if rej:
                L.append('ADK: ' + ' / '.join(rej))
            if r.get('stream') not in (None, 'OK', 'OK(!)'):
                L.append(f"streaming: {r['stream']}: {r.get('stream_detail', '')[:400]}")
            if r.get('template') not in (None, 'OK'):
                L.append(f"template: {r['template']}: {r.get('template_detail', '')[:600]}")
    return '\n'.join(L) + '\n'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tokenizer', help='dir with the Gemma 4 tokenizer files (the 31b w4a16 model dir /2)')
    ap.add_argument('--vllm-src', help='dir holding extracted vllm/ sources (Mac): run the REAL parser without vllm')
    ap.add_argument('--fake', action='store_true', help='use the tiny built-in fake parser (harness self-test)')
    ap.add_argument('--chunk', type=int, default=3, help='chars per fake-tokenizer token (streaming deltas)')
    ap.add_argument('--filter', action='append', default=[], help='only cases whose id contains this (repeatable)')
    ap.add_argument('--family', choices=['canonical', 'glitch'])
    ap.add_argument('--no-streaming', action='store_true')
    ap.add_argument('--no-template', action='store_true',
                    help='skip the chat-template history round trip (render with apply_chat_template, parse again)')
    ap.add_argument('--stream-max-chars', type=int, default=60000,
                    help='skip the streaming simulation above this raw length (it is quadratic)')
    ap.add_argument('--long-chars', type=int, help='override the size of the main long string (default ~13k)')
    ap.add_argument('--list', action='store_true', help='list case ids and exit')
    ap.add_argument('--json', help='write full results here')
    ap.add_argument('--md', help='write the markdown report here')
    ap.add_argument('--quiet', action='store_true', help='print only the summary')
    a = ap.parse_args(argv)

    cases = build_cases(a.long_chars)
    if a.filter:
        cases = [c for c in cases if any(f in c.id for f in a.filter)]
    if a.family:
        cases = [c for c in cases if c.family == a.family]
    if a.list:
        for c in cases:
            print(f'{c.family:9} {c.group:10} {c.id}')
        return 0
    try:
        env = load_env(a)
    except EnvError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2

    meta = {'python': sys.version.split()[0], 'n_cases': len(cases)}
    if env.real_tokenizer:
        vocab = env.tokenizer.get_vocab()
        meta['special_token_ids'] = {t: vocab.get(t) for t in
                                     (TOOL_CALL_START, TOOL_CALL_END, STR_DELIM, THINK_START, THINK_END)}
    use_tpl = env.real_tokenizer and hasattr(env.tokenizer, 'apply_chat_template') and not a.no_template
    rows = [run_case(env, c, streaming=not a.no_streaming, stream_max_chars=a.stream_max_chars, template=use_tpl)
            for c in cases]
    md = render_markdown(env, rows, meta)
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(md, encoding='utf-8')
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({'env': env.info, 'needs_windows_run': env.needs_windows_run,
                                            'meta': meta, 'rows': rows}, indent=2, ensure_ascii=False), encoding='utf-8')
    if a.quiet:
        print('\n'.join(md.split('## Table')[0].splitlines()))
    else:
        print(md)
    return 1 if any(classify(r) == 'REGRESSION' for r in rows) else 0


if __name__ == '__main__':
    sys.exit(main())
