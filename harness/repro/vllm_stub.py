"""Load vLLM's gemma4 tool/reasoning parser SOURCE files without importing vllm (Mac, CPU, no wheel install).

The parser modules only need a handful of vllm names (protocol dataclasses, a logger, the abstract base
classes, `find_common_prefix`, `make_tool_call_id`). This module registers tiny stand-ins for them in
`sys.modules`, then executes the real `gemma4_tool_parser.py`, `gemma4_reasoning_parser.py` and
`basic_parsers.py` from an extracted copy of the vllm 0.19.1 wheel. So the parsing logic under test is
the real vLLM code, just not run inside the real package.

Stand-ins copied from vllm 0.19.1 (kept byte-for-byte in behaviour):
  ToolParser.__init__/vocab/adjust_request-less base, ReasoningParser.__init__/vocab,
  find_common_prefix, DeltaMessage/DeltaToolCall/... field names.

What this does NOT cover: vllm's request/response (de)serialisation, the detokenizer, the real tokenizer
(use --tokenizer on the Windows box for that), or the streaming glue in serving.py.

Layout expected under `src_dir`: <src_dir>/vllm/tool_parsers/gemma4_tool_parser.py,
<src_dir>/vllm/reasoning/{gemma4_reasoning_parser,basic_parsers}.py. Extract them from the wheel with
`unzip -q vllm-0.19.1-*.whl 'vllm/tool_parsers/*' 'vllm/reasoning/*' -d <src_dir>` (data/vllm-src/0.19.1).
"""

from __future__ import annotations

import importlib.util
import itertools
import logging
import sys
import types
import zlib
from functools import cached_property
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_counter = itertools.count()


class _Base(BaseModel):
    model_config = {'extra': 'allow'}


class FunctionCall(_Base):
    name: str
    arguments: str


class ToolCall(_Base):
    id: str | None = None
    type: str = 'function'
    function: FunctionCall


class DeltaFunctionCall(_Base):
    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(_Base):
    id: str | None = None
    type: str | None = None
    index: int = 0
    function: DeltaFunctionCall | None = None


class DeltaMessage(_Base):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)


class ExtractedToolCallInformation(_Base):
    tools_called: bool
    tool_calls: list[ToolCall]
    content: str | None = None


class ChatCompletionRequest(_Base):
    tools: Any = None
    tool_choice: Any = None
    skip_special_tokens: bool = True


class ResponsesRequest(_Base):
    pass


class ToolParser:
    """vllm/tool_parsers/abstract_tool_parser.py:ToolParser (init + vocab only)."""

    def __init__(self, tokenizer: Any, tools: list | None = None):
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        self.model_tokenizer = tokenizer
        self.tools = tools

    @cached_property
    def vocab(self) -> dict[str, int]:
        return self.model_tokenizer.get_vocab()

    def adjust_request(self, request: Any) -> Any:
        return request


class ReasoningParser:
    """vllm/reasoning/abs_reasoning_parsers.py:ReasoningParser (init + vocab only)."""

    def __init__(self, tokenizer: Any, *args: Any, **kwargs: Any):
        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> dict[str, int]:
        return self.model_tokenizer.get_vocab()


def find_common_prefix(s1: str, s2: str) -> str:
    """vllm/tool_parsers/utils.py:find_common_prefix (verbatim logic)."""
    prefix = ''
    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def make_tool_call_id(*_a: Any, **_k: Any) -> str:
    return f'chatcmpl-tool-{next(_counter):016x}'


def _module(name: str, **attrs: Any) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def _install_stubs() -> None:
    for pkg in ('vllm', 'vllm.entrypoints', 'vllm.entrypoints.openai', 'vllm.entrypoints.openai.chat_completion',
                'vllm.entrypoints.openai.engine', 'vllm.entrypoints.openai.responses', 'vllm.tool_parsers',
                'vllm.reasoning', 'vllm.tokenizers'):
        m = _module(pkg)
        m.__path__ = []  # type: ignore[attr-defined]
    _module('vllm.entrypoints.chat_utils', make_tool_call_id=make_tool_call_id)
    _module('vllm.entrypoints.openai.chat_completion.protocol', ChatCompletionRequest=ChatCompletionRequest)
    _module('vllm.entrypoints.openai.engine.protocol', DeltaFunctionCall=DeltaFunctionCall,
            DeltaMessage=DeltaMessage, DeltaToolCall=DeltaToolCall,
            ExtractedToolCallInformation=ExtractedToolCallInformation, FunctionCall=FunctionCall,
            ToolCall=ToolCall)
    _module('vllm.entrypoints.openai.responses.protocol', ResponsesRequest=ResponsesRequest)
    _module('vllm.logger', init_logger=lambda name=None: logging.getLogger(name or 'vllm-stub'))
    _module('vllm.tokenizers', TokenizerLike=object)
    _module('vllm.tool_parsers.abstract_tool_parser', ToolParser=ToolParser, Tool=Any)
    _module('vllm.tool_parsers.utils', find_common_prefix=find_common_prefix)
    _module('vllm.reasoning.abs_reasoning_parsers', ReasoningParser=ReasoningParser)


def _load(modname: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec and spec.loader, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def load_gemma4_parsers(src_dir: str | Path) -> tuple[type, type, dict[str, Any]]:
    """Return (Gemma4ToolParser, Gemma4ReasoningParser, info) from extracted vllm sources.

    Only call this when `import vllm` is not what you want (Mac). It replaces any `vllm*` entries in
    sys.modules, so never use it in a process that also needs the real package."""
    root = Path(src_dir) / 'vllm'
    tp = root / 'tool_parsers' / 'gemma4_tool_parser.py'
    rp = root / 'reasoning' / 'gemma4_reasoning_parser.py'
    bp = root / 'reasoning' / 'basic_parsers.py'
    for p in (tp, rp, bp):
        if not p.exists():
            raise FileNotFoundError(f'{p} not found; extract vllm/tool_parsers and vllm/reasoning from the wheel')
    _install_stubs()
    tool_mod = _load('vllm.tool_parsers.gemma4_tool_parser', tp)
    _load('vllm.reasoning.basic_parsers', bp)
    reas_mod = _load('vllm.reasoning.gemma4_reasoning_parser', rp)
    info = {'source': 'extracted-source (stubbed vllm imports)', 'tool_parser_file': str(tp),
            'reasoning_parser_file': str(rp)}
    return tool_mod.Gemma4ToolParser, reas_mod.Gemma4ReasoningParser, info


class FakeTokenizer:
    """Just enough tokenizer for the parsers' constructors and for chunking streaming deltas.

    Special tokens are atomic; everything else is split into `chunk` characters per token. Token ids are
    arbitrary but stable. This is NOT Gemma's tokenizer: use --tokenizer for the real one."""

    SPECIALS = ['<|tool_call>', '<tool_call|>', '<|"|>', '<|channel>', '<channel|>', '<|turn>', '<turn|>',
                '<|tool_response>', '<tool_response|>', '<|think|>', '<eos>']

    def __init__(self, chunk: int = 3) -> None:
        self.chunk = chunk
        self._vocab = {t: 48 + i for i, t in enumerate(self.SPECIALS)}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def pieces(self, text: str) -> list[str]:
        out: list[str] = []
        i = 0
        buf = ''
        while i < len(text):
            for s in self.SPECIALS:
                if text.startswith(s, i):
                    if buf:
                        out.extend(buf[j:j + self.chunk] for j in range(0, len(buf), self.chunk))
                        buf = ''
                    out.append(s)
                    i += len(s)
                    break
            else:
                buf += text[i]
                i += 1
        if buf:
            out.extend(buf[j:j + self.chunk] for j in range(0, len(buf), self.chunk))
        return out

    def apply_chat_template(self, messages: list, tools: Any = None, tokenize: bool = False,
                            add_generation_prompt: bool = False) -> str:
        """Minimal template: only enough to exercise the template round-trip check (real runs use Gemma's)."""
        from harness.repro.parser_cases import render_call

        out = ''
        for m in messages:
            out += f"<|turn>{m['role']}\n"
            for tc in m.get('tool_calls') or []:
                out += render_call(tc['function']['name'], tc['function']['arguments'])
            out += (m.get('content') or '') + '<turn|>\n'
        return out

    def token_ids(self, text: str) -> list[int]:
        return [self._vocab.get(p, 1000 + (zlib.crc32(p.encode()) % 1000)) for p in self.pieces(text)]


__all__ = ['FakeTokenizer', 'load_gemma4_parsers']
