"""Harness contracts of the pinned google-adk, checked on CPU in seconds (no GPU, no Docker, no competition data).

    uv run --project harness/runner python -m pytest harness/contracts -q -s

B1   the compaction serializer (`LlmEventSummarizer._format_events_for_prompt`) passes only `part.text` to the
     summarizer: tool calls and tool results never reach it, and entries are joined with a literal backslash-n.
     Fixed upstream in commit bdb5582 (first released in google-adk 2.2.0); the 1.x line through 1.39.1 is unchanged.
     On google-adk >= 2.2.0 the B1 "not in" assertions fail by design. The vendored 2.2.0 serializer
     (harness/analysis/adk220_format.py) is checked side by side.
T    instruction templating (`inject_session_state`): `{name}` missing from state raises KeyError, `{name?}` becomes
     empty, a brace group that is not an identifier (JSON) stays literal, substituted values are not re-scanned.
C    context limit: the token-threshold trigger reads the prompt tokens recorded on the latest event that has usage,
     so tool results appended after that call are not counted; one large result can push the next request past
     max_model_len - max_output_tokens before compaction fires (vLLM then rejects it).
See docs/harness-contracts.md for the source references.
"""

from __future__ import annotations

import asyncio
from importlib.metadata import version
from types import SimpleNamespace

import pytest
from google.adk.apps.compaction import _latest_prompt_token_count
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.utils.instructions_utils import inject_session_state
from google.genai import types

PINNED_ADK = '1.36.1'
ADK = version('google-adk')
P = types.Part


def _vtuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split('.')[:3] if x.isdigit())


def setup_module(_m) -> None:
    print(f'\ngoogle-adk {ADK} (pinned by the scorer: {PINNED_ADK})')


def b1_events() -> list[Event]:
    return [
        Event(author='user', content=types.Content(role='user', parts=[P(text='ISSUE_MARKER fix bug')])),
        Event(author='agent', content=types.Content(role='model', parts=[
            P(text='THOUGHT_MARKER', thought=True),
            P(function_call=types.FunctionCall(name='run_command', args={'command': 'CALL_ARG_MARKER'}))])),
        Event(author='agent', content=types.Content(role='user', parts=[
            P(function_response=types.FunctionResponse(name='run_command', response={'out': 'RESULT_MARKER'}))])),
        Event(author='agent', content=types.Content(role='model', parts=[P(text='TEXT_MARKER')])),
    ]


# ---------------------------------------------------------------------------------------------- B1


def test_b1_serializer_keeps_text_and_drops_tool_fields():
    out = LlmEventSummarizer.__new__(LlmEventSummarizer)._format_events_for_prompt(b1_events())
    print(f'1.x serializer output: {out!r}')
    assert 'ISSUE_MARKER' in out and 'THOUGHT_MARKER' in out and 'TEXT_MARKER' in out
    why = (f'google-adk {ADK}: tool fields reached the summarizer. Expected on >= 2.2.0 (upstream commit bdb5582 '
           f'serializes tool calls and truncated responses); the scorer pins {PINNED_ADK}, where they do not.')
    assert 'CALL_ARG_MARKER' not in out, why
    assert 'RESULT_MARKER' not in out, why
    assert '\\n' in out and '\n' not in out, 'entries are joined with the two characters backslash-n, not a newline'
    assert out == 'user: ISSUE_MARKER fix bug\\nagent: THOUGHT_MARKER\\nagent: TEXT_MARKER'


def test_b1_vendored_220_serializer_keeps_tool_fields():
    """Contrast: the typed serializer of google-adk 2.2.0 (vendored, Apache-2.0) passes calls and results."""
    adk220 = pytest.importorskip('harness.analysis.adk220_format')
    out = adk220.format_events_for_prompt(b1_events())
    print(f'2.2.0 serializer output: {out!r}')
    for marker in ('ISSUE_MARKER', 'TEXT_MARKER', 'CALL_ARG_MARKER', 'RESULT_MARKER'):
        assert marker in out
    assert '\n' in out


def test_b1_pinned_version():
    if ADK != PINNED_ADK:
        pytest.skip(f'google-adk {ADK} installed; the contracts are stated for the pinned {PINNED_ADK}')
    assert _vtuple(ADK) < (2, 2, 0)


# ----------------------------------------------------------------------------------------- templating


def _render(template: str, state: dict) -> str:
    ctx = SimpleNamespace(_invocation_context=SimpleNamespace(session=SimpleNamespace(state=state),
                                                              artifact_service=None))
    return asyncio.run(inject_session_state(template, ctx))


def test_templating_missing_name_raises_keyerror():
    with pytest.raises(KeyError):
        _render('Fix this: {problem_description}. Note {typo_here}.', {'problem_description': 'x'})


def test_templating_optional_name_becomes_empty():
    assert _render('Hints: [{hints?}]', {}) == 'Hints: []'


def test_templating_json_and_empty_braces_stay_literal():
    t = 'Reply as {"status": "ok"} or {}; issue: {problem_description}'
    assert _render(t, {'problem_description': 'P'}) == 'Reply as {"status": "ok"} or {}; issue: P'


def test_templating_substituted_values_are_not_rescanned():
    assert _render('{problem_description}', {'problem_description': 'use {other} here'}) == 'use {other} here'


# ------------------------------------------------------------------------------------------ context


def test_context_trigger_ignores_tool_results_after_the_last_usage():
    """The threshold check sees the previous call's prompt tokens, not the prompt about to be sent."""
    threshold, max_model_len, max_output = 14336, 32768, 16384
    call = Event(author='agent', content=types.Content(role='model', parts=[P(text='reading a file')]),
                 usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=11474,
                                                                            candidates_token_count=3871))
    big = 'x' * (4 * 6000)  # ~6000 tokens of tool output at ADK's 4 chars/token estimate
    result = Event(author='agent', content=types.Content(role='user', parts=[
        P(function_response=types.FunctionResponse(name='read_file', response={'content': big}))]))
    seen = _latest_prompt_token_count([call, result])
    assert seen == 11474 < threshold  # so no compaction runs before the next call
    next_prompt = 11474 + 3871 + 6000
    assert next_prompt + max_output > max_model_len  # vLLM rejects: ContextWindowExceededError, the task dies
    assert next_prompt + 8192 <= max_model_len  # an 8192 output cap leaves room for this result
