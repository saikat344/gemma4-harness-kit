# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Vendored: the typed compaction-summarizer serializer from google-adk 2.2.0 (P-E5 arm Y).

Source: google-adk 2.2.0, `google/adk/apps/llm_event_summarizer.py`,
`LlmEventSummarizer._format_events_for_prompt`, `LlmEventSummarizer._truncate` and the class constant
`_MAX_TOOL_CONTENT_CHARS` (upstream commit bdb5582, the first release with the typed serializer).
Read from the PyPI wheel `google_adk-2.2.0-py3-none-any.whl`
(sha256 ebdf3d931dc2b9c5b30d995358fc2ae99d59594c48a4aaf7496869ccd2c5f245); the wheel was unzipped and read,
never installed or executed. Only these three items are copied.

Adaptations (minimal; behaviour on an event list is unchanged):
- The two methods became module-level functions (they never used instance state; `self._truncate` ->
  `_truncate`, `self._MAX_TOOL_CONTENT_CHARS` -> the module constant).
- They run on google-adk 1.36.1 `Event` objects (harness/runner/.venv). Every attribute they read
  (`event.content.parts`, `event.actions.compaction`, `event.author`, `part.thought`, `part.text`,
  `part.function_call.{name,args}`, `part.function_response.{name,response}`) exists with the same meaning
  in 1.36.1, so no access had to change. No type annotations reference 2.x types.
- NOT vendored: 2.2.0 also changed `_DEFAULT_PROMPT_TEMPLATE` (new wording, real newlines). P-E5 keeps the
  1.36.1 prompt for both arms, so only the serializer differs.
"""

from __future__ import annotations

from typing import Any

# Tool call args and responses can be large (e.g. search results). Cap how
# much of each is rendered so compaction does not inflate the very context
# it exists to shrink.
_MAX_TOOL_CONTENT_CHARS = 2000


def format_events_for_prompt(events: list[Any]) -> str:
  """Formats events into prompt text, including thoughts and tool calls.

  Thoughts carry the agent's analysis of tool responses, and tool calls and
  responses carry the evidence retrieved so far, so all three are included.
  Thoughts emitted by a compaction event are skipped so a prior summary's
  reasoning does not leak into the next summary.
  """
  formatted_history = []
  for event in events:
    if not (event.content and event.content.parts):
      continue
    is_compaction = bool(event.actions and event.actions.compaction)
    for part in event.content.parts:
      if part.thought and part.text:
        if not is_compaction:
          formatted_history.append(f'{event.author} (thought): {part.text}')
      elif part.text:
        formatted_history.append(f'{event.author}: {part.text}')
      if part.function_call:
        args = _truncate(str(part.function_call.args))
        formatted_history.append(
            f'{event.author} called tool: {part.function_call.name}({args})'
        )
      if part.function_response:
        response = _truncate(str(part.function_response.response))
        formatted_history.append(
            f'Tool response from {part.function_response.name}: {response}'
        )
  return '\n'.join(formatted_history)


def _truncate(text: str) -> str:
  """Caps `text` at the tool-content limit, marking dropped characters."""
  limit = _MAX_TOOL_CONTENT_CHARS
  if len(text) <= limit:
    return text
  return f'{text[:limit]}... [truncated {len(text) - limit} chars]'
