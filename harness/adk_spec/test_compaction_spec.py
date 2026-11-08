"""Characterization tests for google-adk token-threshold compaction.

Every test asserts what the pinned scorer stack (google-adk 1.36.1) does today,
as documented in docs/harness-contracts.md §7 (release; compaction.md in the
dev repo). Run the same file against other ADK
versions (harness/adk_spec/run_matrix.py): a test that passes on 1.36.1 and
fails elsewhere marks a behaviour upstream changed.

No network, no GPU. The model is a scripted BaseLlm (or a LiteLlm with a fake
client), so everything except the model itself is the real ADK code path:
Runner -> LlmAgent flow -> CompactionRequestProcessor -> window selection ->
LlmEventSummarizer -> contents re-injection.

Markers in the scripted conversation:
  TASK_MARKER     the user's task message
  SYSTEM_MARKER   the agent instruction
  THOUGHT_<i>     agent thought text on call i
  PATH_<i>        read_file argument on call i
  RESULT_PATH_<i> the tool result for call i
  SUMTHOUGHT_<k>  summarizer thought on summary k
  SUMMARY_<k>     summarizer visible text on summary k
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest
from pydantic import PrivateAttr

from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
import google.adk.apps.compaction as compaction
from google.adk.events.event import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

APP = "spec"
USER = "u"
SUMMARIZER_USAGE = 3400


# --- scripted model ---------------------------------------------------------


@dataclass
class Call:
  kind: str  # "agent" | "summarizer"
  index: int  # per-kind index
  contents: list[types.Content]
  system_instruction: str
  has_tools: bool
  config: dict


def _texts(contents: list[types.Content]) -> list[str]:
  return [p.text for c in contents for p in (c.parts or []) if p.text]


def _blob(contents: list[types.Content]) -> str:
  """Every text, function-call arg and function-response value, as one string."""
  out = []
  for c in contents:
    for p in c.parts or []:
      if p.text:
        out.append(p.text)
      if p.function_call:
        out.append(f"{p.function_call.name}({p.function_call.args})")
      if p.function_response:
        out.append(f"{p.function_response.name}->{p.function_response.response}")
  return "\n".join(out)


def _usage(n: int | None):
  if n is None:
    return None
  return types.GenerateContentResponseUsageMetadata(
      prompt_token_count=n, candidates_token_count=10, total_token_count=n + 10
  )


class ScriptedLlm(BaseLlm):
  """Agent calls follow `usage`; summarizer calls return SUMTHOUGHT_k + SUMMARY_k."""

  model: str = "scripted-llm"
  _usage: list = PrivateAttr(default_factory=list)
  _final_usage: int | None = PrivateAttr(default=800)
  _log: list = PrivateAttr(default_factory=list)

  @classmethod
  def supported_models(cls) -> list[str]:
    return ["scripted-llm"]

  def _is_summarizer(self, req) -> bool:
    return not req.tools_dict and any(
        "conversation history" in t for t in _texts(req.contents)[:1]
    )

  async def generate_content_async(self, llm_request, stream: bool = False):
    kind = "summarizer" if self._is_summarizer(llm_request) else "agent"
    index = sum(1 for c in self._log if c.kind == kind)
    cfg = llm_request.config
    self._log.append(
        Call(
            kind=kind,
            index=index,
            contents=[c.model_copy(deep=True) for c in llm_request.contents],
            system_instruction=str(cfg.system_instruction) if cfg and cfg.system_instruction else "",
            has_tools=bool(llm_request.tools_dict),
            config=cfg.model_dump(exclude_none=True) if cfg else {},
        )
    )
    if kind == "summarizer":
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[
                  types.Part(text=f"SUMTHOUGHT_{index}", thought=True),
                  types.Part(text=f"SUMMARY_{index}"),
              ],
          ),
          usage_metadata=_usage(SUMMARIZER_USAGE),
      )
    elif index < len(self._usage):
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[
                  types.Part(text=f"THOUGHT_{index}", thought=True),
                  types.Part(
                      function_call=types.FunctionCall(
                          name="read_file", args={"path": f"PATH_{index}"}
                      )
                  ),
              ],
          ),
          usage_metadata=_usage(self._usage[index]),
      )
    else:
      yield LlmResponse(
          content=types.Content(role="model", parts=[types.Part(text="DONE")]),
          usage_metadata=_usage(self._final_usage),
      )


class Spy(BasePlugin):
  """Records what plugin on_event_callback sees (the harness trace hook)."""

  def __init__(self):
    super().__init__(name="spy")
    self.seen: list[Event] = []

  async def on_event_callback(self, *, invocation_context, event):
    self.seen.append(event)
    return None


@dataclass
class Run:
  log: list[Call]
  yielded: list[Event]
  spied: list[Event]
  session_events: list[Event]
  error: BaseException | None = None

  def calls(self, kind: str) -> list[Call]:
    return [c for c in self.log if c.kind == kind]

  def agent_call_after_summary(self, k: int) -> Call:
    seen = -1
    for c in self.log:
      if c.kind == "summarizer":
        seen = c.index
      elif seen == k:
        return c
    raise AssertionError(f"no agent call after summary {k}")

  def order(self) -> list[str]:
    return [f"{c.kind[0]}{c.index}" for c in self.log]


def run_scenario(
    *,
    usage: list[int | None],
    threshold: int,
    retention: int,
    tool_chars: int = 200,
    final_usage: int | None = 800,
    pre_events: Callable[[], list[Event]] | None = None,
) -> Run:
  def read_file(path: str) -> dict:
    """Reads a file from the workspace."""
    return {"content": f"RESULT_{path} " + "x" * tool_chars}

  llm = ScriptedLlm()
  llm._usage = list(usage)
  llm._final_usage = final_usage
  agent = LlmAgent(
      name="agent", model=llm, instruction="SYSTEM_MARKER", tools=[read_file]
  )
  spy = Spy()
  app = App(
      name=APP,
      root_agent=agent,
      plugins=[spy],
      events_compaction_config=EventsCompactionConfig(
          compaction_interval=999,
          overlap_size=0,
          token_threshold=threshold,
          event_retention_size=retention,
      ),
  )
  service = InMemorySessionService()
  runner = Runner(app=app, session_service=service)

  async def go() -> Run:
    session = await service.create_session(app_name=APP, user_id=USER)
    for ev in (pre_events() if pre_events else []):
      await service.append_event(session=session, event=ev)
    yielded, error = [], None
    try:
      async for ev in runner.run_async(
          user_id=USER,
          session_id=session.id,
          new_message=types.Content(
              role="user", parts=[types.Part(text="TASK_MARKER fix the bug")]
          ),
      ):
        yielded.append(ev)
    except Exception as e:  # recorded; tests decide whether it matters
      error = e
    final = await service.get_session(
        app_name=APP, user_id=USER, session_id=session.id
    )
    return Run(llm._log, yielded, spy.seen, list(final.events), error)

  return asyncio.run(go())


def _summary_input(run: Run, k: int) -> str:
  return "\n".join(_texts(run.calls("summarizer")[k].contents))


# --- main scenario: two compactions ----------------------------------------
# usage per agent call: crossing 1200 after call 2 and after call 4.
# retention 2 keeps the last tool turn (call event + response event) raw.


@pytest.fixture(scope="module")
def two():
  run = run_scenario(
      usage=[500, 900, 1300, 700, 1300, 700], threshold=1200, retention=2
  )
  assert run.error is None, repr(run.error)
  return run


# trigger


def test_t01_compacts_before_the_call_after_threshold_is_crossed(two):
  """Two summaries, each right after the agent call whose usage reached 1200."""
  assert two.order() == ["a0", "a1", "a2", "s0", "a3", "a4", "s1", "a5", "a6"]


# serializer (what reaches the summarizer)


def test_s01_task_text_reaches_summarizer(two):
  assert "TASK_MARKER" in _summary_input(two, 0)


def test_s02_agent_thoughts_reach_summarizer(two):
  text = _summary_input(two, 0)
  assert "THOUGHT_0" in text and "THOUGHT_1" in text


def test_s03_thoughts_are_unlabeled(two):
  """1.36.1 renders thought text exactly like visible text (`agent: ...`)."""
  text = _summary_input(two, 0)
  assert "agent: THOUGHT_0" in text
  assert "(thought)" not in text


def test_s04_tool_call_args_never_reach_summarizer(two):
  text = _summary_input(two, 0)
  assert "PATH_0" not in text and "PATH_1" not in text
  assert "read_file" not in text


def test_s05_tool_results_never_reach_summarizer(two):
  text = _summary_input(two, 0)
  assert "RESULT_PATH_0" not in text and "RESULT_PATH_1" not in text


def test_s06_history_joined_by_literal_backslash_n(two):
  text = _summary_input(two, 0)
  a, b = text.index("THOUGHT_0"), text.index("THOUGHT_1")
  between = text[a:b]
  assert "\\n" in between  # two characters: backslash, n
  assert "\n" not in between  # no real newline


def test_s07_seed_carries_previous_summary(two):
  assert "SUMMARY_0" in _summary_input(two, 1)


def test_s08_seed_carries_previous_summarizer_thinking(two):
  """Summarizer k's own thinking is fed into summarizer k+1 (still true on 2.2.0/main)."""
  assert "SUMTHOUGHT_0" in _summary_input(two, 1)


def test_s09_summarizer_request_has_no_tools_or_system_instruction(two):
  for c in two.calls("summarizer"):
    assert not c.has_tools
    assert c.system_instruction == ""


def test_s10_summarizer_request_is_one_user_message(two):
  for c in two.calls("summarizer"):
    assert [x.role for x in c.contents] == ["user"]
    assert len(c.contents[0].parts) == 1


def test_s11_summarizer_runs_on_the_agents_own_model_object(two):
  """No summarizer configured: ADK used agent.canonical_model (our ScriptedLlm)."""
  assert len(two.calls("summarizer")) == 2


def test_s12_default_prompt_wording(two):
  text = _summary_input(two, 0)
  assert text.startswith(
      "The following is a conversation history between a user and an AI agent."
      " Please summarize the conversation"
  )


# re-injection (what the agent sees next)


def test_r01_summary_reinjected_as_user_for_context(two):
  c = two.agent_call_after_summary(0)
  assert c.contents[0].role == "user"
  assert _texts([c.contents[0]]) == ["For context:", "[model] said: SUMMARY_0"]


def test_r02_summarizer_thinking_hidden_from_agent(two):
  for c in two.calls("agent"):
    assert "SUMTHOUGHT" not in _blob(c.contents)


def test_r03_task_message_gone_after_compaction(two):
  c = two.agent_call_after_summary(0)
  assert "TASK_MARKER" not in _blob(c.contents)


def test_r04_last_tool_turn_kept_verbatim_older_ones_dropped(two):
  blob = _blob(two.agent_call_after_summary(0).contents)
  assert "PATH_2" in blob and "RESULT_PATH_2" in blob
  assert "PATH_0" not in blob and "PATH_1" not in blob


def test_r05_system_instruction_survives(two):
  assert "SYSTEM_MARKER" in two.agent_call_after_summary(1).system_instruction


def test_r06_only_latest_summary_shown(two):
  """Format-independent (r01 covers the wording): newest summary in, older one out."""
  texts = _texts(two.agent_call_after_summary(1).contents)
  assert any("SUMMARY_1" in t for t in texts)
  assert not any("SUMMARY_0" in t for t in texts)


# visibility


def test_v01_compaction_events_are_not_yielded(two):
  assert not [e for e in two.yielded if e.actions and e.actions.compaction]


def test_v02_plugin_on_event_never_sees_compaction(two):
  """The scorer trace (adk_eval_core) is fed by this hook."""
  assert not [e for e in two.spied if e.actions and e.actions.compaction]


def test_v03_session_holds_the_compaction_events(two):
  assert len([e for e in two.session_events if e.actions and e.actions.compaction]) == 2


# --- trigger lag: a large tool result is not counted ------------------------


def test_l01_large_tool_result_not_counted_before_next_call():
  """Call 0 reports 200 prompt tokens; its tool result adds ~5,000 (20k chars).

  1.36.1 reads only the previous call's prompt_token_count, so it does not
  compact before call 1. Upstream main a72a8ed3 counts the tool output.
  """
  run = run_scenario(
      usage=[200], threshold=1000, retention=1, tool_chars=20_000, final_usage=300
  )
  assert run.error is None, repr(run.error)
  assert run.order() == ["a0", "a1"]


# --- orphaned function_call blocks later compaction (upstream #7236) --------


def _orphan_history() -> list[Event]:
  return [
      Event(
          author="user",
          invocation_id="old",
          content=types.Content(role="user", parts=[types.Part(text="OLD_TASK")]),
      ),
      Event(
          author="agent",
          invocation_id="old",
          content=types.Content(role="model", parts=[types.Part(text="OLD_ANSWER")]),
          usage_metadata=_usage(100),
      ),
      Event(
          author="agent",
          invocation_id="old",
          content=types.Content(
              role="model",
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          id="orphan-1", name="read_file", args={"path": "ORPHAN"}
                      )
                  )
              ],
          ),
      ),
  ]


def test_o01_orphan_call_blocks_all_later_compactions():
  """Every agent call is over threshold, yet at most one compaction happens."""
  run = run_scenario(
      usage=[1300] * 6,
      threshold=1200,
      retention=2,
      final_usage=1300,
      pre_events=_orphan_history,
  )
  assert run.error is None, repr(run.error)
  assert len(run.calls("agent")) == 7
  assert len(run.calls("summarizer")) <= 1


# --- summarizer inherits the agent's LiteLlm args (thinking) ----------------


class FakeLiteClient(LiteLLMClient):

  def __init__(self):
    self.kwargs: list[dict] = []

  async def acompletion(self, model, messages, tools, **kwargs):
    import litellm

    self.kwargs.append(dict(model=model, messages=messages, tools=tools, **kwargs))
    return litellm.ModelResponse(
        id="x",
        model=model,
        choices=[
            litellm.Choices(
                index=0,
                finish_reason="stop",
                message=litellm.Message(role="assistant", content="SUMMARY"),
            )
        ],
        usage=litellm.Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )


THINKING = {
    "chat_template_kwargs": {"enable_thinking": True},
    "thinking_token_budget": 4096,
}


def _lite_agent(name: str) -> tuple[LlmAgent, FakeLiteClient]:
  client = FakeLiteClient()
  # Same shape adk-submission produces: thinking args on the model object.
  model = LiteLlm(
      model="openai/gemma-4", api_base="http://127.0.0.1:9/v1", api_key="test",
      extra_body=THINKING, llm_client=client,
  )
  agent = LlmAgent(
      name=name,
      model=model,
      generate_content_config=types.GenerateContentConfig(
          temperature=0.3, max_output_tokens=8192
      ),
  )
  return agent, client


def _summarize_with(config: EventsCompactionConfig, agent: LlmAgent) -> None:
  compaction._ensure_compaction_summarizer(config=config, agent=agent)
  ev = Event(
      author="user",
      content=types.Content(role="user", parts=[types.Part(text="hello")]),
  )
  asyncio.run(config.summarizer.maybe_summarize_events(events=[ev]))


def _config() -> EventsCompactionConfig:
  return EventsCompactionConfig(
      compaction_interval=999, overlap_size=0, token_threshold=10, event_retention_size=0
  )


def test_k01_summarizer_inherits_agent_thinking_args():
  agent, client = _lite_agent("agent")
  _summarize_with(_config(), agent)
  assert client.kwargs, "summarizer did not call the agent's model"
  assert client.kwargs[0]["extra_body"] == THINKING


def test_k02_summarizer_ignores_agent_generation_config():
  agent, client = _lite_agent("agent")
  _summarize_with(_config(), agent)
  sent = client.kwargs[0]
  assert "temperature" not in sent
  assert "max_tokens" not in sent and "max_completion_tokens" not in sent


def test_k04_litellm_sends_past_thoughts_as_reasoning_content():
  """A role=model turn's thought part goes out as `reasoning_content`.

  Also why the >=2.13 re-injection (summary as the agent's own model turn, thought
  part included) would put summarizer thinking in front of the agent.
  """
  from google.adk.models.llm_request import LlmRequest

  client = FakeLiteClient()
  model = LiteLlm(model="openai/gemma-4", api_base="http://127.0.0.1:9/v1",
                  api_key="test", llm_client=client)
  req = LlmRequest(model="openai/gemma-4", contents=[
      types.Content(role="user", parts=[types.Part(text="TASK")]),
      types.Content(role="model", parts=[
          types.Part(text="PAST_THOUGHT", thought=True), types.Part(text="VISIBLE")]),
      types.Content(role="user", parts=[types.Part(text="go on")]),
  ])

  async def go():
    async for _ in model.generate_content_async(req):
      pass

  asyncio.run(go())
  assistant = [m for m in client.kwargs[0]["messages"] if m["role"] == "assistant"]
  assert assistant[0]["content"] == "VISIBLE"
  assert assistant[0]["reasoning_content"] == "PAST_THOUGHT"


def test_k03_summarizer_cached_on_shared_config():
  """One EventsCompactionConfig shared by two agents keeps the first agent's model."""
  config = _config()
  a1, c1 = _lite_agent("first")
  a2, c2 = _lite_agent("second")
  _summarize_with(config, a1)
  _summarize_with(config, a2)
  assert len(c1.kwargs) == 2 and not c2.kwargs
