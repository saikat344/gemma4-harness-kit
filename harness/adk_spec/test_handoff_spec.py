"""Characterization tests for what a second agent sees after a hand-off.

Same contract as test_compaction_spec.py: every test asserts what the pinned
scorer stack (google-adk 1.36.1) does today (docs/harness-contracts.md §7),
and run_matrix.py runs it against other ADK versions. A submission can add a
second LlmAgent (to carry its own model or LoRA adapter) in two ways:

  transfer   root has `sub_agents`; the model calls `transfer_to_agent`.
             The sub-agent shares the session, but every event authored by
             another agent is rewritten as role=user `For context:` text
             (flows/llm_flows/contents.py `_present_other_agent_message`),
             and that agent's thoughts are dropped.
  agent_tool root wraps the sub-agent in AgentTool (the official sample's
             LoRA pattern). The sub-agent runs in a fresh session: its own
             instruction (session state templated) and the request string.

Markers: TASK_MARKER (user task), BASE_THOUGHT / FIXER_THOUGHT (thought text),
PD_MARKER (session state `problem_description`), RESULT_<path> (tool result),
REQUEST_MARKER (AgentTool request string).
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import PrivateAttr

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

APP = "spec"
USER = "u"


class ScriptedLlm(BaseLlm):
  """Returns its script one response per call and records every request."""

  _script: list = PrivateAttr(default_factory=list)
  _log: list = PrivateAttr(default_factory=list)

  def __init__(self, model: str, script: list[list[types.Part]]):
    super().__init__(model=model)
    self._script = list(script)

  async def generate_content_async(self, llm_request, stream: bool = False):
    cfg = llm_request.config
    self._log.append({
        "contents": [c.model_copy(deep=True) for c in llm_request.contents],
        "system": str(cfg.system_instruction) if cfg and cfg.system_instruction else "",
    })
    parts = self._script.pop(0) if self._script else [types.Part(text="DONE")]
    yield LlmResponse(content=types.Content(role="model", parts=parts))


def _thought(text: str) -> types.Part:
  return types.Part(text=text, thought=True)


def _call(name: str, **args) -> types.Part:
  return types.Part(function_call=types.FunctionCall(name=name, args=args))


def read_file(filepath: str) -> dict:
  """Reads a file from the workspace."""
  return {"content": f"RESULT_{filepath}"}


def _run(root: LlmAgent, state: dict | None = None) -> None:
  service = InMemorySessionService()
  runner = Runner(app_name=APP, agent=root, session_service=service)

  async def go():
    session = await service.create_session(
        app_name=APP, user_id=USER, state=state or {}
    )
    msg = types.Content(role="user", parts=[types.Part(text="TASK_MARKER fix a.py")])
    async for _ in runner.run_async(
        user_id=USER, session_id=session.id, new_message=msg
    ):
      pass

  asyncio.run(go())


def _parts(req: dict):
  return [(c.role, p) for c in req["contents"] for p in c.parts or []]


def _texts(req: dict) -> list[str]:
  return [p.text for _, p in _parts(req) if p.text]


# --- transfer_to_agent -------------------------------------------------------


@pytest.fixture(scope="module")
def transfer():
  fixer_llm = ScriptedLlm("fixer-llm", [
      [_thought("FIXER_THOUGHT"), _call("read_file", filepath="b.py")],
      [_call("transfer_to_agent", agent_name="base")],
  ])
  base_llm = ScriptedLlm("base-llm", [
      [_thought("BASE_THOUGHT"), _call("read_file", filepath="a.py")],
      [_call("transfer_to_agent", agent_name="fixer")],
      [types.Part(text="base done")],
  ])
  fixer = LlmAgent(name="fixer", description="edits files", model=fixer_llm,
                   instruction="FIXER_SYSTEM", tools=[read_file])
  base = LlmAgent(name="base", model=base_llm, instruction="BASE_SYSTEM",
                  tools=[read_file], sub_agents=[fixer])
  _run(base)
  return {"fixer": fixer_llm._log, "base": base_llm._log}


def test_h01_transfer_other_agents_calls_become_user_text(transfer):
  first = transfer["fixer"][0]
  assert any(t.startswith("For context:") for t in _texts(first))
  assert any("read_file" in t and "a.py" in t for t in _texts(first)[1:])
  # No typed function_call/function_response parts from the base agent remain.
  assert not [p for _, p in _parts(first) if p.function_call or p.function_response]
  assert {role for role, _ in _parts(first)} == {"user"}


def test_h02_transfer_tool_results_survive_as_text(transfer):
  first = transfer["fixer"][0]
  assert any("[base] `read_file` tool returned result:" in t and "RESULT_a.py" in t
             for t in _texts(first))


def test_h03_transfer_other_agents_thinking_dropped(transfer):
  assert not any("BASE_THOUGHT" in t for t in _texts(transfer["fixer"][0]))


def test_h04_transfer_back_rewrites_the_sub_agents_turns_too(transfer):
  after = transfer["base"][-1]
  texts = _texts(after)
  assert any("RESULT_b.py" in t for t in texts)
  assert not [p for c in after["contents"] for p in c.parts or []
              if p.function_call and p.function_call.args.get("filepath") == "b.py"]
  assert not any("FIXER_THOUGHT" in t for t in texts)
  # The base agent's own earlier thought is still native.
  assert any(p.thought and p.text == "BASE_THOUGHT" for _, p in _parts(after))


def test_h07_transfer_wording_is_python_repr_lines(transfer):
  # 1.36.1 wording: one `For context:` part, then one part per call or result,
  # args and results as Python dict repr (not JSON).
  texts = _texts(transfer["fixer"][0])
  assert "[base] called tool `read_file` with parameters: {'filepath': 'a.py'}" in texts
  assert "[base] `read_file` tool returned result: {'content': 'RESULT_a.py'}" in texts


# --- AgentTool -----------------------------------------------------------------


@pytest.fixture(scope="module")
def agent_tool():
  fixer_llm = ScriptedLlm("fixer-llm", [
      [_thought("FIXER_THOUGHT"), _call("read_file", filepath="b.py")],
      [types.Part(text="fixer answer")],
  ])
  base_llm = ScriptedLlm("base-llm", [
      [_thought("BASE_THOUGHT"), _call("read_file", filepath="a.py")],
      [_call("fixer", request="REQUEST_MARKER edit a.py")],
      [types.Part(text="base done")],
  ])
  fixer = LlmAgent(name="fixer", description="edits files", model=fixer_llm,
                   instruction="FIXER_SYSTEM {problem_description}", tools=[read_file])
  base = LlmAgent(name="base", model=base_llm, instruction="BASE_SYSTEM",
                  tools=[read_file, AgentTool(fixer, skip_summarization=True)])
  _run(base, state={"problem_description": "PD_MARKER"})
  return {"fixer": fixer_llm._log, "base": base_llm._log}


def test_h05_agent_tool_sees_only_the_request_string(agent_tool):
  first = agent_tool["fixer"][0]
  assert _texts(first) == ["REQUEST_MARKER edit a.py"]
  blob = "\n".join(_texts(first))
  assert "TASK_MARKER" not in blob and "RESULT_a.py" not in blob


def test_h06_agent_tool_instruction_templates_session_state(agent_tool):
  assert "PD_MARKER" in agent_tool["fixer"][0]["system"]
