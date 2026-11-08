"""Background sub-agents on the real main graph, run the way the CLI runs it (astream_events v3 + checkpointer)."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config

from EvoScientist.config import EvoScientistConfig
from EvoScientist.EvoScientist import create_cli_agent
from EvoScientist.middleware.model_retry import ModelRetryMiddleware
from EvoScientist.paths import Workspace
from EvoScientist.stream.events import stream_agent_events

pytestmark = pytest.mark.usefixtures("no_mcp")


@pytest.fixture(autouse=True)
def test_config(monkeypatch):
    monkeypatch.setattr("EvoScientist.EvoScientist._config", EvoScientistConfig(enable_async_subagents=False))


@pytest.fixture(autouse=True)
def no_retry_waits(monkeypatch):
    build = ModelRetryMiddleware.__init__

    async def _sleep(seconds):
        return None

    def _build(self, **kwargs):
        build(self, **{**kwargs, "sleep": _sleep, "rng": lambda: 0.5})

    monkeypatch.setattr(ModelRetryMiddleware, "__init__", _build)


def _text(text):
    return ChatGenerationChunk(message=AIMessageChunk(content=text))


_IDS = iter(range(10**6))


def _calls(*calls):
    chunks = [{"name": n, "args": json.dumps(a), "id": f"call_{next(_IDS)}_{n}", "index": i} for i, (n, a) in enumerate(calls)]
    return ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks))


class Scripted(GenericFakeChatModel):
    """Main agent follows `script` (one entry per model call); a sub-agent sleeps `sub_seconds[desc]` then answers."""

    script: list = []
    sub_seconds: dict = {}
    main_calls: int = 0
    sub_spans: list = []

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        metadata = get_config().get("metadata") or {}
        if metadata.get("lc_agent_name") == "general-purpose":
            desc = next(m.text for m in messages if m.type == "human")
            t0 = time.monotonic()
            await asyncio.sleep(self.sub_seconds.get(desc, 0.0))
            self.sub_spans.append((desc, t0, time.monotonic()))
            yield _text(f"reply to {desc}")
            return
        step = self.script[min(self.main_calls, len(self.script) - 1)]
        self.main_calls += 1
        yield step

    def bind_tools(self, tools, **kwargs):
        return self


def _agent(tmp_path, model):
    cfg = EvoScientistConfig(enable_async_subagents=False, research_tree_background_tasks=True)
    return create_cli_agent(workspace=Workspace(tmp_path), config=cfg, chat_model=model, checkpointer=InMemorySaver())


def _tool_results(events, name):
    return [e for e in events if e.get("type") == "tool_result" and e.get("name") == name]


async def test_two_jobs_overlap_and_their_replies_come_back_through_collect(tmp_path):
    model = Scripted(messages=iter([]))
    model.script = [
        _calls(("task_start", {"subagent_type": "general-purpose", "description": "A"}),
               ("task_start", {"subagent_type": "general-purpose", "description": "B"})),
        _calls(("task_collect", {"wait_minutes": 0.1})),
        _calls(("task_collect", {"wait_minutes": 0.1})),
        _text("All done."),
    ]
    model.sub_seconds = {"A": 1.0, "B": 2.0}
    agent = _agent(tmp_path, model)
    t0 = time.monotonic()
    events = [e async for e in stream_agent_events(agent, "go", "t1")]
    elapsed = time.monotonic() - t0

    starts = _tool_results(events, "task_start")
    assert [e["content"] for e in starts] == [
        "Started bg-1 (general-purpose). It runs in the background; 1 job(s) running now. "
        "Call task_collect to get the reply when you need it.",
        "Started bg-2 (general-purpose). It runs in the background; 2 job(s) running now. "
        "Call task_collect to get the reply when you need it.",
    ]
    collects = [e["content"] for e in _tool_results(events, "task_collect")]
    assert "reply to A" in collects[0] and "Still running: bg-2" in collects[0]
    assert "reply to B" in collects[1] and "reply to A" not in collects[1]      # each reply once
    spans = {d: (a, b) for d, a, b in model.sub_spans}
    assert spans["B"][0] < spans["A"][1]                                      # B started before A finished
    assert elapsed < 3.0                                                      # not 1.0 + 2.0 in sequence
    assert next(e for e in events if e["type"] == "done")["response"] == "All done."
    state = await agent.aget_state({"configurable": {"thread_id": "t1"}})
    assert not state.next and not state.tasks                                 # nothing left parked


async def test_a_run_that_ends_while_a_job_runs_waits_for_it_and_hands_back_the_reply(tmp_path):
    model = Scripted(messages=iter([]))
    model.script = [
        _calls(("task_start", {"subagent_type": "general-purpose", "description": "slow"})),
        _text("I will wait for it later."),
    ]
    model.sub_seconds = {"slow": 2.0}
    agent = _agent(tmp_path, model)
    t0 = time.monotonic()
    events = [e async for e in stream_agent_events(agent, "go", "t2")]
    elapsed = time.monotonic() - t0
    assert next(e for e in events if e["type"] == "done")
    assert elapsed >= 1.8                                   # the run did not end at once and drop the job
    state = await agent.aget_state({"configurable": {"thread_id": "t2"}})
    notes = [m for m in state.values["messages"] if m.type == "human" and "background work" in m.text]
    assert notes and "reply to slow" in notes[0].text     # the reply reached the model before the end
    saved = json.loads((tmp_path / ".research" / "background" / "bg-1.json").read_text())
    assert saved["status"] == "done" and saved["reported"] is True


async def test_a_sub_agent_that_cannot_run_is_reported_not_lost(tmp_path):
    """The task tool answers an unusable sub-agent with its error text; the job must still come back."""
    model = Scripted(messages=iter([]))
    model.script = [
        _calls(("task_start", {"subagent_type": "no-such-agent", "description": "x"})),
        _text("ok"),
    ]
    agent = _agent(tmp_path, model)
    events = [e async for e in stream_agent_events(agent, "go", "t3")]
    state = await agent.aget_state({"configurable": {"thread_id": "t3"}})
    notes = [m.text for m in state.values["messages"] if m.type == "human" and "background work" in m.text]
    assert notes and "bg-1" in notes[0] and "writing-agent" in notes[0]   # the error lists the real ones
    assert next(e for e in events if e["type"] == "done")
