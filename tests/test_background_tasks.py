"""BackgroundTaskMiddleware: task_start returns at once, task_collect hands back each finished reply once."""
from __future__ import annotations

import asyncio
import json
import threading
import time

from langchain.agents.middleware.types import ModelRequest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage

from EvoScientist.middleware.background_tasks import BackgroundTaskMiddleware


def _runtime(call_id: str = "c1") -> ToolRuntime:
    return ToolRuntime(state={}, context=None, config={}, stream_writer=lambda *a, **k: None,
                       tool_call_id=call_id, store=None)


class _Task:
    """Stands in for deepagents' `task` tool: both entry points, and it records the call id it was given."""

    def __init__(self, delays: dict[str, float] | None = None, fails: tuple[str, ...] = (),
                 block: dict[str, threading.Event] | None = None) -> None:
        self.delays = delays or {}
        self.fails = set(fails)
        self.block = block or {}
        self.calls: list[tuple[str, str, str | None]] = []

    def _note(self, description: str, subagent_type: str, runtime: ToolRuntime) -> None:
        self.calls.append((subagent_type, description, runtime.tool_call_id))
        if subagent_type in self.fails:
            raise RuntimeError("model gave up")
        if (gate := self.block.get(description)) is not None:
            gate.wait(5)

    def func(self, description: str, subagent_type: str, runtime: ToolRuntime) -> str:
        self._note(description, subagent_type, runtime)
        time.sleep(self.delays.get(description, 0.0))
        return f"reply of {description}"

    async def coroutine(self, description: str, subagent_type: str, runtime: ToolRuntime) -> str:
        self._note(description, subagent_type, runtime)
        await asyncio.sleep(self.delays.get(description, 0.0))
        return f"reply of {description}"


def _armed(tmp_path, task: _Task | None = None) -> BackgroundTaskMiddleware:
    mw = BackgroundTaskMiddleware(tmp_path)
    mw._task_tool = task or _Task()
    return mw


def test_start_returns_at_once_and_each_reply_is_reported_once(tmp_path):
    task = _Task({"slow": 0.5})
    mw = _armed(tmp_path, task)
    t0 = time.monotonic()
    r1 = mw._start_sync("general-purpose", "slow", _runtime("c1"))
    r2 = mw._start_sync("general-purpose", "fast", _runtime("c2"))
    assert time.monotonic() - t0 < 0.2 and "bg-1" in r1 and "bg-2" in r2
    out = mw._collect_sync(0.1)                                    # the first to finish comes back alone
    assert "reply of fast" in out and "reply of slow" not in out and "Still running: bg-1" in out
    out = mw._collect_sync(0.1)
    assert "reply of slow" in out and "reply of fast" not in out   # reported once only
    assert "No background jobs" in mw._collect_sync(0)
    # The real task tool ran once per job, under a call id of its own (not task_start's).
    assert [c[2] for c in task.calls] == ["c1-bg-1", "c2-bg-2"]


def test_collect_never_blocks_when_nothing_is_running(tmp_path):
    mw = _armed(tmp_path)
    t0 = time.monotonic()
    assert "No background jobs" in mw._collect_sync(5)
    assert time.monotonic() - t0 < 1


def test_async_start_runs_jobs_side_by_side(tmp_path):
    async def go():
        mw = _armed(tmp_path, _Task({"a": 0.4, "b": 0.4, "c": 0.4}))
        t0 = time.monotonic()
        for i, d in enumerate("abc"):
            text = await mw._start_async("general-purpose", d, _runtime(f"id{i}"))
            assert f"bg-{i + 1}" in text
        assert time.monotonic() - t0 < 0.2
        await asyncio.sleep(0.5)
        out = await mw._collect_async(0)
        assert all(f"reply of {d}" in out for d in "abc")
        assert time.monotonic() - t0 < 1.2                     # not 3 x 0.4 one after another
    asyncio.run(go())


def test_a_failing_job_is_a_result_the_coordinator_has_to_see(tmp_path):
    mw = _armed(tmp_path, _Task(fails=("boom",)))
    mw._start_sync("general-purpose", "ok", _runtime("c1"))
    assert "reply of ok" in mw._collect_sync(0.1)
    mw._start_sync("boom", "d", _runtime("c2"))                # the thread raises before it can answer
    out = mw._collect_sync(0.1)
    assert "bg-2" in out and "failed" in out and "model gave up" in out


def test_a_restart_reports_an_interrupted_job_as_lost_and_continues_the_ids(tmp_path):
    gate = threading.Event()
    mw = _armed(tmp_path, _Task(block={"stuck": gate}))
    mw._start_sync("general-purpose", "stuck", _runtime("c1"))
    time.sleep(0.2)
    saved = json.loads((tmp_path / ".research" / "background" / "bg-1.json").read_text())
    assert saved["status"] == "running"
    again = BackgroundTaskMiddleware(tmp_path)                 # the process restarted
    out = again._collect_sync(0)
    assert "bg-1" in out and "lost" in out
    assert again._n == 1                                       # the next job is bg-2, never a reused id
    gate.set()


def test_start_without_the_task_tool_or_arguments_is_an_error_not_a_hang(tmp_path):
    mw = BackgroundTaskMiddleware(tmp_path)
    assert "not available" in mw._start_sync("x", "d", _runtime())
    assert "not available" in asyncio.run(mw._start_async("x", "d", _runtime()))
    mw._task_tool = _Task()
    assert "required" in mw._start_sync("", "d", _runtime())


def test_the_model_call_finds_the_task_tool_and_is_told_what_finished(tmp_path):
    mw = BackgroundTaskMiddleware(tmp_path)
    assert mw.wrap_model_call(ModelRequest(model=None, messages=[], tools=[]), lambda r: r).system_message is None

    task = _Task()
    task.name = "task"                                          # a stand-in, matched by name only
    mw.wrap_model_call(ModelRequest(model=None, messages=[], tools=[task]), lambda r: r)
    assert mw._task_tool is task

    mw._start_sync("general-purpose", "done", _runtime("c1"))
    time.sleep(0.2)                                             # finished, not collected
    request = mw.wrap_model_call(
        ModelRequest(model=None, messages=[], tools=[task]),
        lambda r: r,
    )
    note = request.system_message.text
    assert "Finished and not collected yet: bg-1" in note and "task_collect" in note


def test_the_run_cannot_end_while_a_job_is_open(tmp_path):
    mw = _armed(tmp_path, _Task({"slow": 0.3}))
    assert mw.after_agent({"messages": [AIMessage("done")]}, None) is None   # nothing open: it ends
    mw._start_sync("general-purpose", "slow", _runtime("c1"))
    t0 = time.monotonic()
    jump = mw.after_agent({"messages": [AIMessage("done")]}, None)
    assert time.monotonic() - t0 >= 0.25                        # it waited for the job
    assert jump["jump_to"] == "model" and len(jump["messages"]) == 1
    text = jump["messages"][0].text
    assert "bg-1" in text and "reply of slow" in text
    assert mw.after_agent({"messages": [AIMessage("done")]}, None) is None   # collected now: it ends
