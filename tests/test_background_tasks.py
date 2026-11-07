"""BackgroundTaskMiddleware: task_start returns at once, task_collect hands back finished replies once."""
from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from EvoScientist.middleware.background_tasks import COLLECT, START, BackgroundTaskMiddleware


def _task_tool() -> StructuredTool:
    return StructuredTool.from_function(func=lambda **_: "", name="task", description="t", infer_schema=False,
                                        args_schema=None)


def _req(name: str, args: dict, cid: str = "c1") -> ToolCallRequest:
    return ToolCallRequest(tool_call={"name": name, "args": args, "id": cid, "type": "tool_call"}, tool=None,
                           state={}, runtime=None)


def _armed(tmp_path) -> BackgroundTaskMiddleware:
    mw = BackgroundTaskMiddleware(tmp_path)
    mw._task_tool = _task_tool()
    return mw


def _sync_handler(delays: dict[str, float], seen: list):
    def handler(request):
        args = request.tool_call["args"]
        seen.append((request.tool_call["name"], args["subagent_type"]))
        time.sleep(delays.get(args["subagent_type"], 0))
        return ToolMessage(content=f"reply of {args['subagent_type']}", tool_call_id=request.tool_call["id"])
    return handler


def test_sync_start_returns_at_once_and_collect_gets_each_reply_once(tmp_path):
    mw, seen = _armed(tmp_path), []
    h = _sync_handler({"slow": 0.6, "fast": 0.0}, seen)
    t0 = time.monotonic()
    r1 = mw.wrap_tool_call(_req(START, {"subagent_type": "slow", "description": "d"}, "a"), h)
    r2 = mw.wrap_tool_call(_req(START, {"subagent_type": "fast", "description": "d"}, "b"), h)
    assert time.monotonic() - t0 < 0.3 and "bg-1" in r1.content and "bg-2" in r2.content
    out = mw.wrap_tool_call(_req(COLLECT, {"wait_minutes": 0.1}), h).content          # first result: the fast one
    assert "reply of fast" in out and "reply of slow" not in out and "Still running: bg-1" in out
    out = mw.wrap_tool_call(_req(COLLECT, {"wait_minutes": 0.1}), h).content
    assert "reply of slow" in out and "reply of fast" not in out                       # reported once only
    assert all(c[0] == "task" for c in seen)                                           # the real task tool ran
    assert "No background jobs" in mw.wrap_tool_call(_req(COLLECT, {}), h).content


def test_collect_waits_no_longer_than_asked_and_never_blocks_with_nothing_running(tmp_path):
    mw = _armed(tmp_path)
    t0 = time.monotonic()
    assert "No background jobs" in mw.wrap_tool_call(_req(COLLECT, {"wait_minutes": 5}), lambda r: None).content
    assert time.monotonic() - t0 < 1
    gate = threading.Event()
    def blocked(request):
        gate.wait(5)
        return ToolMessage(content="late", tool_call_id="x")
    mw.wrap_tool_call(_req(START, {"subagent_type": "s", "description": "d"}), blocked)
    t0 = time.monotonic()
    out = mw.wrap_tool_call(_req(COLLECT, {"wait_minutes": 0.02}), blocked).content    # 1.2 s
    assert 1.0 < time.monotonic() - t0 < 3 and "Still running: bg-1" in out
    gate.set()


def test_async_path_runs_jobs_side_by_side(tmp_path):
    async def go():
        mw = _armed(tmp_path)
        async def h(request):
            await asyncio.sleep(0.4)
            return Command(update={"messages": [ToolMessage(content=f"R-{request.tool_call['args']['subagent_type']}",
                                                            tool_call_id=request.tool_call["id"])]})
        t0 = time.monotonic()
        for i, s in enumerate(("a", "b", "c")):
            await mw.awrap_tool_call(_req(START, {"subagent_type": s, "description": "d"}, f"id{i}"), h)
        assert time.monotonic() - t0 < 0.3
        out = (await mw.awrap_tool_call(_req(COLLECT, {"wait_minutes": 0.5}), h)).content
        await asyncio.sleep(0.6)
        out += (await mw.awrap_tool_call(_req(COLLECT, {}), h)).content
        assert all(f"R-{s}" in out for s in "abc")
        assert time.monotonic() - t0 < 1.5                                              # not 3 x 0.4 in sequence
    asyncio.run(go())


def test_a_failing_job_is_a_result_and_a_restart_marks_running_jobs_lost(tmp_path):
    mw = _armed(tmp_path)
    def boom(request):
        raise RuntimeError("model gave up")
    mw.wrap_tool_call(_req(START, {"subagent_type": "s", "description": "d"}), boom)
    out = mw.wrap_tool_call(_req(COLLECT, {"wait_minutes": 0.1}), boom).content
    assert "failed" in out and "model gave up" in out
    gate = threading.Event()
    mw.wrap_tool_call(_req(START, {"subagent_type": "s2", "description": "d"}), lambda r: (gate.wait(5), ToolMessage(content="x", tool_call_id="i"))[1])
    time.sleep(0.2)
    saved = json.loads((tmp_path / ".research" / "background" / "bg-2.json").read_text())
    assert saved["status"] == "running"
    again = BackgroundTaskMiddleware(tmp_path)                                          # the process restarted
    out = again.wrap_tool_call(_req(COLLECT, {}), lambda r: None).content
    assert "bg-2" in out and "lost" in out and "bg-1" not in out
    assert again._n == 2                                                                # ids continue, never reused
    gate.set()


def test_start_without_the_task_tool_or_arguments_is_an_error_not_a_hang(tmp_path):
    mw = BackgroundTaskMiddleware(tmp_path)
    r = mw.wrap_tool_call(_req(START, {"subagent_type": "s", "description": "d"}), lambda r: None)
    assert r.status == "error"
    mw._task_tool = _task_tool()
    assert mw.wrap_tool_call(_req(START, {"subagent_type": "s"}), lambda r: None).status == "error"
    other = mw.wrap_tool_call(_req("tree_status", {}), lambda r: ToolMessage(content="ok", tool_call_id="c1"))
    assert other.content == "ok"                                                         # other tools pass through


def test_model_call_remembers_the_task_tool(tmp_path):
    mw = BackgroundTaskMiddleware(tmp_path)
    t = _task_tool()
    req = ModelRequest(model=None, messages=[], tools=[t])
    mw.wrap_model_call(req, lambda r: "ok")
    assert mw._task_tool is t
