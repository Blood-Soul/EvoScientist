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


# --------------------------------------------------------------- automatic analysis


class _Hub:
    """Just the hub surface the auto-start paths touch: notes out, open nodes and their jobs."""

    def __init__(self, open_nodes=()) -> None:
        self.notes: list[str] = []
        self.open = set(open_nodes)
        self.jobs: dict[str, str] = {}

    def _emit(self, channel, text):
        self.notes.append(text)

    def job_ended(self, node, job):
        pass

    def job_of(self, node):
        return self.jobs.get(node)

    def is_open(self, node):
        return node in self.open

    def attach_job(self, node, job):
        self.jobs[node] = job


def _tree(tmp_path, nodes: dict, analyses: dict | None = None, seconds_left: float = 3600) -> None:
    (tmp_path / ".research").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".research" / "state.json").write_text(json.dumps({
        "deadline": time.time() + seconds_left, "nodes": nodes, "analyses": analyses or {},
    }))


def _scored(status: str = "done") -> dict:
    return {"status": status, "score": 0.5}


def test_analysis_is_due_only_for_scored_unanalysed_nodes(tmp_path):
    _tree(tmp_path, {"n1": _scored(), "n2": _scored(), "n3": {"status": "failed", "score": None},
                     "n4": {"status": "running", "score": None}}, analyses={"n2": {}})
    mw = BackgroundTaskMiddleware(tmp_path, hub=_Hub(), auto_analysis=True)
    assert mw._analysis_due() == ["n1"]


def test_nothing_starts_in_the_last_minutes(tmp_path):
    _tree(tmp_path, {"n1": _scored()}, seconds_left=60)
    mw = BackgroundTaskMiddleware(tmp_path, hub=_Hub(), auto_analysis=True)
    assert mw._analysis_due() == []


def test_closing_a_node_starts_its_analysis_and_notes_the_main_agent(tmp_path):
    import asyncio
    from types import SimpleNamespace

    _tree(tmp_path, {"n1": _scored()})
    hub, task = _Hub(), _Task()
    mw = BackgroundTaskMiddleware(tmp_path, hub=hub, auto_analysis=True)
    mw._task_tool = task

    async def handler(request):
        return "node closed"

    request = SimpleNamespace(tool_call={"name": "tree_finish", "args": {"node": "n1"}, "id": "c1"},
                              runtime=_runtime("c1"))
    asyncio.run(mw.awrap_tool_call(request, handler))
    assert task.calls
    assert task.calls[0][0] == "node-analyst"
    assert task.calls[0][1].startswith("n1")
    assert any("Analysis of n1 started" in note for note in hub.notes)


def test_an_abandoned_node_starts_no_analysis_and_a_done_job_is_not_restarted(tmp_path):
    import asyncio
    from types import SimpleNamespace

    _tree(tmp_path, {"n1": _scored(), "n2": {"status": "abandoned", "score": None}})
    hub, task = _Hub(), _Task()
    mw = BackgroundTaskMiddleware(tmp_path, hub=hub, auto_analysis=True)
    mw._task_tool = task

    async def handler(request):
        return "closed"

    runtime = _runtime("c1")
    abandoned = SimpleNamespace(tool_call={"name": "tree_finish", "args": {"node": "n2", "abandon_reason": "x"},
                                           "id": "c2"}, runtime=runtime)
    asyncio.run(mw.awrap_tool_call(abandoned, handler))
    assert task.calls == []

    first = SimpleNamespace(tool_call={"name": "tree_finish", "args": {"node": "n1"}, "id": "c1"}, runtime=runtime)
    asyncio.run(mw.awrap_tool_call(first, handler))
    mw.jobs[next(iter(mw.jobs))]["status"] = "done"   # the job finished; the file is not recorded yet
    asyncio.run(mw.awrap_tool_call(first, handler))
    assert [c[0] for c in task.calls] == ["node-analyst"]


def test_opening_a_node_starts_its_agent(tmp_path):
    import asyncio
    from types import SimpleNamespace

    _tree(tmp_path, {"n1": {"status": "running", "score": None}})
    (tmp_path / ".research" / "briefs").mkdir(parents=True, exist_ok=True)
    hub, task = _Hub(open_nodes={"n1"}), _Task()
    mw = BackgroundTaskMiddleware(tmp_path, hub=hub, auto_node_agent=True)
    mw._task_tool = task

    async def handler(request):
        return "n1 opened"

    request = SimpleNamespace(tool_call={"name": "tree_new", "args": {"plan_id": "P1"}, "id": "c9"},
                              runtime=_runtime("c9"))
    asyncio.run(mw.awrap_tool_call(request, handler))
    assert [c[0] for c in task.calls] == ["node-agent"]
    assert task.calls[0][1].startswith("n1")
    assert "nodes/n1/BRIEF.md" in task.calls[0][1]
    assert any("node-agent started on n1" in note for note in hub.notes)


def test_a_node_that_already_has_an_agent_is_not_started_again(tmp_path):
    _tree(tmp_path, {"n1": {"status": "running", "score": None}})
    hub = _Hub(open_nodes={"n1"})
    hub.jobs["n1"] = "bg-1"
    mw = BackgroundTaskMiddleware(tmp_path, hub=hub, auto_node_agent=True)
    mw._task_tool = _Task()
    assert mw._nodes_without_agent() == []
