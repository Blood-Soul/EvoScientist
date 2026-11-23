"""Let the coordinator hand work to any sub-agent without waiting for it.

`task` blocks the caller until the sub-agent returns. In a run where several pieces of work are
independent (a plan for one direction while a node trains, plans for two directions at once), that
makes the slowest piece the clock for all of them. This middleware gives the main agent two tools:

- `task_start(subagent_type, description)` starts the sub-agent through the real `task` tool in the
  background and returns a job id at once. It works for every sub-agent type `task` can reach.
- `task_collect(wait_minutes)` returns the jobs that have finished since the last call, with their full
  replies, and says what is still running. It waits at most `wait_minutes` for a first result.

Both are ordinary tools, executed by the tool node, so they appear in the event stream like any other.
A background job calls the `task` tool's own function with a tool-call id of its own, so its sub-agent
is not mistaken for a result of the `task_start` call.

Two things keep work from being lost:

- Before each model call, the agent is told which jobs have finished and not been collected.
- The agent cannot end its run while a job is running or a finished reply is uncollected: the run waits
  for the next job to finish and goes back to the model with the replies. A non-interactive run is a
  single turn, so ending it would otherwise kill the jobs.

Results are also written to `.research/background/<job>.json`; after a restart a job that was running is
reported as lost. No time limit is set here; the run's own clock bounds the work. State updates a
sub-agent would return through `task` (private channels) are dropped: its files and its reply are what
the coordinator uses.

Main agent only, and only when `research_tree_background_tasks` is on.

In a skeleton run (a coordination hub is given, see :mod:`EvoScientist.research_tree.coord`) three things change:

- A `node-agent` job names its node on the first line of its description (`n3`, or `node: n3`). The job runs with
  that node in its context, so the node agent's GPU lease, questions and shell environment are the node's. One
  node has one agent at a time.
- What finished is a note at the end of the context (:mod:`.coord_notes`), not text added to the system prompt,
  so the prompt cache stays valid.
- `task_collect` also returns when the hub has a note for the main agent (a question, a GPU request, a free GPU),
  so a node that waits for an answer is never stuck behind a long wait.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import json
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, Field

from ..research_tree.coord import CURRENT_NODE, MAIN

START = "task_start"
COLLECT = "task_collect"
POLL_SECONDS = 0.2  # how soon a finished job is noticed while waiting
MAX_WAIT_MINUTES = 15.0  # one wait never blocks longer; the agent (or the end-of-run guard) waits again
NODE_AGENT = "node-agent"
_NODE_LINE = re.compile(r"^\s*(?:node\s*[:=]?\s*)?(n\d+)\b", re.IGNORECASE)


def node_of(description: str) -> str:
    """The node a node-agent description names on its first line, or ''."""
    m = _NODE_LINE.match((description or "").strip().splitlines()[0] if (description or "").strip() else "")
    return m.group(1).lower() if m else ""


class StartArgs(BaseModel):
    subagent_type: str = Field(description="The sub-agent to run, as for `task`.")
    description: str = Field(
        description="Everything the sub-agent needs, as for `task`. It starts at once and you do not wait for it."
    )


class CollectArgs(BaseModel):
    wait_minutes: float = Field(
        default=0.0,
        description="Minutes to wait for a first result when nothing has finished yet (0 returns at once).",
    )


def _text_of(result: Any) -> str:
    """The reply text of what the `task` tool returned (a string, a ToolMessage, or a Command carrying one)."""
    msg = result
    if isinstance(result, Command):
        update = result.update if isinstance(result.update, dict) else {}
        msgs = update.get("messages") or []
        msg = msgs[-1] if msgs else None
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def _wait_of(args: float | None) -> float:
    return min(max(float(args or 0), 0.0), MAX_WAIT_MINUTES) * 60


class BackgroundTaskMiddleware(AgentMiddleware):
    """Main agent only: start sub-agents in the background and collect what they return."""

    name = "background_tasks"

    def __init__(self, root: str | Path, *, now: Callable[[], float] = time.time, hub=None) -> None:
        super().__init__()
        self.dir = Path(root) / ".research" / "background"
        self.now = now
        self.hub = hub
        self.jobs: dict[str, dict[str, Any]] = {}
        self._handles: dict[str, Any] = {}  # job id -> asyncio.Task or Thread (kept so a task is not collected)
        self._task_tool: Any = None
        self._lock = threading.Lock()
        self._n = self._recover()
        self.tools = [
            StructuredTool.from_function(
                func=self._start_sync,
                coroutine=self._start_async,
                name=START,
                description=(
                    "Run a sub-agent in the background and get a job id back at once, instead of waiting as `task` "
                    "does. Use it when the work does not have to be finished before your next decision: a plan for "
                    "another direction while a node runs, plans for several directions together. Collect the replies "
                    f"with `{COLLECT}`. Works for every sub-agent that `task` can reach."
                ),
                args_schema=StartArgs,
                infer_schema=False,
            ),
            StructuredTool.from_function(
                func=self._collect_sync,
                coroutine=self._collect_async,
                name=COLLECT,
                description=(
                    "Return the background jobs that have finished since your last call, with their full replies, and "
                    "say what is still running. Give `wait_minutes` to wait for a first result when none has finished "
                    "yet. A job that is still running is not lost: call again later."
                ),
                args_schema=CollectArgs,
                infer_schema=False,
            ),
        ]

    # ------------------------------------------------------------------ persistence

    def _recover(self) -> int:
        n = 0
        if self.dir.is_dir():
            for f in sorted(self.dir.glob("bg-*.json")):
                try:
                    rec = json.loads(f.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                n = max(n, int(rec.get("n", 0)))
                if rec.get("status") == "running":
                    rec.update(status="lost", reported=False,
                               text="The agent process stopped while this job was running; its reply was not kept.")
                    self._write(rec)
                self.jobs[rec["id"]] = rec
        return n

    def _write(self, rec: dict[str, Any]) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.dir / f"{rec['id']}.json.tmp"
            tmp.write_text(json.dumps(rec, ensure_ascii=False))
            tmp.replace(self.dir / f"{rec['id']}.json")
        except OSError:
            pass  # the in-memory record is still right; only the restart copy is missing

    # ------------------------------------------------------------------ bookkeeping

    def _running(self) -> list[dict[str, Any]]:
        return [j for j in self.jobs.values() if j["status"] == "running"]

    def _unreported(self) -> list[dict[str, Any]]:
        return [j for j in self.jobs.values() if j["status"] != "running" and not j["reported"]]

    def _new_job(self, sub: str, desc: str, tool_call_id: str = "") -> dict[str, Any]:
        with self._lock:
            self._n += 1
            rec = {"id": f"bg-{self._n}", "n": self._n, "subagent": sub, "description": desc[:300],
                   "tool_call_id": tool_call_id,
                   "started": self.now(), "status": "running", "reported": False, "text": ""}
            self.jobs[rec["id"]] = rec
        self._write(rec)
        return rec

    def _finish(self, rec: dict[str, Any], text: str, status: str, writer=None) -> None:
        with self._lock:
            rec.update(status=status, text=text, finished=self.now(), reported=False)
        self._write(rec)
        if self.hub is not None:
            if rec.get("node"):
                self.hub.job_ended(rec["node"], rec["id"])
            self.hub._emit(MAIN, f"Background job {rec['id']} ({rec['subagent']}"
                           + (f", {rec['node']}" if rec.get("node") else "")
                           + f") {status}. Read its reply with {COLLECT}.")
        self._announce(rec, writer)

    @staticmethod
    def _announce(rec: dict[str, Any], writer) -> None:
        """Tell the run's event stream that this job is over, while the run is still live.

        A background job is not a task of the parent graph, so LangGraph cannot
        close its sub-agent handle from a task result: left to the handle, the
        end arrives only at run teardown and the UI shows every finished
        sub-agent as running for the rest of the run. The stream writer is the
        one channel still open inside the job. Never raises: a missing or
        closed stream is not a reason to lose the job's reply.
        """
        if writer is None:
            return
        try:
            writer({
                "type": "background_job_finished",
                "job": rec["id"],
                "tool_call_id": rec.get("tool_call_id") or "",
                "status": rec["status"],
            })
        except Exception:
            pass

    def _take_finished(self) -> list[dict[str, Any]]:
        with self._lock:
            done = self._unreported()
            for j in done:
                j["reported"] = True
        for j in done:
            self._write(j)
        return done

    def _report(self, done: list[dict[str, Any]]) -> str:
        now = self.now()
        parts = []
        for j in done:
            mins = ((j.get("finished") or now) - j["started"]) / 60
            parts.append(f"## {j['id']} ({j['subagent']}) {j['status']} after {mins:.0f} min\n\n{j['text']}")
        running = self._running()
        if running:
            parts.append("Still running: " + "; ".join(
                f"{j['id']} ({j['subagent']}, {(now - j['started']) / 60:.0f} min so far)" for j in running))
        elif not done:
            parts.append("No background jobs are running and none has an uncollected reply.")
        return "\n\n".join(parts)

    def _check_start(self, subagent_type: str, description: str) -> str | None:
        if not subagent_type or not description:
            return "Error: subagent_type and description are both required."
        if self._task_tool is None:
            return "Error: the `task` tool is not available to this agent, so nothing can run in the background."
        if self.hub is not None and subagent_type == NODE_AGENT:
            nid = node_of(description)
            if not nid:
                return ("Error: name the node on the first line of the description (for example `n3`), so the "
                        "node agent's GPU lease and questions are the node's.")
            if not self.hub.is_open(nid):
                return f"Error: {nid} is not an open node; open it with tree_new first."
            if (job := self.hub.job_of(nid)) and self.jobs.get(job, {}).get("status") == "running":
                return f"Error: {job} is already running node-agent on {nid}; one node has one agent at a time."
        return None

    def _bind(self, rec: dict[str, Any], subagent_type: str, description: str) -> None:
        """Tie a node-agent job to its node, and run it with that node in its context."""
        if self.hub is None or subagent_type != NODE_AGENT:
            return
        rec["node"] = node_of(description)
        self.hub.attach_job(rec["node"], rec["id"])
        self._write(rec)

    def _started_text(self, rec: dict[str, Any]) -> str:
        return (f"Started {rec['id']} ({rec['subagent']}). It runs in the background; {len(self._running())} job(s) "
                f"running now. Call {COLLECT} to get the reply when you need it.")

    @staticmethod
    def _job_runtime(runtime: ToolRuntime, jid: str) -> ToolRuntime:
        # Its own call id: the sub-agent's events and stored namespace then belong to the job, not to task_start.
        return dataclasses.replace(runtime, tool_call_id=f"{runtime.tool_call_id}-{jid}")

    # ------------------------------------------------------------------ tools

    async def _start_async(self, subagent_type: str, description: str, runtime: ToolRuntime) -> str:
        if (err := self._check_start(subagent_type, description)) is not None:
            return err
        rec = self._new_job(subagent_type, description, runtime.tool_call_id)
        self._bind(rec, subagent_type, description)
        run = self._task_tool.coroutine
        rt = self._job_runtime(runtime, rec["id"])
        writer = getattr(runtime, "stream_writer", None)

        async def job() -> None:
            if rec.get("node"):
                CURRENT_NODE.set(rec["node"])  # this task's own context: the node agent acts for this node
            try:
                self._finish(rec, _text_of(await run(description=description, subagent_type=subagent_type, runtime=rt)),
                             "done", writer)
            except asyncio.CancelledError:
                self._finish(rec, "The job was cancelled before it returned.", "lost", writer)
                raise
            except Exception as e:  # a failed sub-agent is a result the coordinator has to see
                self._finish(rec, f"The sub-agent failed: {type(e).__name__}: {e}", "failed", writer)

        self._handles[rec["id"]] = asyncio.get_running_loop().create_task(job(), context=contextvars.copy_context())
        return self._started_text(rec)

    def _start_sync(self, subagent_type: str, description: str, runtime: ToolRuntime) -> str:
        if (err := self._check_start(subagent_type, description)) is not None:
            return err
        rec = self._new_job(subagent_type, description, runtime.tool_call_id)
        self._bind(rec, subagent_type, description)
        run = self._task_tool.func
        rt = self._job_runtime(runtime, rec["id"])
        writer = getattr(runtime, "stream_writer", None)

        def job() -> None:
            if rec.get("node"):
                CURRENT_NODE.set(rec["node"])
            try:
                self._finish(rec, _text_of(run(description=description, subagent_type=subagent_type, runtime=rt)), "done", writer)
            except Exception as e:
                self._finish(rec, f"The sub-agent failed: {type(e).__name__}: {e}", "failed", writer)

        t = threading.Thread(target=contextvars.copy_context().run, args=(job,), daemon=True)
        self._handles[rec["id"]] = t
        t.start()
        return self._started_text(rec)

    def _hub_calls(self) -> bool:
        """The hub has something for the main agent: a question, a GPU request, a free GPU."""
        return self.hub is not None and self.hub.peek(MAIN)

    async def _wait_async(self, seconds: float) -> list[dict[str, Any]]:
        end = time.monotonic() + seconds
        while True:
            done = self._take_finished()
            if done or self._hub_calls() or time.monotonic() >= end or not self._running():
                return done
            await asyncio.sleep(POLL_SECONDS)

    def _wait_sync(self, seconds: float) -> list[dict[str, Any]]:
        end = time.monotonic() + seconds
        while True:
            done = self._take_finished()
            if done or self._hub_calls() or time.monotonic() >= end or not self._running():
                return done
            time.sleep(POLL_SECONDS)

    def _collected(self, done: list[dict[str, Any]]) -> str:
        text = self._report(done)
        if self.hub is not None:
            notes = self.hub.take_notes(MAIN)
            if notes:
                text += "\n\nNotes from the run:\n" + "\n".join(f"- {n}" for n in notes)
        return text

    async def _collect_async(self, wait_minutes: float = 0.0, runtime: ToolRuntime | None = None) -> str:
        return self._collected(await self._wait_async(_wait_of(wait_minutes)))

    def _collect_sync(self, wait_minutes: float = 0.0, runtime: ToolRuntime | None = None) -> str:
        return self._collected(self._wait_sync(_wait_of(wait_minutes)))

    # ------------------------------------------------------------------ model call: find `task`, say what finished

    def _prepare(self, request):
        for t in request.tools or []:
            if getattr(t, "name", None) == "task":
                self._task_tool = t
        done, running = self._unreported(), self._running()
        if not done or self.hub is not None:  # with a hub, what finished arrives as a note (coord_notes)
            return request
        from deepagents.middleware._utils import append_to_system_message

        note = ("## Background jobs\n\nFinished and not collected yet: "
                + ", ".join(f"{j['id']} ({j['subagent']}, {j['status']})" for j in done)
                + f". Call `{COLLECT}` to read the replies."
                + (f" Still running: {', '.join(j['id'] for j in running)}." if running else ""))
        return request.override(system_message=append_to_system_message(request.system_message, note))

    def wrap_model_call(self, request, handler):
        return handler(self._prepare(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._prepare(request))

    # ------------------------------------------------------------------ end of run: do not drop running work

    def _ending(self, state) -> bool:
        from .proactive_mode import _read_proactive_mode

        if _read_proactive_mode():  # a shadow turn pushes a note; it must not wait for or adopt a job
            return False
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        return isinstance(last, AIMessage) and not last.tool_calls and bool(self._running() or self._unreported())

    def _back_to_model(self, done: list[dict[str, Any]]) -> dict[str, Any]:
        text = ("You were about to finish while background work was still open. "
                + ("These jobs have finished:\n\n" if done else "No job finished in the last wait.\n\n")
                + self._collected(done)
                + "\n\nUse what came back (it may change the record), and finish only when nothing is left running.")
        return {"messages": [HumanMessage(content=text)], "jump_to": "model"}

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state, runtime):
        if not self._ending(state):
            return None
        return self._back_to_model(self._wait_sync(MAX_WAIT_MINUTES * 60))

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(self, state, runtime):
        if not self._ending(state):
            return None
        return self._back_to_model(await self._wait_async(MAX_WAIT_MINUTES * 60))
