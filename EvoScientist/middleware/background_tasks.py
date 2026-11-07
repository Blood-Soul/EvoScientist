"""Let the coordinator hand work to any sub-agent without waiting for it.

`task` blocks the caller until the sub-agent returns. In a run where several pieces of work are
independent (a plan for one direction while a node trains, plans for two directions at once), that
makes the slowest piece the clock for all of them. This middleware gives the main agent two tools:

- `task_start(subagent_type, description)` runs the real `task` tool in the background and returns a
  job id at once. It works for every sub-agent type, so nothing has to be flagged `async` per agent.
- `task_collect(wait_minutes)` returns the jobs that have finished since the last call, with their
  full results, and says what is still running. It waits at most `wait_minutes` for a first result.

Results are also written to `.research/background/<job>.json` so that they survive a restart of the
agent process; a job that was running when the process died is reported as lost, not as running.
The middleware sets no limit on how long a job may run: what bounds the work is the run's own clock.
State updates a sub-agent would return through `task` (private channels) are dropped; the files it
writes and the text it replies with are what the coordinator uses.

Main agent only, and only when `research_tree_background_tasks` is on.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelRequest, ToolCallRequest

START = "task_start"
COLLECT = "task_collect"
POLL_SECONDS = 0.2  # how soon a finished job is noticed while waiting
MAX_WAIT_MINUTES = 15.0  # one call never blocks longer; the model calls again to keep waiting


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


def _stub(name: str, description: str, schema: type[BaseModel]) -> StructuredTool:
    def _never(**_: Any) -> str:  # the middleware answers these calls before the tool node runs them
        return f"{name} is handled by the background-task middleware and is not available here."

    return StructuredTool.from_function(
        func=_never, name=name, description=description, args_schema=schema, infer_schema=False
    )


def _text_of(result: Any) -> str:
    """The reply text of what the `task` tool returned (a ToolMessage or a Command carrying one)."""
    msg = result
    if isinstance(result, Command):
        msgs = (result.update or {}).get("messages") if isinstance(result.update, dict) else None
        msg = msgs[-1] if msgs else None
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


class BackgroundTaskMiddleware(AgentMiddleware):
    """Main agent only: start sub-agents in the background and collect what they return."""

    name = "background_tasks"

    def __init__(self, root: str | Path, *, now: Callable[[], float] = time.time) -> None:
        super().__init__()
        self.dir = Path(root) / ".research" / "background"
        self.now = now
        self.jobs: dict[str, dict[str, Any]] = {}
        self._task_tool: Any = None
        self._lock = threading.Lock()
        self._n = self._recover()
        self.tools = [
            _stub(
                START,
                "Run a sub-agent in the background and get a job id back at once, instead of waiting as `task` does. "
                "Use it when the work does not have to be finished before your next decision: a plan for another "
                "direction while a node runs, plans for several directions together. Collect the replies with "
                f"`{COLLECT}`. Works for every sub-agent that `task` can reach.",
                StartArgs,
            ),
            _stub(
                COLLECT,
                "Return the background jobs that have finished since your last call, with their full replies, and say "
                "what is still running. Give `wait_minutes` to wait for a first result when none has finished yet. "
                "A job that is still running is not lost: call again later.",
                CollectArgs,
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
                    rec["status"] = "lost"
                    rec["text"] = "The agent process stopped while this job was running; its reply was not kept."
                    self._write(rec)
                rec["reported"] = rec.get("reported", rec.get("status") != "done")
                self.jobs[rec["id"]] = rec
        return n

    def _write(self, rec: dict[str, Any]) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            keep = {k: v for k, v in rec.items() if k not in ("future", "waiter")}
            tmp = self.dir / f"{rec['id']}.json.tmp"
            tmp.write_text(json.dumps(keep, ensure_ascii=False))
            tmp.replace(self.dir / f"{rec['id']}.json")
        except OSError:
            pass  # the in-memory record is still right; only the restart copy is missing

    # ------------------------------------------------------------------ model call: find the real task tool

    def _remember_task_tool(self, request: ModelRequest) -> None:
        for t in request.tools or []:
            if getattr(t, "name", None) == "task":
                self._task_tool = t

    def wrap_model_call(self, request, handler):
        self._remember_task_tool(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._remember_task_tool(request)
        return await handler(request)

    # ------------------------------------------------------------------ tool calls

    def _begin(self, request: ToolCallRequest) -> tuple[ToolMessage, ToolCallRequest | None, dict | None]:
        call = request.tool_call
        args = call.get("args") or {}
        sub, desc = args.get("subagent_type"), args.get("description")
        if not sub or not desc:
            return self._reply(call, "subagent_type and description are both required.", error=True), None, None
        if self._task_tool is None:
            return self._reply(call, "The `task` tool is not available to this agent.", error=True), None, None
        with self._lock:
            self._n += 1
            jid = f"bg-{self._n}"
            rec = {
                "id": jid, "n": self._n, "subagent": sub, "description": str(desc)[:300],
                "started": self.now(), "status": "running", "reported": False, "text": "",
            }
            self.jobs[jid] = rec
        self._write(rec)
        inner = request.override(
            tool_call={"name": "task", "args": {"subagent_type": sub, "description": desc}, "id": call["id"], "type": "tool_call"},
            tool=self._task_tool,
        )
        running = sum(1 for j in self.jobs.values() if j["status"] == "running")
        msg = (f"Started {jid} ({sub}). It runs in the background; {running} job(s) running now. "
               f"Call {COLLECT} to get the reply when you need it.")
        return self._reply(call, msg), inner, rec

    def _finish(self, rec: dict[str, Any], text: str, status: str) -> None:
        with self._lock:
            rec.update(status=status, text=text, finished=self.now(), reported=False)
        self._write(rec)

    async def _run_async(self, handler, inner, rec) -> None:
        try:
            self._finish(rec, _text_of(await handler(inner)), "done")
        except asyncio.CancelledError:
            self._finish(rec, "The job was cancelled before it returned.", "lost")
            raise
        except Exception as e:  # a failed sub-agent is a result the coordinator has to see
            self._finish(rec, f"The sub-agent failed: {type(e).__name__}: {e}", "failed")

    def _run_sync(self, handler, inner, rec) -> None:
        try:
            self._finish(rec, _text_of(handler(inner)), "done")
        except Exception as e:
            self._finish(rec, f"The sub-agent failed: {type(e).__name__}: {e}", "failed")

    def _reply(self, call: dict, text: str, *, error: bool = False) -> ToolMessage:
        return ToolMessage(content=text, tool_call_id=call["id"], name=call.get("name", START),
                           status="error" if error else "success")

    def _take_finished(self) -> list[dict[str, Any]]:
        with self._lock:
            done = [j for j in self.jobs.values() if j["status"] != "running" and not j["reported"]]
            for j in done:
                j["reported"] = True
        for j in done:
            self._write(j)
        return done

    def _collect_text(self, done: list[dict[str, Any]]) -> str:
        now = self.now()
        parts = []
        for j in done:
            mins = ((j.get("finished") or now) - j["started"]) / 60
            parts.append(f"## {j['id']} ({j['subagent']}) {j['status']} after {mins:.0f} min\n\n{j['text']}")
        running = [j for j in self.jobs.values() if j["status"] == "running"]
        if running:
            parts.append("Still running: " + "; ".join(
                f"{j['id']} ({j['subagent']}, {(now - j['started']) / 60:.0f} min so far)" for j in running))
        elif not done:
            parts.append("No background jobs are running and none has an unreported reply.")
        return "\n\n".join(parts)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        name = request.tool_call.get("name")
        if name == START:
            reply, inner, rec = self._begin(request)
            if inner is not None:
                ctx = contextvars.copy_context()
                threading.Thread(target=ctx.run, args=(self._run_sync, handler, inner, rec), daemon=True).start()
            return reply
        if name == COLLECT:
            wait = min(max(float((request.tool_call.get("args") or {}).get("wait_minutes") or 0), 0.0), MAX_WAIT_MINUTES)
            end = time.monotonic() + wait * 60
            while True:
                done = self._take_finished()
                if done or time.monotonic() >= end or not any(j["status"] == "running" for j in self.jobs.values()):
                    return self._reply(request.tool_call, self._collect_text(done))
                time.sleep(POLL_SECONDS)
        return handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]
    ) -> ToolMessage | Command[Any]:
        name = request.tool_call.get("name")
        if name == START:
            reply, inner, rec = self._begin(request)
            if inner is not None:
                rec["waiter"] = asyncio.get_running_loop().create_task(self._run_async(handler, inner, rec))
            return reply
        if name == COLLECT:
            wait = min(max(float((request.tool_call.get("args") or {}).get("wait_minutes") or 0), 0.0), MAX_WAIT_MINUTES)
            end = time.monotonic() + wait * 60
            while True:
                done = self._take_finished()
                if done or time.monotonic() >= end or not any(j["status"] == "running" for j in self.jobs.values()):
                    return self._reply(request.tool_call, self._collect_text(done))
                await asyncio.sleep(POLL_SECONDS)
        return await handler(request)
