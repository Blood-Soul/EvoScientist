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

Two more steps need no call when their switches are on:

- `auto_analysis`: when `tree_finish` closes a scored node, a `node-analyst` job starts for it; when the job finishes,
  its file is recorded with `tree_analysis` and the hub note carries the reply and the recorded id. A node already
  analysed, already running an analysis, or without a score (abandoned, failed) is skipped, and nothing starts in the
  last few minutes of the run.
- `auto_plan_register`: a `write_file` to `reports/drafts/plan-*.md` records that skeleton with `tree_plan` once per
  file, so the coordinator goes straight to the decision. A revision after the review is still recorded by the
  coordinator with `tree_plan(path=...)`, which updates the same plan (plans keep their source path).
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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, Field

from ..research_tree.coord import CURRENT_NODE, MAIN

ANALYST = "node-analyst"
AUTO_ANALYSIS_MIN_REMAINING = 300.0  # s: an analysis started with less time than this cannot finish
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

    def __init__(self, root: str | Path, *, now: Callable[[], float] = time.time, hub=None,
                 auto_analysis: bool = False, auto_node_agent: bool = False,
                 auto_plan_register: bool = False,
                 analysis_min_remaining: float = AUTO_ANALYSIS_MIN_REMAINING) -> None:
        super().__init__()
        self.root = Path(root)
        self.dir = self.root / ".research" / "background"
        self.now = now
        self.hub = hub
        # Start node-analyst on every scored node that has no analysis yet (see the module docstring).
        self.auto_analysis = auto_analysis and hub is not None
        # Start node-agent on a node the coordinator opens: the brief is written by tree_new, and a node without
        # an agent runs nothing.
        self.auto_node_agent = auto_node_agent and hub is not None
        self.analysis_min_remaining = analysis_min_remaining
        self._state_mtime: tuple[float, int] | None = None
        self._state_cache: dict[str, Any] | None = None
        self._tree_tools: dict[str, Any] = {}
        self.auto_plan_register = auto_plan_register and hub is not None
        self._auto_plans: dict[str, str] = self._load_auto_plans()
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
            if rec["subagent"] == ANALYST and (nid := node_of(rec.get("description") or "")):
                reply = " ".join((text or "").split())[:240]
                if status == "done":
                    recorded = self._record_analysis(nid)
                    self.hub._emit(MAIN, f"Analysis of {nid} {status} ({rec['id']}). {reply} {recorded}"
                                   if recorded else
                                   f"Analysis of {nid} {status} ({rec['id']}). {reply} Record it with "
                                   f"tree_analysis(node='{nid}', path='reports/drafts/analysis-{nid}.md').")
                else:
                    # A failed analyst may have left a partial file; recording it would make the node look
                    # analysed and hide the failure. Leave the node without an analysis (it can be retried).
                    self.hub._emit(MAIN, f"Analysis of {nid} failed ({rec['id']}). {reply} "
                                   f"The node still has no analysis; start node-analyst on it again with "
                                   f"task_start if it is still worth one.")
            else:
                summary = " ".join((text or "").split())[:300]
                self.hub._emit(MAIN, f"Background job {rec['id']} ({rec['subagent']}"
                               + (f", {rec['node']}" if rec.get("node") else "")
                               + f") {status}. {summary}"
                               + (f" (full reply: {COLLECT})" if len(text or "") > 300 else ""))
        self._announce(rec, writer)

    def _record_analysis(self, nid: str) -> str:
        """Record the analysis file the analyst wrote; returns the note fragment, or "" to leave it to the coordinator."""
        if not self.auto_analysis:
            return ""
        rel = f"reports/drafts/analysis-{nid}.md"
        tool = self._tree_tools.get("tree_analysis")
        if tool is None or not (self.root / rel).is_file():
            return ""
        try:
            out = tool.func(node=nid, path=rel)
        except Exception:  # recording is a convenience; the coordinator can still do it
            return ""
        return (out.strip() if out.startswith("Analysis of") else f"Recording it failed: {out.strip()}")

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

    # ------------------------------------------------------------------ automatic analysis

    def _tree_state(self) -> dict[str, Any] | None:
        """The tree's state, re-read only when the file changed (one stat per tool call, not one parse)."""
        path = self.root / ".research" / "state.json"
        try:
            stat = path.stat()
        except OSError:
            return None
        stamp = (stat.st_mtime, stat.st_size)
        if stamp != self._state_mtime:
            try:
                self._state_cache = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                self._state_cache = None
            self._state_mtime = stamp
        return self._state_cache

    def _analysis_due(self) -> list[str]:
        """Scored nodes with no recorded analysis and no analysis job running, soonest-first by id."""
        state = self._tree_state()
        if not isinstance(state, dict):
            return []
        deadline = state.get("deadline")
        if isinstance(deadline, (int, float)) and deadline - self.now() < self.analysis_min_remaining:
            return []
        analysed = state.get("analyses") or {}
        # A finished analysis that the coordinator has not recorded yet is not started again: the note told it
        # where the file is. A failed or lost one may be retried.
        with self._lock:
            attempted = {
                node_of(job.get("description") or "")
                for job in self.jobs.values()
                if job.get("subagent") == ANALYST and job.get("status") in ("running", "done")
            }
        return sorted(
            nid for nid, node in (state.get("nodes") or {}).items()
            if node.get("score") is not None
            and node.get("status") != "running"
            and nid not in analysed
            and nid not in attempted
        )

    def _analysis_job_running(self, nid: str) -> bool:
        with self._lock:
            return any(job.get("status") == "running" and job.get("subagent") == ANALYST
                       and node_of(job.get("description") or "") == nid for job in self.jobs.values())

    @staticmethod
    def _analysis_description(nid: str) -> str:
        return (
            f"{nid}\n\nAnalyse node {nid}: what it got right, what is wrong and why, measured on what it saved, "
            f"and the choices it leaves open. Write the analysis to reports/drafts/analysis-{nid}.md. Other nodes "
            "may be training: work on this node's saved outputs on the CPU, and record a check that needs the GPU "
            "as an open choice instead of running it."
        )

    def _start_due_analyses(self, runtime: Any) -> list[str]:
        """Start an analysis for every due node; returns the node ids started. Never raises."""
        if not self.auto_analysis or self._task_tool is None or runtime is None:
            return []
        started: list[str] = []
        for nid in self._analysis_due():
            try:
                text = self._start_sync(ANALYST, self._analysis_description(nid), runtime)
            except Exception:  # an analysis that cannot start must not fail the tool call that closed a node
                continue
            if text.startswith("Started "):
                started.append(nid)
                if self.hub is not None:
                    self.hub._emit(MAIN, f"Analysis of {nid} started on its own ({text.split('(')[0].strip()}). "
                                         f"Its file will be reports/drafts/analysis-{nid}.md and is recorded "
                                         "automatically when it finishes.")
            elif self.hub is not None:
                self.hub._emit(MAIN, f"Could not start node-analyst on {nid}: {text.strip()[:180]}")
        return started

    async def _start_due_analyses_async(self, runtime: Any) -> list[str]:
        if not self.auto_analysis or self._task_tool is None or runtime is None:
            return []
        started: list[str] = []
        for nid in self._analysis_due():
            try:
                text = await self._start_async(ANALYST, self._analysis_description(nid), runtime)
            except Exception:  # a failed start is retried on the next tree_new; the node keeps its note
                continue
            if text.startswith("Started "):
                started.append(nid)
                if self.hub is not None:
                    self.hub._emit(MAIN, f"Analysis of {nid} started on its own ({text.split('(')[0].strip()}). "
                                         f"Its file will be reports/drafts/analysis-{nid}.md and is recorded "
                                         "automatically when it finishes.")
            elif self.hub is not None:
                self.hub._emit(MAIN, f"Could not start node-analyst on {nid}: {text.strip()[:180]}")
        return started

    def _nodes_without_agent(self) -> list[str]:
        """Open nodes that have never had a node-agent job: a node whose agent ended is the coordinator's to
        restart or close, not something to start again behind its back (it may have finished its work)."""
        state = self._tree_state()
        if not isinstance(state, dict) or self.hub is None:
            return []
        with self._lock:
            seen = {
                node_of(job.get("description") or "")
                for job in self.jobs.values()
                if job.get("subagent") == NODE_AGENT
            }
        return sorted(
            nid for nid, node in (state.get("nodes") or {}).items()
            if node.get("status") == "running" and self.hub.is_open(nid) and not self.hub.job_of(nid)
            and nid not in seen
        )

    @staticmethod
    def _node_agent_description(nid: str) -> str:
        return (f"{nid}\n\nBuild node {nid}. Its BRIEF.md is at nodes/{nid}/BRIEF.md; read it in full first, then "
                "work inside that directory. Ask for the GPU with request_gpu when you need it.")

    def _start_due_node_agents(self, runtime: Any) -> list[str]:
        """Start node-agent for every open node without one; returns the node ids started. Never raises."""
        if not self.auto_node_agent or self._task_tool is None or runtime is None:
            return []
        started: list[str] = []
        for nid in self._nodes_without_agent():
            try:
                text = self._start_sync(NODE_AGENT, self._node_agent_description(nid), runtime)
            except Exception:  # failing to start must not fail the tool call that opened the node
                continue
            if text.startswith("Started "):
                started.append(nid)
                if self.hub is not None:
                    self.hub._emit(MAIN, f"node-agent started on {nid} on its own ({text.split('(')[0].strip()}).")
            elif self.hub is not None:
                self.hub._emit(MAIN, f"Could not start node-agent on {nid}: {text.strip()[:180]}. "
                                     "Start it with task_start, or close the node with tree_finish.")
        return started

    async def _start_due_node_agents_async(self, runtime: Any) -> list[str]:
        if not self.auto_node_agent or self._task_tool is None or runtime is None:
            return []
        started: list[str] = []
        for nid in self._nodes_without_agent():
            try:
                text = await self._start_async(NODE_AGENT, self._node_agent_description(nid), runtime)
            except Exception:  # a failed start is retried on the next tree_new; the node keeps its note
                continue
            if text.startswith("Started "):
                started.append(nid)
                if self.hub is not None:
                    self.hub._emit(MAIN, f"node-agent started on {nid} on its own ({text.split('(')[0].strip()}).")
            elif self.hub is not None:
                self.hub._emit(MAIN, f"Could not start node-agent on {nid}: {text.strip()[:180]}. "
                                     "Start it with task_start, or close the node with tree_finish.")
        return started

    # ------------------------------------------------------------------ automatic plan registration

    def _auto_plans_path(self) -> Path:
        return self.dir / "auto_plans.json"

    def _load_auto_plans(self) -> dict[str, str]:
        try:
            loaded = json.loads(self._auto_plans_path().read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return {str(k): str(v) for k, v in loaded.items()} if isinstance(loaded, dict) else {}

    def _save_auto_plans(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self._auto_plans_path().with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._auto_plans, indent=1))
            tmp.replace(self._auto_plans_path())
        except OSError:
            pass

    @staticmethod
    def _plan_draft(path: str) -> str:
        """The workspace-relative draft path if this write is a skeleton draft, else "".

        Absolute paths are accepted when they contain `reports/drafts/` (dangerous mode writes
        `/workspace/reports/drafts/plan-x.md`, and a model may use `./reports/...`); the part from
        `reports/drafts/` on is the workspace-relative path tree_plan needs.
        """
        rel = str(path or "").replace("\\", "/")
        marker = "reports/drafts/"
        cut = rel.find(marker)
        if cut < 0:
            return ""
        rel = rel[cut:]
        name = rel.rsplit("/", 1)[-1]
        return rel if name.startswith("plan-") and name.endswith(".md") else ""

    def _register_plan(self, rel: str) -> str:
        """Record a skeleton the coordinator just wrote, once per file; returns the note to the coordinator."""
        if not self.auto_plan_register or rel in self._auto_plans:
            return ""
        tool = self._tree_tools.get("tree_plan")
        if tool is None:
            return ""
        try:
            text = tool.func(path=rel)
        except Exception:  # a registration that fails must not fail the write that produced the skeleton
            text = "ERROR: tree_plan raised"
        pid = text.split(" ", 1)[0] if text.startswith("P") else ""
        if not (pid and pid[1:].isdigit()):
            # The prompt says a written skeleton is recorded on its own; when that did not happen
            # (no survey yet, an unaccepted parent, an error) say so instead of leaving the agent to guess.
            if self.hub is not None:
                self.hub._emit(MAIN, f"{rel} was not recorded automatically ({text.strip()[:180]}). "
                                     f"Record it with tree_plan(path='{rel}') when the tree is ready.")
            return ""
        self._auto_plans[rel] = pid
        self._save_auto_plans()
        if self.hub is not None:
            self.hub._emit(MAIN, f"{rel} recorded as {pid} on its own (a draft). Review it, then decide with "
                                 f"tree_plan(plan_id='{pid}', decision=...).")
        return pid

    def _closes_a_node(self, call: dict[str, Any]) -> bool:
        return (call.get("name") == "tree_finish"
                and not (call.get("args") or {}).get("abandon_reason"))

    @staticmethod
    def _tool_failed(result: Any) -> bool:
        """True when a tool call came back as an error (auto steps must not run on a failed action)."""
        if isinstance(result, ToolMessage):
            return getattr(result, "status", "") == "error"
        content = getattr(result, "content", result)
        if isinstance(content, str):
            return content.lstrip().startswith(("Error", "ERROR", "Refused"))
        return False

    def _refuse_plain_node_task(self, call: dict[str, Any]) -> ToolMessage | None:
        """A sync `task` call for node-agent would run without its node: its GPU lease and questions do not
        exist there. Refuse it with the route that works instead of letting a node burn its whole run."""
        if self.hub is None or call.get("name") != "task":
            return None
        args = call.get("args") or {}
        if str(args.get("subagent_type") or "") != NODE_AGENT:
            return None
        return ToolMessage(
            content=("Refused: node-agent must run in the background so its GPU lease, questions and notes "
                     "belong to its node. tree_new starts it automatically; if it is not running, use "
                     "task_start(subagent_type='node-agent', description='n1\\n...')."),
            tool_call_id=str(call.get("id") or ""), name=str(call.get("name") or "task"), status="error")

    def wrap_tool_call(self, request, handler):
        refused = self._refuse_plain_node_task(request.tool_call)
        if refused is not None:
            return refused
        result = handler(request)
        if self._tool_failed(result):
            return result
        runtime = request.runtime
        if self._closes_a_node(request.tool_call):
            self._start_due_analyses(runtime)
        if self._opens_a_node(request.tool_call):
            self._start_due_node_agents(runtime)
        self._maybe_register_plan(request.tool_call)
        return result

    async def awrap_tool_call(self, request, handler):
        refused = self._refuse_plain_node_task(request.tool_call)
        if refused is not None:
            return refused
        result = await handler(request)
        if self._tool_failed(result):
            return result
        runtime = request.runtime
        if self._closes_a_node(request.tool_call):
            await self._start_due_analyses_async(runtime)
        if self._opens_a_node(request.tool_call):
            await self._start_due_node_agents_async(runtime)
        self._maybe_register_plan(request.tool_call)
        return result

    def _maybe_register_plan(self, call: dict[str, Any]) -> None:
        if call.get("name") != "write_file" or not self.auto_plan_register:
            return
        args = call.get("args") or {}
        rel = self._plan_draft(str(args.get("file_path") or args.get("path") or ""))
        if rel:
            self._register_plan(rel)

    @staticmethod
    def _opens_a_node(call: dict[str, Any]) -> bool:
        return call.get("name") == "tree_new"

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
            name = getattr(t, "name", None)
            if name == "task":
                self._task_tool = t
            elif name in ("tree_analysis", "tree_plan"):
                self._tree_tools[name] = t
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
