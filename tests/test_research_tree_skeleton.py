"""Skeleton runs: the GPU lease, questions, notes at the end of the context, GPU isolation, the tree and the wiring."""
from __future__ import annotations

import asyncio
import contextvars
import subprocess
import threading
import time
from pathlib import Path

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage

from EvoScientist.middleware.background_tasks import BackgroundTaskMiddleware, node_of
from EvoScientist.middleware.coord_notes import HEADER, CoordNotesMiddleware
from EvoScientist.research_tree import coord
from EvoScientist.research_tree.coord import CURRENT_NODE, MAIN, Hub
from EvoScientist.research_tree.coord_tools import create_main_tools, create_node_tools
from EvoScientist.research_tree.tools import create_research_tree_tools
from EvoScientist.research_tree.tree import ResearchTree, TreeError


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, minutes: float) -> None:
        self.t += minutes * 60


@pytest.fixture(autouse=True)
def _no_active_hub():
    coord.deactivate()
    yield
    coord.deactivate()


def _hub(tmp_path, **kw) -> tuple[Hub, Clock, list]:
    clock = Clock()
    killed: list = []
    hub = Hub(tmp_path, now=clock, kill=lambda p: killed.append(p) or [123], **kw)
    for n in ("n1", "n2", "n3"):
        hub.open_node(n)
    return hub, clock, killed


def _in_node(nid: str, fn, *args):
    ctx = contextvars.copy_context()
    ctx.run(CURRENT_NODE.set, nid)
    return ctx.run(fn, *args)


# ---------------------------------------------------------------------- the lease

class TestLease:
    def test_a_free_gpu_with_nobody_waiting_goes_at_once_and_main_is_told(self, tmp_path):
        hub, _, _ = _hub(tmp_path)
        assert hub.request("n1", "full training") == "granted"
        assert hub.holder == "n1"
        main = hub.take_notes(MAIN)
        assert any("n1 took the free GPU: full training" in n and "gpu_revoke" in n for n in main)
        # the others learn where the GPU is; the holder is not told about itself
        assert any("held by n1" in n for n in hub.take_notes("n2"))
        assert not any("held by n1" in n for n in hub.take_notes("n1"))

    def test_a_busy_gpu_queues_the_request_and_main_decides(self, tmp_path):
        hub, _, killed = _hub(tmp_path)
        hub.request("n1", "training")
        hub.take_notes(MAIN)
        assert hub.request("n2", "check tokenizer truncation on the full data") == "queued"
        assert hub.request("n2", "again") == "queued" and len(hub.queue) == 1  # asked once
        assert any("n2 asks for the GPU: check tokenizer truncation" in n for n in hub.take_notes(MAIN))
        with pytest.raises(ValueError, match="n1 holds the GPU"):
            hub.grant("n2")
        (tmp_path / "nodes" / "n1").mkdir(parents=True)
        assert hub.release("n1") == [123] and killed  # its processes are stopped
        notes = hub.take_notes(MAIN)
        assert any("The GPU is free: n1 released it" in n and "n2 (check tokenizer" in n for n in notes)
        assert hub.holder == ""  # main decides; a free GPU does not jump to the queue at once
        assert "n2 holds the GPU now" in hub.grant("n2")
        assert hub.queue == []

    def test_without_a_decision_the_earliest_request_gets_it(self, tmp_path):
        hub, clock, _ = _hub(tmp_path, auto_grant_seconds=600)
        hub.request("n1", "training")
        hub.request("n2", "a")
        clock.advance(1)
        hub.request("n3", "b")
        hub.release("n1")
        clock.advance(9)
        assert hub.lease("n2") == "queued"
        clock.advance(1.1)
        assert hub.lease("n2") == "granted" and hub.queue[0]["node"] == "n3"
        assert any("went to the earliest request, n2" in n for n in hub.take_notes(MAIN))

    def test_revoke_stops_the_node_tells_it_and_frees_the_gpu(self, tmp_path):
        hub, _, _ = _hub(tmp_path)
        (tmp_path / "nodes" / "n1").mkdir(parents=True)
        hub.request("n1", "training")
        hub.take_notes("n1")
        assert hub.revoke("n1", "n2 is close to the deadline") == [123]
        assert any("took the GPU back: n2 is close to the deadline" in n for n in hub.take_notes("n1"))
        assert hub.holder == ""
        with pytest.raises(ValueError, match="does not hold"):
            hub.revoke("n1", "again")

    def test_closing_a_node_frees_its_lease_and_drops_its_request(self, tmp_path):
        hub, _, _ = _hub(tmp_path)
        hub.request("n1", "t")
        hub.request("n2", "t")
        hub.close_node("n2")
        assert hub.queue == [] and hub.lease("n2") == "closed"
        hub.close_node("n1")
        assert hub.holder == "" and hub.lease("n1") == "closed"

    def test_a_job_that_ends_drops_its_request_but_not_its_lease(self, tmp_path):
        hub, _, _ = _hub(tmp_path)
        hub.attach_job("n1", "bg-1")
        hub.attach_job("n2", "bg-2")
        hub.request("n1", "t")
        hub.request("n2", "t")
        hub.job_ended("n2", "bg-2")
        assert hub.queue == []
        hub.job_ended("n1", "bg-1")
        assert hub.holder == "n1"  # only the main agent ends a lease
        assert any("finished while holding the GPU" in n for n in hub.take_notes(MAIN))
        hub.job_ended("n1", "bg-old")  # a stale job id changes nothing
        assert hub.job_of("n1") == ""


# ---------------------------------------------------------------------- notes

class TestNotes:
    def test_each_note_is_delivered_once_to_its_audience(self, tmp_path):
        hub, _, _ = _hub(tmp_path)
        hub._emit("n1", "for n1")
        hub._emit(coord.NODES, "for every node")
        hub._emit(MAIN, "for main")
        assert [n.split("] ", 1)[1] for n in hub.take_notes("n1")] == ["for n1", "for every node"]
        assert hub.take_notes("n1") == []
        assert [n.split("] ", 1)[1] for n in hub.take_notes("n2")] == ["for every node"]
        assert [n.split("] ", 1)[1] for n in hub.take_notes(MAIN)] == ["for main"]
        assert (tmp_path / ".research" / "coord.jsonl").read_text().count("\n") == 3

    def test_a_node_opened_later_does_not_get_old_notes(self, tmp_path):
        hub, _, _ = _hub(tmp_path)
        hub._emit(coord.NODES, "old")
        hub.open_node("n4")
        assert hub.take_notes("n4") == []

    def test_the_middleware_appends_a_message_and_never_touches_the_system_prompt(self, tmp_path):
        hub, _, _ = _hub(tmp_path)
        main = CoordNotesMiddleware(hub, main=True)
        node = CoordNotesMiddleware(hub, main=False)
        assert main.before_model({"messages": []}, None) is None  # nothing new: nothing added
        hub.request("n1", "training")
        out = main.before_model({"messages": []}, None)
        msg = out["messages"][0]
        assert isinstance(msg, HumanMessage) and msg.content.startswith(HEADER) and "n1 took the free GPU" in msg.content
        assert set(out) == {"messages"}  # an update to the message list only
        assert main.before_model({"messages": []}, None) is None  # shown once
        # a node agent gets its own notes, through the node of its context
        assert node.before_model({"messages": []}, None) is None  # outside a node: nothing
        got = _in_node("n2", node.before_model, {"messages": []}, None)
        assert "held by n1" in got["messages"][0].content
        assert asyncio.run(main.abefore_model({"messages": []}, None)) is None


# ---------------------------------------------------------------------- the tools

class TestTools:
    def test_request_gpu_waits_in_the_session_until_granted(self, tmp_path):
        hub, _, _ = _hub(tmp_path)
        hub.request("n1", "training")
        tools = {t.name: t for t in create_node_tools(hub)}
        out: dict = {}

        def ask():
            out["r"] = tools["request_gpu"].invoke({"reason": "full training"})

        ctx = contextvars.copy_context()
        ctx.run(CURRENT_NODE.set, "n2")
        t = threading.Thread(target=ctx.run, args=(ask,))
        t.start()
        time.sleep(0.8)
        assert t.is_alive() and hub.queue[0]["node"] == "n2"
        hub.release("n1")
        hub.grant("n2")
        t.join(5)
        assert "n2 holds the GPU now" in out["r"]

    def test_request_gpu_gives_up_after_its_wait_and_stays_queued(self, tmp_path):
        hub, _, _ = _hub(tmp_path, wait_seconds=0.6)
        hub.request("n1", "t")
        tools = {t.name: t for t in create_node_tools(hub)}
        r = _in_node("n2", tools["request_gpu"].invoke, {"reason": "x"})
        assert "Still waiting" in r and hub.lease("n2") == "queued"

    def test_node_tools_outside_a_node_and_without_a_reason(self, tmp_path):
        hub, _, _ = _hub(tmp_path)
        tools = {t.name: t for t in create_node_tools(hub)}
        assert "not running a node" in tools["request_gpu"].invoke({"reason": "x"})
        assert "not running a node" in tools["ask_main"].invoke({"question": "q"})
        assert "give the reason" in _in_node("n1", tools["request_gpu"].invoke, {"reason": " "})
        assert "does not hold" in _in_node("n1", tools["release_gpu"].invoke, {})

    def test_ask_main_gets_the_answer(self, tmp_path):
        hub, _, _ = _hub(tmp_path)
        node = {t.name: t for t in create_node_tools(hub)}
        main = {t.name: t for t in create_main_tools(hub)}
        out: dict = {}
        ctx = contextvars.copy_context()
        ctx.run(CURRENT_NODE.set, "n1")
        t = threading.Thread(target=ctx.run, args=(lambda: out.update(r=node["ask_main"].invoke(
            {"question": "the labels are 40% empty; keep the abstain head?"})),))
        t.start()
        time.sleep(0.8)
        note = next(n for n in hub.take_notes(MAIN) if "Question Q1 from n1" in n)
        assert "answer_node('Q1'" in note
        assert "Q1 answered" in main["answer_node"].invoke({"question_id": "Q1", "answer": "keep it"})
        t.join(5)
        assert out["r"] == "The main agent's answer to Q1: keep it"
        assert "answered already" in main["answer_node"].invoke({"question_id": "Q1", "answer": "x"})

    def test_ask_main_times_out_to_the_skeletons_choice_and_a_late_answer_is_a_note(self, tmp_path):
        hub, _, _ = _hub(tmp_path, answer_seconds=0.6)
        node = {t.name: t for t in create_node_tools(hub)}
        main = {t.name: t for t in create_main_tools(hub)}
        r = _in_node("n1", node["ask_main"].invoke, {"question": "q"})
        assert "No answer to Q1" in r and "skeleton's choice" in r and "DESIGN.md" in r
        assert "had timed out" in main["answer_node"].invoke({"question_id": "Q1", "answer": "late"})
        assert any("late answer to your question Q1" in n for n in hub.take_notes("n1"))

    def test_main_tools_refuse_with_text(self, tmp_path):
        hub, _, _ = _hub(tmp_path)
        main = {t.name: t for t in create_main_tools(hub)}
        assert main["gpu_grant"].invoke({"node": "n9"}).startswith("ERROR: n9 is not an open node")
        assert main["gpu_revoke"].invoke({"node": "n1", "reason": ""}).startswith("ERROR: say why")
        assert main["gpu_revoke"].invoke({"node": "n1", "reason": "x"}).startswith("ERROR: n1 does not hold")
        assert main["answer_node"].invoke({"question_id": "Q7", "answer": "a"}).startswith("ERROR: no question Q7")
        assert "holds the GPU now" in main["gpu_grant"].invoke({"node": "n1"})
        assert any("gave you the GPU" in n for n in hub.take_notes("n1"))  # unasked: the node is told


# ---------------------------------------------------------------------- GPU isolation

class TestIsolation:
    def test_outside_a_skeleton_run_nothing_changes(self):
        assert coord.shell_env({"A": "1"}) == {"A": "1"}
        assert coord.refuse_command("CUDA_VISIBLE_DEVICES=0 python t.py") is None

    def test_only_the_holder_sees_the_gpu(self, tmp_path):
        hub = coord.hub_for(tmp_path)
        hub.open_node("n1")
        hub.open_node("n2")
        hub.request("n1", "t")
        assert coord.shell_env({"A": "1"}) == {"A": "1", "CUDA_VISIBLE_DEVICES": ""}  # the main agent
        assert _in_node("n2", coord.shell_env, {"A": "1"})["CUDA_VISIBLE_DEVICES"] == ""
        assert _in_node("n1", coord.shell_env, {"A": "1"}) == {"A": "1"}
        assert "CUDA_VISIBLE_DEVICES" in coord.shell_env(None)  # None means "inherit": filled from os.environ

    def test_setting_the_variable_without_the_lease_is_refused(self, tmp_path):
        hub = coord.hub_for(tmp_path)
        hub.open_node("n1")
        assert "request_gpu" in _in_node("n1", coord.refuse_command, "CUDA_VISIBLE_DEVICES=0 python t.py")
        assert _in_node("n1", coord.refuse_command, "python t.py") is None
        hub.request("n1", "t")
        assert _in_node("n1", coord.refuse_command, "CUDA_VISIBLE_DEVICES=0 python t.py") is None

    def test_the_sandbox_shell_applies_it(self, tmp_path):
        from EvoScientist.backends import CustomSandboxBackend

        hub = coord.hub_for(tmp_path)
        hub.open_node("n1")
        backend = CustomSandboxBackend(root_dir=str(tmp_path), virtual_mode=True)
        out = _in_node("n1", backend.execute, "echo [${CUDA_VISIBLE_DEVICES-unset}]")
        assert "[]" in out.output
        refused = _in_node("n1", backend.execute, "CUDA_VISIBLE_DEVICES=0 echo hi")
        assert refused.exit_code == 1 and "request_gpu" in refused.output
        hub.request("n1", "t")
        out = _in_node("n1", backend.execute, "echo [${CUDA_VISIBLE_DEVICES-unset}]")
        assert "[]" not in out.output


# ---------------------------------------------------------------------- background jobs

def _runtime(call_id: str = "c1") -> ToolRuntime:
    return ToolRuntime(state={}, context=None, config={}, stream_writer=lambda *a, **k: None,
                       tool_call_id=call_id, store=None)


class _Task:
    def __init__(self, gate: threading.Event | None = None) -> None:
        self.gate = gate
        self.seen: list[str] = []

    def func(self, description, subagent_type, runtime):
        self.seen.append(CURRENT_NODE.get())
        if self.gate is not None:
            self.gate.wait(5)
        return "built"

    async def coroutine(self, description, subagent_type, runtime):
        self.seen.append(CURRENT_NODE.get())
        return "built"


class TestBackground:
    def test_node_of(self):
        assert node_of("n3\nRead nodes/n3/BRIEF.md") == "n3"
        assert node_of("node: N12 - build it") == "n12"
        assert node_of("Build the node") == ""

    def test_a_node_agent_job_runs_in_its_node_and_one_at_a_time(self, tmp_path):
        hub, _, _ = _hub(tmp_path)
        gate = threading.Event()
        mw = BackgroundTaskMiddleware(tmp_path, hub=hub)
        mw._task_tool = _Task(gate)
        assert "name the node" in mw._start_sync("node-agent", "build it", _runtime())
        assert "not an open node" in mw._start_sync("node-agent", "n9\nbuild", _runtime())
        assert "Started bg-1" in mw._start_sync("node-agent", "n1\nread nodes/n1/BRIEF.md", _runtime())
        time.sleep(0.2)
        assert "already running node-agent on n1" in mw._start_sync("node-agent", "n1\nagain", _runtime("c2"))
        assert hub.job_of("n1") == "bg-1"
        gate.set()
        time.sleep(0.3)
        assert mw._task_tool.seen == ["n1"]
        assert hub.job_of("n1") == ""
        assert any("bg-1 (node-agent, n1) done" in n for n in hub.take_notes(MAIN))

    def test_an_async_job_runs_in_its_node(self, tmp_path):
        hub, _, _ = _hub(tmp_path)
        mw = BackgroundTaskMiddleware(tmp_path, hub=hub)
        task = _Task()
        mw._task_tool = task

        async def go():
            await mw._start_async("node-agent", "n2\nbuild", _runtime())
            await asyncio.sleep(0.2)

        asyncio.run(go())
        assert task.seen == ["n2"] and CURRENT_NODE.get() == ""

    def test_collect_returns_early_for_a_question(self, tmp_path):
        hub, _, _ = _hub(tmp_path)
        gate = threading.Event()
        mw = BackgroundTaskMiddleware(tmp_path, hub=hub)
        mw._task_tool = _Task(gate)
        mw._start_sync("node-agent", "n1\nbuild", _runtime())
        hub.take_notes(MAIN)
        threading.Timer(0.5, lambda: hub.ask("n1", "is the premise right?")).start()
        t0 = time.monotonic()
        out = mw._collect_sync(wait_minutes=1)
        assert time.monotonic() - t0 < 5
        assert "Question Q1 from n1" in out and "Still running: bg-1" in out
        gate.set()

    def test_with_a_hub_the_system_prompt_is_left_alone(self, tmp_path):
        from langchain.agents.middleware.types import ModelRequest

        hub, _, _ = _hub(tmp_path)
        mw = BackgroundTaskMiddleware(tmp_path, hub=hub)
        mw._task_tool = _Task()
        mw._start_sync("general-purpose", "x", _runtime())
        time.sleep(0.2)
        req = mw.wrap_model_call(ModelRequest(model=None, messages=[], tools=[]), lambda r: r)
        assert req.system_message is None


# ---------------------------------------------------------------------- the tree in a skeleton run

class FakeJudge:
    def score(self, pred, deliv=None):
        rows = Path(pred).read_text().split()
        per = [1.0 if r == "1" else 0.0 for r in rows]
        return {"score": sum(per) / len(per), "per_sample": per, "eval_sha": "abc"}


SKELETON = """# Plan: span head
## Direction and hypothesis
- extractive spans should beat generation because answers are copied [E-abc123]; fails if most answers are not in the passage
## What to build
- a span head on a pretrained encoder [E-abc123] (applies: same copy setting)
## Risks and refutation
- answers missing from the passage
## Open choices
- abstain head or not (default: abstain head)
"""


@pytest.fixture
def skel(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    clock = Clock()
    hub = Hub(root, now=clock, kill=lambda p: [])
    tree = ResearchTree(root, judge=FakeJudge(), now=clock, require_plans=True, hub=hub)
    tree.init(budget_minutes=600)
    tree.survey([{"name": "A", "mechanism": "m"}])
    return tree, hub, root, clock


def _accepted(tree, root) -> str:
    d = root / "reports" / "drafts"
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan-a.md").write_text(SKELETON)
    pid = tree.plan(text=SKELETON, cluster="A").split()[0]
    tree.plan(plan_id=pid, decision="accepted")
    return pid


class TestTree:
    def test_several_nodes_open_at_once_and_none_holds_the_gpu(self, skel):
        tree, hub, root, clock = skel
        out1 = tree.new_node(kind="explore", title="a", hypothesis="h", plan_id=_accepted(tree, root))
        out2 = tree.new_node(kind="explore", title="b", hypothesis="h", plan_id=_accepted(tree, root))
        assert "node-agent" in out1 and "first line" in out1 and "Created n2" in out2
        assert hub.is_open("n1") and hub.is_open("n2") and hub.holder == ""
        brief = (root / "nodes" / "n1" / "BRIEF.md").read_text()
        assert "### The accepted skeleton (P1): work out its details and build it" in brief
        assert "request_gpu(reason)" in brief and "DESIGN.md" in brief and "budget" not in brief
        with pytest.raises(TreeError, match="gpu_grant"):
            tree.start("n1")
        status = tree.status()
        assert "Running: n1 (explore) for 0.0 min, no agent running" in status and "GPU: free" in status

    def test_status_shows_the_lease_the_queue_and_what_is_due(self, skel):
        tree, hub, root, clock = skel
        tree.new_node(kind="explore", title="a", hypothesis="h", plan_id=_accepted(tree, root))
        tree.new_node(kind="explore", title="b", hypothesis="h", plan_id=_accepted(tree, root))
        hub.attach_job("n1", "bg-1")
        hub.attach_job("n2", "bg-2")
        hub.request("n1", "full training")
        hub.request("n2", "profiling run")
        hub.ask("n2", "drop the abstain head?")
        clock.advance(12)
        status = tree.status()
        assert "n1 (explore) for 12.0 min, holds the GPU (full training)" in status
        assert "GPU: held by n1 for 12 min (full training)" in status
        assert "waiting: n2 for 12 min (profiling run)" in status
        assert "question Q1 from n2" in status
        last = status.splitlines()[-1]
        assert "answer_node" in last and "write the next skeleton" in last
        hub.release("n1")
        assert "the GPU is free and n2 asked for it: gpu_grant" in tree.status().splitlines()[-1]

    def test_finish_frees_the_gpu_keeps_the_design_and_says_when_it_is_missing(self, skel):
        tree, hub, root, clock = skel
        tree.new_node(kind="explore", title="a", hypothesis="h", plan_id=_accepted(tree, root))
        tree.new_node(kind="explore", title="b", hypothesis="h", plan_id=_accepted(tree, root))
        hub.request("n1", "t")
        for nid in ("n1", "n2"):
            node = root / "nodes" / nid
            (node / "holdout_predictions.csv").write_text("1\n" * 300)
            (node / "submission.csv").write_text("d")
        (root / "nodes" / "n1" / "DESIGN.md").write_text("# Design\nlr 3e-5 from the length measurement\n")
        clock.advance(30)
        done = tree.finish("n1", change="c", observation="o")
        assert "closed as done" in done and "after 30.0 min." in done and "no DESIGN.md" not in done
        assert hub.holder == "" and not hub.is_open("n1")
        assert (root / ".research" / "artifacts" / "n1" / "DESIGN.md").read_text().startswith("# Design")
        assert "left no DESIGN.md" in tree.finish("n2", change="c", observation="o")

    def test_the_tools_carry_the_main_agents_gpu_and_answer_tools(self, tmp_path):
        hub = Hub(tmp_path)
        names = [t.name for t in create_research_tree_tools(tmp_path, hub=hub)]
        assert names[-3:] == ["gpu_grant", "gpu_revoke", "answer_node"]
        assert "gpu_grant" not in [t.name for t in create_research_tree_tools(tmp_path)]
        plan = next(t for t in create_research_tree_tools(tmp_path, hub=hub) if t.name == "tree_plan")
        assert "skeleton you wrote" in plan.description

    def test_a_plain_tree_is_unchanged_by_the_hub_code(self, tmp_path):
        root = tmp_path / "ws"
        root.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        tree = ResearchTree(root, judge=FakeJudge(), now=Clock(), require_plans=False)
        tree.init(budget_minutes=600)
        out = tree.new_node(kind="explore", title="a", hypothesis="h")
        assert "node-agent" not in out and "GPU:" not in tree.status()


# ---------------------------------------------------------------------- prompts, sub-agents and wiring

class TestWiring:
    def _prompt(self, **kw):
        from EvoScientist.prompts import get_system_prompt

        return get_system_prompt(research_tree=True, research_tree_reports=True, available_tools=["search_experience"], **kw)

    def test_the_skeleton_prompt_replaces_the_plan_writer_flow(self):
        on = self._prompt(research_tree_background=True, research_tree_skeleton=True)
        assert "## Plans (skeletons)" in on and "review_plan(path, task)" in on and "gpu_grant(node)" in on
        assert "plan-writer" not in on and "code-agent" not in on
        assert "budget of a node" not in on.lower()
        off = self._prompt(research_tree_background=True)
        assert "plan-writer" in off and "Plans (skeletons)" not in off
        # the switch needs background jobs: without them nothing changes
        assert "Plans (skeletons)" not in self._prompt(research_tree_skeleton=True)

    def test_node_agent_yaml_grants_the_lease_tools_and_the_experience_tools(self):
        from EvoScientist.utils import load_subagents

        subs = {s["name"]: s for s in load_subagents(Path(__file__).parents[1] / "EvoScientist" / "subagents")}
        names = set(subs["node-agent"]["_tool_names"])
        assert {"request_gpu", "release_gpu", "ask_main", "search_experience", "search_paper_text", "read_paper"} <= names
        text = " ".join(subs["node-agent"]["system_prompt"].split())
        assert "DESIGN.md" in text and "not its values" in text and "drawn from E-" in text

    def test_the_run_picks_its_sub_agents_and_hub_from_the_switch(self, tmp_path):
        from EvoScientist.EvoScientist import _select_subagents, _skeleton_hub
        from EvoScientist.config.settings import EvoScientistConfig

        class WS:
            root = tmp_path

        subs = [{"name": n} for n in ("plan-writer", "code-agent", "node-agent", "node-analyst")]
        off = EvoScientistConfig(research_tree_background_tasks=True)
        assert _skeleton_hub(off, WS) is None
        assert [s["name"] for s in _select_subagents(subs, off, WS)] == ["plan-writer", "code-agent", "node-analyst"]
        on = EvoScientistConfig(research_tree_background_tasks=True, research_tree_skeleton=True)
        assert _skeleton_hub(on, WS) is coord.active_hub()
        assert [s["name"] for s in _select_subagents(subs, on, WS)] == ["node-agent", "node-analyst"]
        half = EvoScientistConfig(research_tree_background_tasks=False, research_tree_skeleton=True)
        assert _skeleton_hub(half, WS) is None
