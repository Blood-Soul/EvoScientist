"""Plans and analyses around research-tree nodes: the rules, the files, the wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from EvoScientist.paths import Workspace
from EvoScientist.prompts import RESEARCH_TREE_REPORTS, get_system_prompt
from EvoScientist.research_tree.reports import report_path
from EvoScientist.research_tree.tools import create_research_tree_tools
from EvoScientist.research_tree.tree import ResearchTree, TreeError
from tests.test_research_tree import Clock, FakeJudge, do_node, git

PLAN = "# Plan: shared-norm reader\nTrain with answer-free windows in each batch.\n"


def make(tmp_path: Path, *, require_plans: bool):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "data").mkdir()
    (root / "data" / "train.txt").write_text("rows")
    clock = Clock()
    tree = ResearchTree(root, judge=FakeJudge(), now=clock, require_plans=require_plans)
    tree.init(budget_minutes=900)
    return tree, root, clock


@pytest.fixture
def strict(tmp_path):
    return make(tmp_path, require_plans=True)


def new_plan(tree, **kw) -> str:
    out = tree.plan(text=kw.pop("text", PLAN), **kw)
    pid = out.split()[0]
    tree.plan(plan_id=pid, decision="accepted")
    return pid


def first_node(tree, root, clock, pid):
    nid = do_node(
        tree,
        root,
        clock,
        minutes_used=30,
        n_right=200,
        kind="explore",
        title="baseline",
        hypothesis="a plain reader sets a floor",
        plan_id=pid,
    )
    tree.finish(nid, change="plain reader", observation="ok", next_step="x")
    return nid


class TestPlanLifecycle:
    def test_record_decide_and_status(self, strict) -> None:
        tree, root, _ = strict
        out = tree.plan(text=PLAN)
        assert out.startswith("P1 recorded as a draft")
        assert report_path(root, "plans", "P1").read_text() == PLAN
        assert "P1 draft" in tree.status()
        tree.plan(plan_id="P1", decision="accepted")
        assert "P1 accepted" in tree.status()

    def test_text_xor_decision(self, strict) -> None:
        tree, *_ = strict
        with pytest.raises(TreeError, match="either `text`"):
            tree.plan()
        with pytest.raises(TreeError, match="either `text`"):
            tree.plan(text=PLAN, decision="accepted")

    def test_reject_and_revise_need_a_reason(self, strict) -> None:
        tree, *_ = strict
        tree.plan(text=PLAN)
        with pytest.raises(TreeError, match="say why"):
            tree.plan(plan_id="P1", decision="rejected")
        with pytest.raises(TreeError, match="say why"):
            tree.plan(plan_id="P1", decision="revise")
        with pytest.raises(TreeError, match="decision must be one of"):
            tree.plan(plan_id="P1", decision="maybe", reason="r")

    def test_two_rewrites_then_a_decision(self, strict) -> None:
        tree, root, _ = strict
        tree.plan(text=PLAN)
        for round_no in (1, 2):
            sent = tree.plan(plan_id="P1", decision="revise", reason="too slow")
            assert "too slow" in sent
            tree.plan(plan_id="P1", text=PLAN + f"round {round_no}\n")
        # the version each rewrite replaced is kept
        hist = root / ".research" / "reports_history"
        assert (hist / "P1.v1.md").read_text() == PLAN
        assert (hist / "P1.v2.md").read_text() == PLAN + "round 1\n"
        with pytest.raises(TreeError, match="rewritten 2 times"):
            tree.plan(plan_id="P1", text="again")
        with pytest.raises(TreeError, match="rewritten 2 times"):
            tree.plan(plan_id="P1", decision="revise", reason="still")
        tree.plan(plan_id="P1", decision="accepted")
        with pytest.raises(TreeError, match="already accepted"):
            tree.plan(plan_id="P1", decision="rejected", reason="late")
        with pytest.raises(TreeError, match="only a draft"):
            tree.plan(plan_id="P1", text="after the decision")

    def test_cluster_and_parents_are_checked(self, strict) -> None:
        tree, *_ = strict
        with pytest.raises(TreeError, match="not a surveyed family"):
            tree.plan(text=PLAN, cluster="nope")
        with pytest.raises(TreeError, match="finished, scored node"):
            tree.plan(text=PLAN, parents=["n9"])


class TestNodesNeedPlans:
    def test_explore_needs_an_accepted_plan(self, strict) -> None:
        tree, *_ = strict
        args = dict(kind="explore", title="t", hypothesis="h")
        with pytest.raises(TreeError, match="need an accepted plan"):
            tree.new_node(**args)
        with pytest.raises(TreeError, match="no plan P1"):
            tree.new_node(**args, plan_id="P1")
        tree.plan(text=PLAN)
        with pytest.raises(TreeError, match="draft, not accepted"):
            tree.new_node(**args, plan_id="P1")

    def test_the_plan_text_reaches_the_brief_and_binds_one_node(self, strict) -> None:
        tree, root, clock = strict
        pid = new_plan(tree)
        out = tree.new_node(kind="explore", title="t", hypothesis="h", plan_id=pid)
        assert "### The accepted plan (P1): build this" in out
        assert "answer-free windows" in out
        assert "follow the accepted plan P1" in out
        assert (root / ".research" / "briefs" / "n1.md").read_text().count("P1") >= 1
        clock.advance(10)
        with pytest.raises(TreeError, match="already backs n1"):
            tree.new_node(
                kind="explore", title="u", hypothesis="h", plan_id=pid, gpu=False
            )

    def test_an_abandoned_node_frees_its_plan(self, strict) -> None:
        tree, root, clock = strict
        pid = new_plan(tree)
        nid = do_node(
            tree,
            root,
            clock,
            minutes_used=5,
            n_right=0,
            kind="explore",
            title="t",
            hypothesis="h",
            plan_id=pid,
        )
        tree.finish(nid, change="-", observation="-", abandon_reason="bad start")
        again = tree.new_node(
            kind="explore", title="t2", hypothesis="h", plan_id=pid, based_on=[nid]
        )
        assert "Created n2" in again

    def test_pilot_and_merge_take_no_plan(self, strict) -> None:
        tree, *_ = strict
        pid = new_plan(tree)
        with pytest.raises(TreeError, match="does not take a plan"):
            tree.new_node(kind="pilot", title="p", hypothesis="h", plan_id=pid)
        assert "Created n1" in tree.new_node(kind="pilot", title="p", hypothesis="h")

    def test_a_refine_needs_a_matching_plan_and_an_analysis(self, strict) -> None:
        tree, root, clock = strict
        n1 = first_node(tree, root, clock, new_plan(tree))
        refine = dict(
            kind="refine", title="r", hypothesis="h", parents=[n1], based_on=[n1]
        )
        p_root = new_plan(tree)
        with pytest.raises(TreeError, match="must match"):
            tree.new_node(**refine, plan_id=p_root)
        p2 = new_plan(tree, parents=[n1])
        with pytest.raises(TreeError, match="has no analysis yet"):
            tree.new_node(**refine, plan_id=p2)
        tree.analysis(
            node=n1,
            text="# Analysis of n1\n...",
            strengths=["clean pipeline"],
            flaws=["no-answer recall is low"],
            improvements=["train the null score on answer-free windows"],
        )
        out = tree.new_node(**refine, plan_id=p2)
        assert "Analysis A1 of n1:" in out
        assert "no-answer recall is low" in out

    def test_the_default_tree_is_unchanged(self, tmp_path) -> None:
        tree, root, clock = make(tmp_path, require_plans=False)
        n1 = do_node(
            tree,
            root,
            clock,
            minutes_used=30,
            n_right=200,
            kind="explore",
            title="t",
            hypothesis="h",
        )
        tree.finish(n1, change="c", observation="o")
        out = tree.new_node(
            kind="refine", title="r", hypothesis="h", parents=[n1], based_on=[n1]
        )
        assert "Created n2" in out
        # a plan still has to be sound when it is given
        with pytest.raises(TreeError, match="no plan P1"):
            tree.new_node(
                kind="explore", title="e", hypothesis="h", plan_id="P1", based_on=[n1]
            )


class TestAnalyses:
    def test_rules_and_rewrite(self, strict) -> None:
        tree, root, clock = strict
        n1 = first_node(tree, root, clock, new_plan(tree))
        with pytest.raises(TreeError, match="not a finished node"):
            tree.analysis(node="n7", text="t", flaws=["f"])
        with pytest.raises(TreeError, match="text is required"):
            tree.analysis(node=n1, text=" ", flaws=["f"])
        with pytest.raises(TreeError, match="at least one flaw"):
            tree.analysis(node=n1, text="t")
        tree.analysis(node=n1, text="first\n", flaws=["f1"])
        tree.analysis(node=n1, text="second\n", flaws=["f1"], improvements=["i1"])
        assert report_path(root, "analyses", "A1").read_text() == "second\n"
        assert (root / ".research" / "reports_history" / "A1.v1.md").read_text() == "first\n"
        assert "A1 of n1: 1 flaws, 1 improvements" in tree.status()

    def test_a_running_node_cannot_be_analysed(self, strict) -> None:
        tree, *_ = strict
        tree.new_node(kind="pilot", title="p", hypothesis="h")
        with pytest.raises(TreeError, match="not a finished node"):
            tree.analysis(node="n1", text="t", flaws=["f"])


def test_reports_stay_out_of_git(strict) -> None:
    tree, root, clock = strict
    n1 = first_node(tree, root, clock, new_plan(tree))
    tree.analysis(node=n1, text="t", flaws=["f"])
    assert (root / "reports" / "plans" / "P1.md").is_file()
    assert "reports" not in git(root, "status", "--porcelain")


class TestTools:
    @pytest.fixture
    def made(self, tmp_path):
        root = tmp_path / "ws"
        root.mkdir()
        tools = create_research_tree_tools(
            root, judge=FakeJudge(), now=Clock(), require_plans=True
        )
        by_name = {t.name: t for t in tools}
        by_name["tree_init"].invoke({"budget_minutes": 600})
        return by_name

    def test_refusals_come_back_as_text(self, made) -> None:
        assert made["tree_plan"].invoke({}).startswith("ERROR: give either `text`")
        out = made["tree_new"].invoke(
            {"kind": "explore", "title": "t", "hypothesis": "h"}
        )
        assert out.startswith("ERROR: explore nodes need an accepted plan")

    def test_plan_then_node_through_the_tools(self, made) -> None:
        assert made["tree_plan"].invoke({"text": PLAN}).startswith("P1 recorded")
        assert "accepted" in made["tree_plan"].invoke(
            {"plan_id": "P1", "decision": "accepted"}
        )
        out = made["tree_new"].invoke(
            {"kind": "explore", "title": "t", "hypothesis": "h", "plan_id": "P1"}
        )
        assert out.startswith("Created n1")
        assert "answer-free windows" in out


def test_the_prompt_section_follows_the_flag() -> None:
    off = get_system_prompt(research_tree=True)
    on = get_system_prompt(research_tree=True, research_tree_reports=True)
    assert RESEARCH_TREE_REPORTS not in off
    assert RESEARCH_TREE_REPORTS in on
    assert RESEARCH_TREE_REPORTS not in get_system_prompt(research_tree_reports=True)
    for name in ("plan-writer", "node-analyst", "tree_plan", "tree_analysis"):
        assert name in RESEARCH_TREE_REPORTS


class TestSubagents:
    @staticmethod
    def _load(name: str) -> dict:
        from EvoScientist.utils import load_subagents

        directory = Path(__file__).parents[1] / "EvoScientist" / "subagents"
        return next(s for s in load_subagents(directory) if s["name"] == name)

    @pytest.mark.parametrize("name", ["plan-writer", "node-analyst"])
    def test_tools_resolve_and_the_pull_tool_is_absent(self, tmp_path, name) -> None:
        from EvoScientist import EvoScientist as evo

        sub = self._load(name)
        assert sub["name"] == name
        granted = set(sub["_tool_names"])
        assert "apply_experience" not in granted
        cfg = SimpleNamespace(
            memory_experience_search_enabled=True,
            memory_paper_fulltext_enabled=True,
            memory_experience_policy_enabled=True,
            memory_experience_coach_enabled=True,
            memory_experience_policy_max_selected=4,
            memory_experience_main_reuse_enabled=False,
            research_tree_enabled=True,
            research_tree_require_plans=True,
        )
        registry, _ = evo._build_paper_tools(
            cfg=cfg, workspace=Workspace(tmp_path), work_dir=tmp_path
        )
        assert granted <= set(registry)

    def test_neither_writes_to_the_tree(self) -> None:
        for name in ("plan-writer", "node-analyst"):
            granted = set(self._load(name)["_tool_names"])
            assert {"tree_status", "tree_history"} <= granted
            assert not granted & {
                "tree_new",
                "tree_finish",
                "tree_plan",
                "tree_analysis",
                "tree_insight",
                "tree_init",
                "tree_finalize",
            }

    def test_named_in_the_config_they_get_a_coach_labelled_with_their_name(
        self, tmp_path
    ) -> None:
        from tests.test_experience_coach import TestCoachedSubagents as T

        subs = [self._load("plan-writer"), self._load("node-analyst")]
        subs.append(self._load("planner-agent"))
        T._inject(
            tmp_path,
            subs,
            memory_experience_coach_subagents="plan-writer,node-analyst",
        )
        writer, analyst, planner = subs
        assert T._coached(writer) and T._coached(analyst)
        assert not T._coached(planner)
        coach = next(m for m in writer["middleware"] if m.name == "experience_coach")
        assert coach._agent == "plan-writer"
