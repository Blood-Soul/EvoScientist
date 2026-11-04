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
        with pytest.raises(TreeError, match="either a plan file"):
            tree.plan()
        with pytest.raises(TreeError, match="either a plan file"):
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
        # an accepted plan no node uses yet can still be dropped; after that it is final
        tree.plan(plan_id="P1", decision="rejected", reason="the record changed")
        with pytest.raises(TreeError, match="already rejected"):
            tree.plan(plan_id="P1", decision="accepted")
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
        # reports mode: the brief is a file, not text the main agent carries
        assert "nodes/n1/BRIEF.md" in out and "### The accepted plan" not in out
        brief = (root / "nodes" / "n1" / "BRIEF.md").read_text()
        assert "### The accepted plan (P1): build this" in brief
        assert "answer-free windows" in brief
        assert "follow the accepted plan P1" in brief
        assert (root / ".research" / "briefs" / "n1.md").read_text() == brief
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
        tree.new_node(**refine, plan_id=p2)
        brief = (root / "nodes" / "n2" / "BRIEF.md").read_text()
        assert "Analysis A1 of n1:" in brief
        assert "no-answer recall is low" in brief

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
        with pytest.raises(TreeError, match="bullet items under"):
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
        assert made["tree_plan"].invoke({}).startswith("ERROR: give either a plan file")
        out = made["tree_new"].invoke(
            {"kind": "explore", "title": "t", "hypothesis": "h"}
        )
        assert out.startswith("ERROR: explore nodes need an accepted plan")

    def test_plan_then_node_through_the_tools(self, made, tmp_path) -> None:
        draft = tmp_path / "ws" / "reports" / "drafts" / "plan-a.md"
        draft.parent.mkdir(parents=True)
        draft.write_text(PLAN)
        out = made["tree_plan"].invoke({"path": "reports/drafts/plan-a.md"})
        assert out.startswith("P1 recorded")
        assert "accepted" in made["tree_plan"].invoke(
            {"plan_id": "P1", "decision": "accepted"}
        )
        out = made["tree_new"].invoke(
            {"kind": "explore", "title": "t", "hypothesis": "h", "plan_id": "P1"}
        )
        assert out.startswith("Created n1")
        brief = (tmp_path / "ws" / "nodes" / "n1" / "BRIEF.md").read_text()
        assert "answer-free windows" in brief


def test_reports_mode_replaces_the_planner_agent_routes() -> None:
    plain = get_system_prompt(research_tree=True)
    on = get_system_prompt(research_tree=True, research_tree_reports=True)
    # the default prompt keeps the planner-agent routes (they are what the other arms use)
    assert "MODE: PLAN" in plain and "MODE: REFLECTION" in plain and "Plan Update JSON" in plain
    # reports mode: no instruction left that sends planning or reflection there
    for gone in ("MODE: PLAN", "MODE: REFLECTION", "Plan Update JSON", "→ planner-agent"):
        assert gone not in on, gone
    for line in on.splitlines():
        if "planner-agent" in line:
            assert "not used" in line or "Do not use" in line or "in place of planner-agent" in line, line
    assert "→ plan-writer" in on and "→ node-analyst" in on
    # no pressure to land a first node fast; a placeholder deliverable is allowed
    assert "cheap, valid baseline" not in on and "a trivial one is fine" in on


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


class TestArmWithoutExperience:
    """Tree flow with the whole experience system off: same prompts, fewer tools."""

    EXPERIENCE_TOOLS = {
        "enqueue_paper_experiences",
        "extract_paper_experiences",
        "search_experience",
        "list_experience",
        "search_paper_text",
        "read_paper",
        "apply_experience",
    }

    @staticmethod
    def _cfg(**over):
        base = dict(
            memory_paper_extraction_enabled=True,
            memory_experience_search_enabled=True,
            memory_paper_fulltext_enabled=True,
            memory_experience_policy_enabled=True,
            memory_experience_coach_enabled=True,
            memory_experience_policy_max_selected=4,
            memory_experience_main_reuse_enabled=False,
            research_tree_enabled=True,
            research_tree_require_plans=True,
        )
        return SimpleNamespace(**{**base, **over})

    def _tools(self, tmp_path, **over):
        from EvoScientist import EvoScientist as evo

        return evo._build_paper_tools(
            cfg=self._cfg(**over), workspace=Workspace(tmp_path), work_dir=tmp_path
        )

    OFF = dict(
        memory_paper_extraction_enabled=False,
        memory_experience_search_enabled=False,
        memory_paper_fulltext_enabled=False,
        memory_experience_policy_enabled=False,
        memory_experience_coach_enabled=False,
    )

    def test_no_experience_tool_reaches_the_agent(self, tmp_path) -> None:
        registry, base = self._tools(tmp_path, **self.OFF)
        names = set(registry) | {t.name for t in base}
        assert not names & self.EXPERIENCE_TOOLS
        assert {"tree_plan", "tree_analysis", "tree_status", "tree_history"} <= names

    def test_the_switch_defaults_to_the_old_behaviour(self, tmp_path) -> None:
        registry, base = self._tools(tmp_path)
        assert {"enqueue_paper_experiences", "extract_paper_experiences"} <= set(registry)
        assert {"enqueue_paper_experiences", "extract_paper_experiences"} <= {
            t.name for t in base
        }

    @pytest.mark.parametrize("name", ["plan-writer", "node-analyst"])
    def test_the_subagents_keep_only_the_tree_reading_tools(
        self, tmp_path, name
    ) -> None:
        from EvoScientist.utils import resolve_subagent_tools

        registry, _ = self._tools(tmp_path, **self.OFF)
        sub = resolve_subagent_tools(TestSubagents._load(name), registry)
        assert {t.name for t in sub["tools"]} == {
            "think_tool",
            "tree_status",
            "tree_history",
        }


ANALYSIS = """# Analysis of n1
## Worth keeping
- clean windowing, no answer is cut at a boundary
- the null score is calibrated on validation
## Flaws
- no-answer recall is low: 31% of answerable rows are abstained,
  mostly long plots
- the threshold was fitted on 7540 rows only
## Improvements
1. train the null score on answer-free windows (E-0123456789ab)
2. add a second epoch for the long plots
## Experience used
- E-0123456789ab
- E-ffffffffffff
"""


class TestFilesAsHandoff:
    """Sub-agents write documents; the tools read them. The main agent carries paths."""

    def test_read_draft_stays_inside_the_workspace(self, tmp_path) -> None:
        from EvoScientist.research_tree.reports import read_draft

        (tmp_path / "ok.md").write_text("x")
        (tmp_path / "empty.md").write_text("  \n")
        assert read_draft(tmp_path, "ok.md") == "x"
        assert read_draft(tmp_path, "/ok.md") == "x"  # a leading slash is the workspace root
        with pytest.raises(TreeError, match="outside the workspace"):
            read_draft(tmp_path, "../ok.md")
        with pytest.raises(TreeError, match="no file at"):
            read_draft(tmp_path, "missing.md")
        with pytest.raises(TreeError, match="is empty"):
            read_draft(tmp_path, "empty.md")
        with pytest.raises(TreeError, match="path is required"):
            read_draft(tmp_path, " ")

    def test_sections_are_read_as_items(self) -> None:
        from EvoScientist.research_tree.reports import (
            experience_ids,
            plan_title,
            section_items,
        )

        assert section_items(ANALYSIS, "Worth keeping")[0].startswith("clean windowing")
        flaws = section_items(ANALYSIS, "Flaws")
        assert len(flaws) == 2 and "mostly long plots" in flaws[0]  # a wrapped line joins its item
        assert section_items(ANALYSIS, "Improvements")[1] == "add a second epoch for the long plots"
        assert experience_ids(ANALYSIS) == ["E-0123456789ab", "E-ffffffffffff"]
        assert section_items("## Experience used\nnone\n", "Experience used") == []
        assert section_items("## Flaws\nonly prose here\n", "Flaws") == ["only prose here"]
        assert section_items(ANALYSIS, "Nope") == []
        assert plan_title("# Plan: shared-norm reader\n...") == "shared-norm reader"

    @pytest.fixture
    def tools(self, tmp_path):
        root = tmp_path / "ws"
        root.mkdir()
        clock = Clock()
        made = {
            t.name: t
            for t in create_research_tree_tools(
                root, judge=FakeJudge(), now=clock, require_plans=True
            )
        }
        made["tree_init"].invoke({"budget_minutes": 900})
        (root / "reports" / "drafts").mkdir(parents=True)
        return made, root, clock

    def test_a_plan_file_gives_title_and_experience_ids(self, tools) -> None:
        made, root, _ = tools
        (root / "reports" / "drafts" / "p.md").write_text(
            PLAN + "## Experience used\n- E-0123456789ab\n"
        )
        made["tree_plan"].invoke({"path": "reports/drafts/p.md"})
        state = (root / ".research" / "state.json").read_text()
        assert '"title": "shared-norm reader"' in state
        assert "E-0123456789ab" in state

    def test_an_analysis_file_gives_the_lists(self, tools) -> None:
        made, root, clock = tools
        (root / "reports" / "drafts" / "p.md").write_text(PLAN)
        made["tree_plan"].invoke({"path": "reports/drafts/p.md"})
        made["tree_plan"].invoke({"plan_id": "P1", "decision": "accepted"})
        out = made["tree_new"].invoke(
            {"kind": "explore", "title": "t", "hypothesis": "h", "plan_id": "P1"}
        )
        node = root / "nodes" / "n1"
        (node / "holdout_predictions.csv").write_text("1\n" * 400)
        (node / "submission.csv").write_text("d")
        clock.advance(30)
        assert "closed as done" in made["tree_finish"].invoke(
            {"node": "n1", "change": "c", "observation": "o"}
        )
        (root / "reports" / "drafts" / "a.md").write_text(ANALYSIS)
        done = made["tree_analysis"].invoke({"node": "n1", "path": "reports/drafts/a.md"})
        assert "A1 recorded for n1" in done and "2 flaws, 2 improvements" in done
        bad = root / "reports" / "drafts" / "bad.md"
        bad.write_text("# Analysis\nnothing under headings\n")
        assert made["tree_analysis"].invoke(
            {"node": "n1", "path": "reports/drafts/bad.md"}
        ).startswith("ERROR: the analysis needs bullet items")
        # the brief is not part of the node's commit
        names = git(root, "ls-tree", "-r", "--name-only", "node/n1")
        assert "BRIEF.md" not in names and "reports" not in names

    def test_an_unused_accepted_plan_can_be_rewritten_a_used_one_cannot(self, strict) -> None:
        tree, root, _ = strict
        pid = new_plan(tree)
        out = tree.plan(plan_id=pid, text=PLAN + "updated\n")
        assert "draft again" in out and "P1 draft" in tree.status()
        tree.plan(plan_id=pid, decision="accepted")
        tree.new_node(kind="explore", title="t", hypothesis="h", plan_id=pid)
        with pytest.raises(TreeError, match="only a draft, or an accepted plan no node uses"):
            tree.plan(plan_id=pid, text="too late")
        with pytest.raises(TreeError, match="already accepted"):
            tree.plan(plan_id=pid, decision="rejected", reason="late")
