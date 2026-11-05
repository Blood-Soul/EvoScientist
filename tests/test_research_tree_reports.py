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

    def test_a_merge_takes_no_plan_and_pilot_is_gone(self, strict) -> None:
        tree, root, clock = strict
        n1 = first_node(tree, root, clock, new_plan(tree))
        n2 = do_node(
            tree, root, clock, minutes_used=30, n_right=210, shift=7, kind="explore",
            title="other", hypothesis="h", plan_id=new_plan(tree), based_on=[n1],
        )
        tree.finish(n2, change="c", observation="o")
        pid = new_plan(tree)
        with pytest.raises(TreeError, match="does not take a plan"):
            tree.new_node(
                kind="merge", title="m", hypothesis="h", parents=[n1, n2], based_on=[n1, n2], plan_id=pid
            )
        with pytest.raises(TreeError, match="kind must be one of explore, refine, merge"):
            tree.new_node(kind="pilot", title="p", hypothesis="h")

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
        pid = new_plan(tree)
        tree.new_node(kind="explore", title="t", hypothesis="h", plan_id=pid)
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
    assert "cheap, valid baseline" not in on and "placeholder deliverable" in on


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


class TestKeepingTheCoordinatorOnTheFlow:
    """The first end-to-end run showed a prompt is not enough: the main agent built its own code in the
    project root and never used the tree. The tools now say what is due next, and the coordinator cannot
    write code files."""

    def test_next_step_follows_the_record(self, strict) -> None:
        tree, root, clock = strict
        assert "tree_survey" in tree.status().splitlines()[-1]
        tree.survey([{"name": "a", "mechanism": "m"}])
        assert "plan-writer" in tree.status().splitlines()[-1]
        tree.plan(text=PLAN)
        assert "decide on the draft plan" in tree.status().splitlines()[-1] and "P1" in tree.status().splitlines()[-1]
        tree.plan(plan_id="P1", decision="accepted")
        assert "open a node on an accepted plan" in tree.status().splitlines()[-1]
        tree.new_node(kind="explore", title="t", hypothesis="h", plan_id="P1")
        assert "n1 is being built" in tree.status().splitlines()[-1]
        node = root / "nodes" / "n1"
        (node / "holdout_predictions.csv").write_text("1\n" * 400)
        (node / "submission.csv").write_text("d")
        clock.advance(30)
        tree.finish("n1", change="c", observation="o")
        assert "node-analyst to analyse n1" in tree.status().splitlines()[-1]
        tree.analysis(node="n1", text="t", flaws=["f"])
        assert "plan-writer" in tree.status().splitlines()[-1]

    def test_the_hints_are_only_for_runs_that_require_plans(self, tmp_path) -> None:
        tree, *_ = make(tmp_path, require_plans=False)
        assert "Next:" not in tree.status()

    def test_init_tells_the_flow_in_reports_mode_only(self, tmp_path) -> None:
        root = tmp_path / "a"
        root.mkdir()
        strict = ResearchTree(root, judge=FakeJudge(), now=Clock(), require_plans=True)
        assert "You do not write code" in strict.init(budget_minutes=300)
        root2 = tmp_path / "b"
        root2.mkdir()
        plain = ResearchTree(root2, judge=FakeJudge(), now=Clock())
        assert "How this run goes" not in plain.init(budget_minutes=300)

    def test_prompt_forbids_building_in_the_project_root(self) -> None:
        text = get_system_prompt(research_tree=True, research_tree_reports=True)
        assert "You do not build anything in the project root" in text
        assert "cheap, valid baseline" not in text

    @pytest.mark.parametrize(
        "call,refused",
        [
            ({"name": "write_file", "args": {"file_path": "/x/src/prep.py"}}, True),
            ({"name": "edit_file", "args": {"file_path": "train.sh"}}, True),
            ({"name": "write_file", "args": {"file_path": "cfg/run.YAML"}}, True),
            ({"name": "write_file", "args": {"file_path": "reports/drafts/plan-a.md"}}, False),
            ({"name": "write_file", "args": {"file_path": "submission.csv"}}, False),
            ({"name": "execute", "args": {"command": "python x.py"}}, False),
            ({"name": "tree_new", "args": {}}, False),
        ],
    )
    def test_guard_refuses_only_code_files(self, call, refused) -> None:
        from EvoScientist.middleware.coordinator_guard import refuses

        assert refuses(call) is refused

    @pytest.mark.asyncio
    async def test_guard_answers_with_an_error_message_and_lets_the_rest_through(self) -> None:
        from types import SimpleNamespace

        from EvoScientist.middleware.coordinator_guard import CoordinatorGuardMiddleware

        guard = CoordinatorGuardMiddleware()
        seen = []

        async def handler(req):
            seen.append(req)
            return "passed"

        blocked = SimpleNamespace(tool_call={"id": "c1", "name": "write_file", "args": {"file_path": "a.py"}})
        out = await guard.awrap_tool_call(blocked, handler)
        assert out.status == "error" and "nodes/<id>/" in out.content and not seen
        assert guard.wrap_tool_call(blocked, lambda r: "sync").status == "error"
        ok = SimpleNamespace(tool_call={"id": "c2", "name": "write_file", "args": {"file_path": "notes.md"}})
        assert await guard.awrap_tool_call(ok, handler) == "passed" and guard.wrap_tool_call(ok, lambda r: "sync") == "sync"


def _guard_present(require, *, for_async_subagent=False):
    from unittest.mock import MagicMock, patch

    from EvoScientist.EvoScientist import _get_default_middleware

    with (
        patch("EvoScientist.middleware.create_tool_selector_middleware", return_value=[]),
        patch("EvoScientist.EvoScientist._ensure_chat_model") as chat,
        patch("EvoScientist.EvoScientist._ensure_config") as config,
    ):
        cfg = MagicMock()
        cfg.enable_ask_user = False
        cfg.auto_mode = True
        cfg.auxiliary_model = ""
        cfg.auxiliary_provider = ""
        cfg.research_tree_require_plans = require
        config.return_value = cfg
        chat.return_value = MagicMock(profile={"max_input_tokens": 200_000})
        import tempfile

        from EvoScientist.paths import Workspace

        with tempfile.TemporaryDirectory() as tmp:
            mw = _get_default_middleware(workspace=Workspace(Path(tmp)), for_async_subagent=for_async_subagent)
    return any(type(m).__name__ == "CoordinatorGuardMiddleware" for m in mw)


def test_the_guard_is_on_the_main_agent_only_and_only_in_reports_mode() -> None:
    assert _guard_present(True) is True
    assert _guard_present(False) is False
    assert _guard_present(True, for_async_subagent=True) is False


class TestCodeOnlyNodes:
    """gpu=false opens a node that only gets code written: no GPU, no running clock, until tree_start."""

    def _two_plans(self, tree):
        tree.survey([{"name": "a", "mechanism": "m"}])
        return new_plan(tree), new_plan(tree)

    def test_a_code_only_node_does_not_hold_the_gpu_or_the_clock(self, strict) -> None:
        tree, root, clock = strict
        p1, p2 = self._two_plans(tree)
        busy = tree.new_node(kind="explore", title="t", hypothesis="h", plan_id=p1)
        assert "Created n1" in busy and "It is code only" not in busy
        # the GPU is taken, yet a second node can be opened code-only
        out = tree.new_node(kind="explore", title="u", hypothesis="h", plan_id=p2, gpu=False)
        assert "Created n2" in out and "counts from tree_start" in out
        clock.advance(200)  # a long time of code writing
        status = tree.status()
        assert "Prepared (code only, no GPU, no clock): n2" in status
        assert "Running: n1" in status and "Running: n2" not in status
        assert "n2 explore" in status and "prepared" in status
        brief = (root / "nodes" / "n2" / "BRIEF.md").read_text()
        assert "this step is code only for now" in brief and "START.md" in brief
        assert "budget of 60 minutes from now" not in brief
        # the deliverables do not contradict "do not train yet"
        leave = brief[brief.index("### What you must leave"):]
        assert "Before the node ends it must contain" not in leave
        assert "leave the code" in leave and "START.md" in leave
        busy = (root / "nodes" / "n1" / "BRIEF.md").read_text()
        assert "Before the node ends it must contain `holdout_predictions.csv`" in busy[busy.index("### What you must leave"):]

    def test_a_second_gpu_node_is_still_refused(self, strict) -> None:
        tree, *_ = strict
        p1, p2 = self._two_plans(tree)
        tree.new_node(kind="explore", title="t", hypothesis="h", plan_id=p1)
        with pytest.raises(TreeError, match="holds the GPU"):
            tree.new_node(kind="explore", title="u", hypothesis="h", plan_id=p2)

    def test_start_needs_a_free_gpu_and_a_prepared_node(self, strict) -> None:
        tree, root, clock = strict
        p1, p2 = self._two_plans(tree)
        tree.new_node(kind="explore", title="t", hypothesis="h", plan_id=p1)
        tree.new_node(kind="explore", title="u", hypothesis="h", plan_id=p2, gpu=False)
        with pytest.raises(TreeError, match="n1 holds the GPU"):
            tree.start("n2")
        with pytest.raises(TreeError, match="not a prepared code-only node"):
            tree.start("n1")  # on the GPU from the start
        (root / "nodes" / "n1" / "holdout_predictions.csv").write_text("1\n" * 400)
        (root / "nodes" / "n1" / "submission.csv").write_text("d")
        clock.advance(30)
        tree.finish("n1", change="c", observation="o")
        assert "the GPU is free: start one of the prepared nodes (n2) with tree_start(node)" in tree.status().splitlines()[-1]
        out = tree.start("n2", minutes=45)
        assert "n2 started with a budget of 45 min" in out and "START.md" in out
        assert "Running: n2 (explore) 0.0/45 min" in tree.status()
        with pytest.raises(TreeError, match="not a prepared code-only node"):
            tree.start("n2")  # already started
        assert "## Started" in (root / "nodes" / "n2" / "BRIEF.md").read_text()

    def test_the_budget_counts_from_tree_start(self, strict) -> None:
        tree, root, clock = strict
        p1, p2 = self._two_plans(tree)
        tree.new_node(kind="explore", title="t", hypothesis="h", plan_id=p1)
        tree.new_node(kind="explore", title="u", hypothesis="h", plan_id=p2, gpu=False)
        (root / "nodes" / "n1" / "holdout_predictions.csv").write_text("1\n" * 400)
        (root / "nodes" / "n1" / "submission.csv").write_text("d")
        clock.advance(40)
        tree.finish("n1", change="c", observation="o")
        clock.advance(25)  # the GPU sat idle: n2 had no clock
        tree.start("n2", minutes=60)
        clock.advance(10)
        assert "Running: n2 (explore) 10.0/60 min" in tree.status()
        node = root / "nodes" / "n2"
        (node / "holdout_predictions.csv").write_text("1\n" * 200 + "0\n" * 200)
        (node / "submission.csv").write_text("d")
        clock.advance(5)
        done = tree.finish("n2", change="c", observation="o")
        assert "after 15.0 min (budget 60)" in done  # measured from tree_start, not from tree_new

    def test_the_hint_names_the_prepared_nodes(self, strict) -> None:
        tree, root, clock = strict
        p1, p2 = self._two_plans(tree)
        tree.new_node(kind="explore", title="t", hypothesis="h", plan_id=p1)
        tree.new_node(kind="explore", title="u", hypothesis="h", plan_id=p2, gpu=False)
        busy = tree.status().splitlines()[-1]
        assert "n1 is being built" in busy and "prepared and waiting for the GPU: n2" in busy
        assert "gpu=false" in busy
        assert "can be dropped with tree_finish(node, abandon_reason=...)" in busy

    def test_a_prepared_node_is_closed_only_by_dropping_it(self, strict) -> None:
        tree, root, clock = strict
        p1, p2 = self._two_plans(tree)
        tree.new_node(kind="explore", title="t", hypothesis="h", plan_id=p1)
        tree.new_node(kind="explore", title="u", hypothesis="h", plan_id=p2, gpu=False)
        with pytest.raises(TreeError, match="never started"):
            tree.finish("n2", change="c", observation="o")
        out = tree.finish("n2", change="-", observation="-", abandon_reason="not needed")
        assert "n2 closed as abandoned" in out
        # its plan is free again for another node (the abandoned node is history now, so cite it)
        again = tree.new_node(
            kind="explore", title="v", hypothesis="h", plan_id=p2, gpu=False, based_on=["n2"]
        )
        assert again.startswith("Created n3")

    def test_finalize_closes_a_prepared_node_that_never_started(self, strict) -> None:
        tree, root, clock = strict
        p1, p2 = self._two_plans(tree)
        tree.new_node(kind="explore", title="t", hypothesis="h", plan_id=p1)
        tree.new_node(kind="explore", title="u", hypothesis="h", plan_id=p2, gpu=False)
        (root / "nodes" / "n1" / "holdout_predictions.csv").write_text("1\n" * 400)
        (root / "nodes" / "n1" / "submission.csv").write_text("d")
        clock.advance(30)
        tree.finish("n1", change="c", observation="o")
        tree.finalize()
        assert "abandoned" in git(root, "notes", "--ref=outcome", "show", "node/n2")

    def test_the_prompt_and_the_tool_say_how_it_works(self) -> None:
        text = get_system_prompt(research_tree=True, research_tree_reports=True)
        assert "tree_start" in text and "gpu=false" in text and "START.md" in text
        made = {t.name: t for t in create_research_tree_tools(Path("."))}
        assert "tree_start" in made and "node" in made["tree_start"].args


class TestAnalysisAndNextPlanInOneRun:
    """node-analyst writes the analysis and the plan that follows; the tools record each from its file."""

    NEXT = "# Plan: fix the end label\n## What to build\n1. one line\n## Evidence\n- the choice (A1 flaw 1)\n## Experience used\nnone\n"

    @pytest.fixture
    def tools(self, tmp_path):
        root = tmp_path / "ws"
        root.mkdir()
        clock = Clock()
        made = {
            t.name: t
            for t in create_research_tree_tools(root, judge=FakeJudge(), now=clock, require_plans=True)
        }
        made["tree_init"].invoke({"budget_minutes": 600})
        made["tree_survey"].invoke({"clusters": [{"name": "A", "mechanism": "m"}]})
        d = root / "reports" / "drafts"
        d.mkdir(parents=True)
        (d / "plan-a.md").write_text(PLAN)
        made["tree_plan"].invoke({"path": "reports/drafts/plan-a.md", "cluster": "A"})
        made["tree_plan"].invoke({"plan_id": "P1", "decision": "accepted"})
        made["tree_new"].invoke({"kind": "explore", "title": "t", "hypothesis": "h", "plan_id": "P1"})
        n = root / "nodes" / "n1"
        (n / "holdout_predictions.csv").write_text("1\n" * 400)
        (n / "submission.csv").write_text("d")
        clock.advance(30)
        made["tree_finish"].invoke({"node": "n1", "change": "c", "observation": "o"})
        return made, root

    def test_the_two_files_of_one_analyst_are_recorded_and_the_refine_opens(self, tools) -> None:
        made, root = tools
        d = root / "reports" / "drafts"
        (d / "analysis-n1.md").write_text(ANALYSIS)
        (d / "plan-next-n1.md").write_text(self.NEXT)
        assert "A1 recorded for n1" in made["tree_analysis"].invoke({"node": "n1", "path": "reports/drafts/analysis-n1.md"})
        out = made["tree_plan"].invoke({"path": "reports/drafts/plan-next-n1.md", "parents": ["n1"], "cluster": "A"})
        assert out.startswith("P2 recorded")
        made["tree_plan"].invoke({"plan_id": "P2", "decision": "accepted"})
        opened = made["tree_new"].invoke(
            {"kind": "refine", "title": "r", "hypothesis": "h", "parents": ["n1"], "based_on": ["n1"], "plan_id": "P2"}
        )
        assert opened.startswith("Created n2")
        brief = (root / "nodes" / "n2" / "BRIEF.md").read_text()
        assert "fix the end label" in brief and "Analysis A1 of n1" in brief  # the plan and the analysis, together

    def test_a_next_plan_without_the_analysis_still_cannot_open_a_refine(self, tools) -> None:
        made, root = tools
        (root / "reports" / "drafts" / "plan-next-n1.md").write_text(self.NEXT)
        made["tree_plan"].invoke({"path": "reports/drafts/plan-next-n1.md", "parents": ["n1"], "cluster": "A"})
        made["tree_plan"].invoke({"plan_id": "P2", "decision": "accepted"})
        out = made["tree_new"].invoke(
            {"kind": "refine", "title": "r", "hypothesis": "h", "parents": ["n1"], "based_on": ["n1"], "plan_id": "P2"}
        )
        assert out.startswith("ERROR: n1 has no analysis yet")

    def test_the_hint_after_a_finished_node_names_both_files_to_record(self, tools) -> None:
        made, _ = tools
        last = made["tree_status"].invoke({}).splitlines()[-1]
        assert "node-analyst to analyse n1 and write the plan that follows" in last
        assert "tree_analysis(node, path)" in last and "tree_plan(path, parents=[node])" in last

    def test_the_definitions_and_the_prompt_describe_the_merge(self) -> None:
        text = get_system_prompt(research_tree=True, research_tree_reports=True)
        assert "the two files to write" in text and "plan-next-<node>.md" in text
        assert "ONE `code_interpreter` eval" in text and "start_async_task" in text
        assert "subagentType" in text
        from tests.test_research_tree_reports import TestSubagents

        analyst = TestSubagents._load("node-analyst")["system_prompt"]
        assert "### The next plan" in analyst and "plan-next-<node id>.md" in analyst
        assert "may run small experiments" in analyst
        writer = TestSubagents._load("plan-writer")["system_prompt"]
        assert "states the result and the choice, not the story of the experiment" in writer
