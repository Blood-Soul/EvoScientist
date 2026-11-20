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


def record_analysis(tree, node: str, text: str = "# Analysis\n## What the node showed\n- x\n") -> str:
    """What node-analyst's file becomes once the main agent records it."""
    return tree.analysis(node=node, text=text, path=f"reports/drafts/analysis-{node}.md")


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

    def test_a_refine_needs_a_matching_plan_and_nothing_else(self, strict) -> None:
        tree, root, clock = strict
        n1 = first_node(tree, root, clock, new_plan(tree))
        refine = dict(
            kind="refine", title="r", hypothesis="h", parents=[n1], based_on=[n1]
        )
        p_root = new_plan(tree)
        with pytest.raises(TreeError, match="must match"):
            tree.new_node(**refine, plan_id=p_root)
        # a plan that builds on n1 is written from n1's analysis: refused until it is recorded
        with pytest.raises(TreeError, match="n1 has no analysis yet"):
            new_plan(tree, parents=[n1])
        record_analysis(tree, n1)
        p2 = new_plan(tree, parents=[n1])
        tree.new_node(**refine, plan_id=p2)
        brief = (root / "nodes" / "n2" / "BRIEF.md").read_text()
        assert "reports/analysis/n1.md" in brief

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


def test_reports_stay_out_of_git(strict) -> None:
    tree, root, clock = strict
    first_node(tree, root, clock, new_plan(tree))
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


def test_reports_mode_rewrites_the_staged_plan_and_code_mode_blocks() -> None:
    plain = get_system_prompt(research_tree=True)
    on = get_system_prompt(research_tree=True, research_tree_reports=True)
    # the single-agent blocks stay in the tree-without-plans profile
    assert "Save the plan to `todos.md`" in plain
    assert "success_criteria.md" in plain
    assert "Before any code delegation, you MUST complete the Code Generation Mode Selection below." in plain
    assert "- Update `todos.md` to reflect new iterations." in plain
    # reports mode: the plan is the tree's, and node code comes with tree_new's brief
    assert "Save the plan to `todos.md`" not in on
    assert "Use `write_todos` to track the execution plan" not in on
    assert "`success_criteria.md` for success signals" not in on
    assert "Code Generation Mode Selection" not in on
    assert "ask the user which code generation mode" not in on
    assert "- Update `todos.md` to reflect new iterations." not in on
    assert "The tree is the plan." in on
    assert "does not stop to ask which code-generation mode" in on
    assert "recorded by the node it closed (`tree_finish`)" in on


def test_a_tree_run_keeps_one_report_and_drops_the_single_agent_template() -> None:
    single = get_system_prompt()
    tree = [
        get_system_prompt(research_tree=True),
        get_system_prompt(research_tree=True, research_tree_reports=True),
    ]
    # the single-agent flow still writes final_report.md from the template
    assert "Write the final report to `final_report.md`" in single
    assert "Experiment Report Template" in single
    # a tree run reports through TRAJECTORY.md; writing a second report is refused by the prompt
    for text in tree:
        assert "Write the final report to `final_report.md`" not in text
        assert "Experiment Report Template" not in text
        assert "Do not also write `final_report.md`" in text
        assert "TRAJECTORY.md` is the report" in text
        assert "tree_finalize" in text
        assert "Confirm `tree_finalize` published the champion's deliverable" in text


def test_the_literature_and_diagnosis_bullets_belong_to_node_analyst_in_a_plan_run() -> None:
    plans = get_system_prompt(research_tree=True, research_tree_reports=True)
    assert "Use the literature where a step is stuck" not in plans
    assert "diagnose from its own errors first" not in plans
    assert "starts from that node's analysis" in plans


def test_the_tree_bullets_follow_the_main_agents_tools() -> None:
    none = get_system_prompt(research_tree=True)
    assert "Stored experience" not in none
    # The literature bullet rests on the skill, not on a switch, so this arm
    # keeps it even when no experience or full-text tool is registered.
    assert "Use the literature where a step is stuck" in none
    assert "(only if you have" not in none

    search_only = get_system_prompt(
        research_tree=True,
        available_tools=["search_experience", "list_experience", "read_paper"],
    )
    assert "Stored experience" in search_only
    assert "Look something up when a decision calls for it." in search_only
    assert "apply_experience" not in search_only

    full = get_system_prompt(
        research_tree=True,
        available_tools=["search_experience", "search_paper_text", "read_paper"],
    )
    assert "Stored experience" in full
    assert "apply_experience" not in full
    assert "Use the literature where a step is stuck" in full


def test_the_background_survey_does_not_promise_stored_experience() -> None:
    off = get_system_prompt(
        research_tree=True,
        research_tree_background=True,
    )
    assert "on the experience it stored" not in off
    assert (
        "the plans stand on what the survey found, and a plan written before"
        " that does not have it." in off
    )

    on = get_system_prompt(
        research_tree=True,
        research_tree_background=True,
        available_tools=["search_experience"],
    )
    assert "on the experience it stored" in on


def test_the_plan_writer_is_not_described_with_a_coach() -> None:
    text = get_system_prompt(research_tree=True, research_tree_reports=True)
    assert "experience coach is mounted" not in text
    assert "The plan-writer does not see this conversation" in text


def test_the_configured_prompt_reads_the_built_tool_list(tmp_path) -> None:
    from types import SimpleNamespace

    from EvoScientist.EvoScientist import _configured_system_prompt

    cfg = SimpleNamespace(
        dangerous_mode=False,
        research_tree_enabled=True,
        research_tree_require_plans=False,
        research_tree_background_tasks=False,
    )

    def tool(name: str):
        return SimpleNamespace(name=name)

    with_experience = _configured_system_prompt(
        cfg, tmp_path, [tool("search_experience"), tool("apply_experience")]
    )
    assert "Stored experience" in with_experience
    without = _configured_system_prompt(cfg, tmp_path, [tool("tree_status")])
    assert "Stored experience" not in without
    assert "Use the literature where a step is stuck" in without


def test_the_prompt_section_follows_the_flag() -> None:
    off = get_system_prompt(research_tree=True)
    on = get_system_prompt(research_tree=True, research_tree_reports=True)
    assert RESEARCH_TREE_REPORTS not in off
    assert RESEARCH_TREE_REPORTS in on
    assert RESEARCH_TREE_REPORTS not in get_system_prompt(research_tree_reports=True)
    for name in ("plan-writer", "node-analyst", "tree_plan"):
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
            memory_plan_review_enabled=True,   # the YAMLs grant `review_plan`; the arm that uses them turns it on
            research_tree_enabled=True,
            research_tree_require_plans=True,
        )
        registry, _ = evo._build_paper_tools(
            cfg=cfg, workspace=Workspace(tmp_path), work_dir=tmp_path
        )
        assert granted <= set(registry)
        # with the switch off the tool is not registered, so a YAML that grants it is skipped with a warning
        cfg.memory_plan_review_enabled = False
        off, _ = evo._build_paper_tools(cfg=cfg, workspace=Workspace(tmp_path), work_dir=tmp_path)
        assert "review_plan" not in off

    @pytest.mark.parametrize("name", ["planner-agent", "research-agent"])
    def test_the_agents_that_do_the_work_can_look_experience_up(self, name) -> None:
        granted = set(self._load(name)["_tool_names"])
        assert {"search_experience", "list_experience"} <= granted
        assert "apply_experience" not in granted

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


class TestArmWithoutExperience:
    """Tree flow with the whole experience system off: same YAML, fewer tools and prompt blocks."""

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
        assert {"tree_plan", "tree_status", "tree_history"} <= names

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
        reading = {"think_tool", "tree_status", "tree_history"}
        # node-analyst records the choices a node leaves open; plan-writer lists them in its plan
        expected = reading | {"tree_choice"} if name == "node-analyst" else reading
        assert {t.name for t in sub["tools"]} == expected

    def test_the_review_step_is_rendered_only_when_the_tool_is_on(self, tmp_path) -> None:
        from EvoScientist.utils import resolve_subagent_tools

        on_registry, _ = self._tools(tmp_path, memory_plan_review_enabled=True)
        on = resolve_subagent_tools(TestSubagents._load("plan-writer"), on_registry)
        assert "Then call `review_plan` once" in on["system_prompt"]
        analyst_on = resolve_subagent_tools(
            TestSubagents._load("node-analyst"), on_registry
        )
        # node-analyst writes an analysis, not a plan: the review belongs to the plan
        assert "review_plan" not in analyst_on["system_prompt"]
        assert "review_plan" not in {t.name for t in analyst_on["tools"]}

        off_registry, _ = self._tools(tmp_path, **self.OFF)
        for name in ("plan-writer", "node-analyst"):
            off = resolve_subagent_tools(TestSubagents._load(name), off_registry)
            assert "review_plan" not in off["system_prompt"]
            # the placeholder is gone, not left as an empty bullet
            assert "- \n" not in off["system_prompt"]

    def test_enqueue_is_agent_driven_only_without_paper_autoqueue(self, tmp_path) -> None:
        from EvoScientist.utils import resolve_subagent_tools

        # autoqueue off: the search results are not queued in code, so the agent
        # keeps the tool and the prompt tells it the fixed post-selection step
        manual_registry, _ = self._tools(tmp_path)
        manual = resolve_subagent_tools(
            TestSubagents._load("research-agent"), manual_registry
        )
        assert "enqueue_paper_experiences" in {t.name for t in manual["tools"]}
        assert (
            "call `enqueue_paper_experiences` yourself exactly once"
            in " ".join(manual["system_prompt"].split())
        )

        # autoqueue on: every scholar_search result is queued by the middleware,
        # so the tool is not registered here and the prompt block drops out; the
        # skill skips its manual step when the tool is absent
        auto_registry, _ = self._tools(
            tmp_path, memory_paper_autoqueue_enabled=True
        )
        assert "enqueue_paper_experiences" not in auto_registry
        auto = resolve_subagent_tools(
            TestSubagents._load("research-agent"), auto_registry
        )
        assert "enqueue_paper_experiences" not in {t.name for t in auto["tools"]}
        assert "enqueue_paper_experiences" not in auto["system_prompt"]
        assert "`paper-navigator` is a skill, not a sub-agent" in auto["system_prompt"]


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
            plan_title,
            section_items,
        )

        assert section_items(ANALYSIS, "Worth keeping")[0].startswith("clean windowing")
        flaws = section_items(ANALYSIS, "Flaws")
        assert len(flaws) == 2 and "mostly long plots" in flaws[0]  # a wrapped line joins its item
        assert section_items(ANALYSIS, "Improvements")[1] == "add a second epoch for the long plots"
        assert section_items("## Flaws\nnone\n", "Flaws") == []
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

    def test_a_plan_file_gives_its_title(self, tools) -> None:
        made, root, _ = tools
        (root / "reports" / "drafts" / "p.md").write_text(PLAN)
        made["tree_plan"].invoke({"path": "reports/drafts/p.md"})
        state = (root / ".research" / "state.json").read_text()
        assert '"title": "shared-norm reader"' in state

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
        after = tree.status().splitlines()[-1]
        assert "plan-writer" in after
        assert "Analysis due for n1" in after and "node-analyst" in after
        assert "nothing waits on it except a plan that builds on that node" in after
        record_analysis(tree, "n1")
        assert "Analysis due" not in tree.status().splitlines()[-1]

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
        cfg.code_interpreter_timeout = 60.0
        cfg.code_interpreter_max_result_chars = 10_000
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
        assert "can be dropped with tree_finish(node, abandon_reason=...)" in busy
        assert "gpu=false" not in busy  # the hint does not push preparing nodes while a node runs

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
        # the capability stays in the tools; the prompt does not tell the agent to use it while a node runs
        text = get_system_prompt(research_tree=True, research_tree_reports=True)
        assert "tree_start" not in text and "While the GPU is busy" not in text
        made = {t.name: t for t in create_research_tree_tools(Path("."))}
        assert "tree_start" in made and "node" in made["tree_start"].args


class TestNextPlanFromTheAnalyst:
    """node-analyst analyses a finished node; tree_analysis records it; plan-writer's plan that builds on
    the node is then recorded with tree_plan and a refine opens on it."""

    ANALYSIS = "# Analysis: n1\n## What the node showed\n- the end label is off by one (slice x)\n"

    NEXT = "# Plan: fix the end label\n## What to build\n1. one line\n## Evidence\n- the choice (n1 slice)\n"

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
        finished = made["tree_finish"].invoke({"node": "n1", "change": "c", "observation": "o"})
        assert "Start node-analyst on n1 now with task_start" in finished
        return made, root

    @staticmethod
    def analyse(made, root, text: str = ANALYSIS) -> str:
        (root / "reports" / "drafts" / "analysis-n1.md").write_text(text)
        return made["tree_analysis"].invoke({"node": "n1", "path": "reports/drafts/analysis-n1.md"})

    def test_a_plan_that_builds_on_a_node_waits_for_its_analysis(self, tools) -> None:
        made, root = tools
        d = root / "reports" / "drafts"
        (d / "plan-next-n1.md").write_text(self.NEXT)
        refused = made["tree_plan"].invoke({"path": "reports/drafts/plan-next-n1.md", "parents": ["n1"], "cluster": "A"})
        assert refused.startswith("ERROR: n1 has no analysis yet")
        assert "task_start" in refused and "tree_analysis" in refused and "new direction" in refused
        # a plan for a new direction does not wait
        (d / "plan-b.md").write_text(PLAN)
        assert made["tree_plan"].invoke({"path": "reports/drafts/plan-b.md", "cluster": "A"}).startswith("P2 recorded")
        assert "Analysis of n1 recorded" in self.analyse(made, root)
        assert (root / "reports" / "analysis" / "n1.md").read_text().startswith("# Analysis: n1")
        out = made["tree_plan"].invoke({"path": "reports/drafts/plan-next-n1.md", "parents": ["n1"], "cluster": "A"})
        assert out.startswith("P3 recorded")
        made["tree_plan"].invoke({"plan_id": "P3", "decision": "accepted"})
        opened = made["tree_new"].invoke(
            {"kind": "refine", "title": "r", "hypothesis": "h", "parents": ["n1"], "based_on": ["n1"], "plan_id": "P3"}
        )
        assert opened.startswith("Created n2")  # node ids and plan ids are separate counters
        brief = (root / "nodes" / "n2" / "BRIEF.md").read_text()
        assert "fix the end label" in brief and "reports/analysis/n1.md" in brief

    def test_an_analysis_is_of_a_finished_node_and_can_be_replaced(self, tools) -> None:
        made, root = tools
        assert made["tree_analysis"].invoke({"node": "n9", "path": "reports/drafts/x.md"}).startswith("ERROR")
        self.analyse(made, root)
        self.analyse(made, root, "# Analysis: n1 (v2)\n")
        assert (root / "reports" / "analysis" / "n1.md").read_text().startswith("# Analysis: n1 (v2)")
        assert (root / ".research" / "reports_history" / "n1.v1.md").is_file()

    def test_the_plan_is_recorded_and_the_refine_opens_on_it(self, tools) -> None:
        made, root = tools
        self.analyse(made, root)
        d = root / "reports" / "drafts"
        (d / "plan-next-n1.md").write_text(self.NEXT)
        out = made["tree_plan"].invoke({"path": "reports/drafts/plan-next-n1.md", "parents": ["n1"], "cluster": "A"})
        assert out.startswith("P2 recorded")
        made["tree_plan"].invoke({"plan_id": "P2", "decision": "accepted"})
        opened = made["tree_new"].invoke(
            {"kind": "refine", "title": "r", "hypothesis": "h", "parents": ["n1"], "based_on": ["n1"], "plan_id": "P2"}
        )
        assert opened.startswith("Created n2")
        assert "fix the end label" in (root / "nodes" / "n2" / "BRIEF.md").read_text()

    def test_a_plan_or_an_observation_in_based_on_is_left_out_and_said_so(self, tools) -> None:
        made, root = tools
        self.analyse(made, root)
        d = root / "reports" / "drafts"
        (d / "plan-next.md").write_text(self.NEXT)
        made["tree_plan"].invoke({"path": "reports/drafts/plan-next.md", "parents": ["n1"], "cluster": "A"})
        made["tree_plan"].invoke({"plan_id": "P2", "decision": "accepted"})
        out = made["tree_new"].invoke(
            {"title": "r", "hypothesis": "h", "parents": ["n1"], "plan_id": "P2",
             "based_on": ["P2", "O-b9f64986cba9a35d", "n1"]}
        )
        assert out.startswith("Created n2")
        assert "Left out of based_on" in out and "P2" in out and "O-b9f64986cba9a35d" in out
        # a made-up id is still refused
        again = made["tree_new"].invoke(
            {"title": "r", "hypothesis": "h", "parents": ["n1"], "based_on": ["n9"]}
        )
        assert again.startswith("ERROR: based_on cites unknown or unfinished ids: n9")

    def test_kind_follows_from_the_parents_when_left_out(self, tools) -> None:
        made, root = tools
        d = root / "reports" / "drafts"
        (d / "plan-b.md").write_text(PLAN)
        made["tree_plan"].invoke({"path": "reports/drafts/plan-b.md", "cluster": "A"})
        made["tree_plan"].invoke({"plan_id": "P2", "decision": "accepted"})
        out = made["tree_new"].invoke({"title": "e", "hypothesis": "h", "plan_id": "P2", "based_on": ["n1"]})
        assert out.startswith("Created n2 (explore)")

    def test_a_rewrite_takes_the_new_parents_and_family(self, tools) -> None:
        made, root = tools
        self.analyse(made, root)
        d = root / "reports" / "drafts"
        (d / "plan-b.md").write_text(PLAN)
        made["tree_plan"].invoke({"path": "reports/drafts/plan-b.md", "cluster": "A"})  # P2, from the root
        out = made["tree_plan"].invoke(
            {"path": "reports/drafts/plan-b.md", "plan_id": "P2", "parents": ["n1"], "cluster": "A"}
        )
        assert "rewritten" in out and "from n1" in out
        made["tree_plan"].invoke({"plan_id": "P2", "decision": "accepted"})
        opened = made["tree_new"].invoke(
            {"kind": "refine", "title": "r", "hypothesis": "h", "parents": ["n1"], "plan_id": "P2", "based_on": ["n1"]}
        )
        assert opened.startswith("Created n2 (refine)"), opened

    def test_an_unknown_family_says_how_to_add_one(self, tools) -> None:
        made, root = tools
        (root / "reports" / "drafts" / "plan-c.md").write_text(PLAN)
        out = made["tree_plan"].invoke({"path": "reports/drafts/plan-c.md", "cluster": "Z"})
        assert out.startswith("ERROR: cluster 'Z' is not a surveyed family") and "added with tree_survey" in out

    def test_an_explore_and_a_refine_are_opened_the_same_way(self, tools) -> None:
        made, root = tools
        d = root / "reports" / "drafts"
        (d / "plan-b.md").write_text(PLAN)
        made["tree_plan"].invoke({"path": "reports/drafts/plan-b.md", "cluster": "A"})
        made["tree_plan"].invoke({"plan_id": "P2", "decision": "accepted"})
        out = made["tree_new"].invoke({"kind": "explore", "title": "e", "hypothesis": "h", "plan_id": "P2", "based_on": ["n1"]})
        assert out.startswith("Created n2")

    def test_after_a_finished_node_both_kinds_of_next_step_stay_open(self, tools) -> None:
        made, root = tools
        self.analyse(made, root)
        last = made["tree_status"].invoke({}).splitlines()[-1]
        assert "plan-writer" in last and "new direction" in last and "builds on an analysed node" in last
        assert "refine" not in last.lower() and "deepen" not in last.lower()

    def test_the_roles_are_stated_once_and_agree(self) -> None:
        text = get_system_prompt(research_tree=True, research_tree_reports=True, research_tree_background=True)
        # who does what, with no "do it yourself or delegate" escape
        assert "Do this yourself or leave it to" not in text
        assert "You do not write the analysis yourself: read it." in text
        assert "`plan-writer` writes every plan." in text
        assert "tree_analysis(node=..., path=...)" in text and "analysis-<node>.md" in text
        assert "choose among them" in text and "tree_choice" in text
        assert "Do not fold an untested design change into a plan." in text
        # no leftover from the flow where node-analyst wrote the plan
        assert "plan-next-<node>.md" not in text
        assert "planned by `node-analyst`" not in text
        assert '"Read a finished node and write the plan for a step that builds on it" → node-analyst' not in text
        assert "ONE `code_interpreter` eval" not in text
        analyst = TestSubagents._load("node-analyst")["system_prompt"]
        assert "You do not\nwrite a plan" in analyst and "analysis-<node id>.md" in analyst
        assert "## What the node showed" in analyst and "may run small experiments" in analyst
        assert "plan-next" not in analyst and "## What to build" not in analyst
        writer = TestSubagents._load("plan-writer")["system_prompt"]
        assert "states the result and the choice, not the story of the experiment" in writer
        assert "reports/analysis/<node>.md" in writer and "## Open choices" in writer

    def test_what_waits_for_what_is_stated(self) -> None:
        text = get_system_prompt(research_tree=True, research_tree_reports=True, research_tree_background=True)
        part = text.split("**What waits for what.**")[1].split("\n")[0]
        assert "start node-analyst on it at once with `task_start`" in part
        assert "The one thing that waits is a plan that builds on that node" in part
        assert "open the reserve plan if there is one" in part
        assert "Background sub-agents work on the CPU while a node holds the GPU" in part


def test_the_literature_is_read_by_the_writer_and_by_the_analyst() -> None:
    writer = TestSubagents._load("plan-writer")["system_prompt"]
    assert "Look at the literature for the direction before you write" in writer
    analyst = TestSubagents._load("node-analyst")["system_prompt"]
    assert analyst.index("search for papers that address them") < analyst.index("## Rules")
    assert "whether it\n  transfers to this task is a judgment you state as yours" in analyst


def test_plans_are_asked_for_a_design_document_not_interfaces_and_a_source_for_every_choice() -> None:
    writer = TestSubagents._load("plan-writer")["system_prompt"]
    assert "design document or a technical report" in writer
    assert "does about the same thing" in writer and "are for the implementer to" in writer
    assert "plan's to decide: learning rate, batch size" in " ".join(writer.split())
    assert "Specific enough that an engineer" not in writer
    # the one rule that stays: no false citations; own judgment is allowed and needs no source line
    assert "Never present a detail as coming from a paper or a record unless you have read it there" in writer
    assert 'labelled "own derivation"' not in writer and "Every concrete choice" not in writer
    assert "Not every choice needs a line" in writer
    # node-analyst writes no plan, so none of the plan-writing rules belong in its definition
    analyst = TestSubagents._load("node-analyst")["system_prompt"]
    assert "written like a design document" not in analyst
    # a hint, not a cap: no word or minute limit appears in either definition
    import re
    for text in (writer, analyst):
        assert not re.search(r"\b(under|at most|no more than) \d+ (words|minutes|lines)\b", text)


class TestOpenChoices:
    """A choice the run has not settled: recorded on a node, listed by tree_status, carried into briefs,
    closed by a later node or by the analyst on saved outputs."""

    @pytest.fixture
    def run(self, tmp_path):
        tree, root, clock = make(tmp_path, require_plans=True)
        tree.survey([{"name": "A", "mechanism": "m"}])
        n1 = first_node(tree, root, clock, new_plan(tree))
        return tree, root, clock, n1

    def test_a_choice_is_recorded_listed_and_ordered(self, run) -> None:
        tree, _, _, n1 = run
        low = tree.choice(node=n1, question="stride 128 or 64", options=["128", "64"], default="128",
                          why_open="never compared", needs="full_node", effect="low")
        high = tree.choice(node=n1, question="gold occurrence", options=["first", "random"], default="random",
                           why_open="review suggestion, untested", needs="short_training", effect="high",
                           source="review of P1, suggestion 1")
        assert low.startswith("C1 recorded on n1") and "default (128) stays" in low
        assert high.startswith("C2 recorded on n1") and f"refine node of {n1} that changes only this" in high
        status = tree.status()
        listed = status.split("Open choices")[1]
        # higher expected effect first, whatever the order they were recorded in
        assert listed.index("C2") < listed.index("C1")
        assert "random (default)" in listed and "needs short_training, effect high" in listed

    def test_bad_choices_are_refused(self, run) -> None:
        tree, _, _, n1 = run
        with pytest.raises(TreeError, match="at least two options"):
            tree.choice(node=n1, question="q", options=["a"], why_open="w", needs="inference", effect="low")
        with pytest.raises(TreeError, match="default must be one of the options"):
            tree.choice(node=n1, question="q", options=["a", "b"], default="c", why_open="w", needs="inference", effect="low")
        with pytest.raises(TreeError, match="needs must be one of"):
            tree.choice(node=n1, question="q", options=["a", "b"], why_open="w", needs="soon", effect="low")
        with pytest.raises(TreeError, match="is not a node"):
            tree.choice(node="n9", question="q", options=["a", "b"], why_open="w", needs="inference", effect="low")

    def test_a_choice_is_closed_with_its_result_and_leaves_the_list(self, run) -> None:
        tree, _, _, n1 = run
        tree.choice(node=n1, question="veto only or rescue", options=["veto", "veto+rescue"], why_open="w",
                    needs="inference", effect="medium")
        with pytest.raises(TreeError, match="settled or dropped"):
            tree.choice(choice_id="C1", status="open", result="x")
        with pytest.raises(TreeError, match="say what closed it"):
            tree.choice(choice_id="C1", status="settled")
        out = tree.choice(choice_id="C1", status="settled", result="veto kept: rescue -0.0001 on cached val",
                          settled_by=n1)
        assert out.startswith("C1 settled")
        assert "Open choices" not in tree.status()

    def test_the_open_choices_of_the_parent_are_in_the_brief(self, run) -> None:
        tree, root, _, n1 = run
        tree.choice(node=n1, question="gold occurrence", options=["first", "random"], default="random",
                    why_open="untested", needs="short_training", effect="high")
        record_analysis(tree, n1)
        pid = new_plan(tree, parents=[n1])
        tree.new_node(kind="refine", title="r", hypothesis="h", parents=[n1], based_on=[n1], plan_id=pid)
        brief = (root / "nodes" / "n2" / "BRIEF.md").read_text()
        assert "Choices still open on the parents" in brief and "Keep each default" in brief
        assert "gold occurrence" in brief and "random (default)" in brief

    def test_the_tools_carry_choices_and_analyses(self, tmp_path) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        made = {t.name: t for t in create_research_tree_tools(root, require_plans=True)}
        assert {"tree_analysis", "tree_choice"} <= set(made)
        assert {"node", "question", "options", "default", "needs", "effect", "choice_id", "status", "result"} <= set(
            made["tree_choice"].args
        )


class TestKeepingTheGpuBusy:
    """r39: the GPU sat idle 19 min on average between nodes (64% busy vs 83-92% in r38) because each refine waited
    for its parent's analysis and no other plan was ready. A plan in reserve and a background builder fix that."""

    def test_the_hint_asks_for_a_reserve_while_a_node_runs_without_one(self, strict) -> None:
        tree, root, clock = strict
        tree.survey([{"name": "a", "mechanism": "m"}])
        tree.plan(text=PLAN)
        tree.plan(plan_id="P1", decision="accepted")
        tree.new_node(kind="explore", title="t", hypothesis="h", plan_id="P1")
        last = tree.status().splitlines()[-1]
        assert "n1 is being built" in last and "no accepted plan is in reserve" in last
        assert "does not depend on the running node" in last
        tree.plan(text=PLAN)
        tree.plan(plan_id="P2", decision="accepted")
        assert "no accepted plan is in reserve" not in tree.status().splitlines()[-1]

    def test_the_background_section_runs_the_builder_in_the_background_and_keeps_a_reserve(self) -> None:
        text = get_system_prompt(research_tree=True, research_tree_reports=True, research_tree_background=True)
        part = text.split("## Work that runs while you decide")[1]
        assert "Hand the brief to `code-agent` with `task_start`, not `task`" in part
        assert "Keep one accepted plan in reserve." in part and "does not depend on that node" in part
        assert "open the reserve plan if there is one" in part
        # the order reads: start, survey, overlap, builder, reserve, what waits
        order = ["The rule that keeps the run moving", "The survey comes first", "Plans and nodes overlap",
                 "Run the node's builder", "Keep one accepted plan in reserve", "What waits for what"]
        assert [part.index(o) for o in order] == sorted(part.index(o) for o in order)
        without = get_system_prompt(research_tree=True, research_tree_reports=True, research_tree_background=False)
        assert "Run the node's builder" not in without and "in reserve" not in without

    def test_a_correction_that_changes_a_setting_is_still_a_design_change(self) -> None:
        writer = " ".join(TestSubagents._load("plan-writer")["system_prompt"].split())
        review = " ".join(open(Path(__file__).parents[1] / "EvoScientist" / "subagents" / "plan-writer.yaml").read().split())
        assert "a loss weight or another training setting" in review
        assert "phrased as a correction of the plan's reasoning" in review
        assert "a setting it changes is still a design change" in review
