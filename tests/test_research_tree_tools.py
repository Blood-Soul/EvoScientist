"""Research-tree tools: refusals come back as text, wiring follows the config flag."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from EvoScientist.config.settings import _ENV_MAPPINGS, EvoScientistConfig
from EvoScientist.paths import Workspace
from EvoScientist.prompts import RESEARCH_TREE, get_system_prompt
from EvoScientist.research_tree.tools import create_research_tree_tools
from tests.test_research_tree import Clock, FakeJudge, accuracy_rows

NAMES = [
    "tree_init",
    "tree_survey",
    "tree_status",
    "tree_history",
    "tree_new",
    "tree_finish",
    "tree_insight",
    "tree_finalize",
]


@pytest.fixture
def tools(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    clock = Clock()
    made = {
        t.name: t
        for t in create_research_tree_tools(root, judge=FakeJudge(), now=clock)
    }
    return made, root, clock


def test_tool_set_and_schemas():
    made = create_research_tree_tools(Path("."))
    assert [t.name for t in made] == NAMES
    new = next(t for t in made if t.name == "tree_new")
    assert {
        "kind",
        "title",
        "hypothesis",
        "based_on",
        "parents",
        "minutes",
        "plan",
        "cluster",
        "component",
        "reuse",
        "avoid",
        "stop_condition",
    } <= set(new.args)


def test_refusals_are_text_not_exceptions(tools):
    made, _root, _ = tools
    assert (
        made["tree_status"]
        .invoke({})
        .startswith("ERROR: there is no research tree yet")
    )
    assert made["tree_init"].invoke({}).startswith("ERROR: budget_minutes")
    assert (
        made["tree_init"]
        .invoke({"budget_minutes": 300})
        .startswith("Research tree initialised")
    )
    out = made["tree_new"].invoke(
        {"kind": "bogus", "title": "t", "hypothesis": "h", "plan": "p"}
    )
    assert out.startswith("ERROR: kind must be one of")


def test_a_node_end_to_end_through_the_tools(tools):
    made, root, clock = tools
    made["tree_init"].invoke({"budget_minutes": 300})
    created = made["tree_new"].invoke(
        {
            "kind": "explore",
            "title": "baseline",
            "hypothesis": "sets a floor",
            "plan": "plain fine-tune",
        }
    )
    nid = created.split()[1]
    (root / "nodes" / nid / "holdout_predictions.csv").write_text(accuracy_rows(240))
    (root / "nodes" / nid / "submission.csv").write_text("deliverable")
    clock.advance(25)
    closed = made["tree_finish"].invoke(
        {"node": nid, "change": "plain", "observation": "fine"}
    )
    assert "score 0.6000" in closed
    assert "new champion" in closed
    assert "hypothesis: sets a floor" in made["tree_history"].invoke({})
    assert (
        made["tree_insight"]
        .invoke({"text": "plain stalls", "evidence": [nid]})
        .startswith("I1 recorded")
    )
    assert "### n1" in made["tree_finalize"].invoke({})
    assert (root / "TRAJECTORY.md").is_file()


def test_config_flag_defaults_off_and_maps_to_env():
    assert EvoScientistConfig().research_tree_enabled is False
    assert (
        _ENV_MAPPINGS["research_tree_enabled"] == "EVOSCIENTIST_RESEARCH_TREE_ENABLED"
    )


def test_prompt_section_only_when_enabled():
    assert RESEARCH_TREE.strip() not in get_system_prompt()
    assert RESEARCH_TREE.strip() in get_system_prompt(research_tree=True)
    for name in (
        "tree_init",
        "tree_survey",
        "tree_status",
        "tree_history",
        "tree_new",
        "tree_finish",
        "tree_insight",
        "tree_finalize",
    ):
        assert name in RESEARCH_TREE


def test_tools_registered_only_when_enabled(tmp_path, monkeypatch):
    from EvoScientist import EvoScientist as agent_module

    monkeypatch.setattr(
        agent_module,
        "_paths_mod",
        SimpleNamespace(MEMORIES_DIR=tmp_path / "mem"),
        raising=False,
    )
    ws = Workspace(root=tmp_path / "proj")
    (tmp_path / "proj").mkdir()
    reg_off, _base_off = agent_module._build_paper_tools(
        cfg=SimpleNamespace(research_tree_enabled=False),
        workspace=ws,
        work_dir=tmp_path / "proj",
    )
    reg_on, base_on = agent_module._build_paper_tools(
        cfg=SimpleNamespace(research_tree_enabled=True),
        workspace=ws,
        work_dir=tmp_path / "proj",
    )
    assert not any(n.startswith("tree_") for n in reg_off)
    assert all(n in reg_on for n in NAMES)
    assert [t.name for t in base_on if t.name.startswith("tree_")] == NAMES


def test_module_cli_finalizes_after_the_agent_stopped(tmp_path, capsys):
    from EvoScientist.research_tree.__main__ import main
    from EvoScientist.research_tree.tree import ResearchTree

    root = tmp_path / "ws"
    root.mkdir()
    clock = Clock()
    tree = ResearchTree(root, judge=FakeJudge(), now=clock)
    tree.init(budget_minutes=300)
    nid = tree.new_node(kind="explore", title="a", hypothesis="h").split()[
        1
    ]  # left running
    assert main(["status", "--root", str(root)]) == 0
    assert "Running: " + nid in capsys.readouterr().out
    assert main(["finalize", "--root", str(root)]) == 0
    assert "abandoned" in capsys.readouterr().out or (root / "TRAJECTORY.md").is_file()
    assert main(["status", "--root", str(tmp_path)]) == 1


def test_tree_tools_survive_tool_selection():
    from EvoScientist.middleware.tool_selector import DEFAULT_ALWAYS_INCLUDE_TOOLS

    assert set(NAMES) <= DEFAULT_ALWAYS_INCLUDE_TOOLS


def test_survey_then_a_node_in_a_family_through_the_tools(tools):
    made, _, _ = tools
    made["tree_init"].invoke({"budget_minutes": 300})
    out = made["tree_survey"].invoke(
        {
            "clusters": [
                {
                    "name": "abstain",
                    "mechanism": "calibrate a no-answer threshold",
                    "sources": ["2101.1 P"],
                }
            ]
        }
    )
    assert out.startswith("Survey recorded")
    assert "abstain" in out
    refused = made["tree_new"].invoke(
        {
            "kind": "explore",
            "title": "t",
            "hypothesis": "h",
            "plan": "p",
            "cluster": "other",
        }
    )
    assert refused.startswith("ERROR: cluster 'other' is not a surveyed family")
    created = made["tree_new"].invoke(
        {
            "kind": "explore",
            "title": "baseline",
            "hypothesis": "floor",
            "plan": "plain",
            "cluster": "abstain",
        }
    )
    assert "## Brief for the sub-agent" in created
    assert "family: abstain" in created
    assert "tried in: n1" in made["tree_status"].invoke({})


def test_every_tool_call_is_journaled_including_refusals(tools):
    import json

    made, root, _clock = tools
    made["tree_status"].invoke({})  # refused: no tree yet
    made["tree_init"].invoke({"budget_minutes": 300})
    made["tree_new"].invoke(
        {"kind": "bogus", "title": "t", "hypothesis": "h", "plan": "p"}
    )
    made["tree_survey"].invoke(
        {"clusters": [{"name": "abstain", "mechanism": "calibrate a threshold"}]}
    )
    calls = [
        json.loads(x)
        for x in (root / ".research" / "calls.jsonl").read_text().splitlines()
    ]
    assert [c["tool"] for c in calls] == [
        "tree_status",
        "tree_init",
        "tree_new",
        "tree_survey",
    ]
    assert [c["ok"] for c in calls] == [False, True, False, True]
    assert calls[2]["args"]["kind"] == "bogus"
    assert calls[2]["result"].startswith("ERROR: kind must be one of")
    assert calls[3]["args"]["clusters"][0]["name"] == "abstain"
    assert all(isinstance(c["ts"], float) and c["seconds"] >= 0 for c in calls)
