"""Experience records reach the agent as labelled text that explains its own fields, never as the stored JSON."""

from __future__ import annotations

import json
from pathlib import Path

from EvoScientist.memory._atomic import atomic_write_json
from EvoScientist.memory.experiences.render import render_browse, render_record, render_search
from EvoScientist.memory.experiences.store import _paper_dir, list_experience_documents

PROJECT = "P-render"

CURRENT = {
    "domain": "question answering",
    "task": "training instance construction",
    "problem": "Most training windows hold no answer.",
    "method": ["null downsampling", "negative subsampling", "balancing null instances"],
    "requires": [
        {"text": "a per-window null label", "basis": "inferred"},
        {"text": "windowed documents", "basis": "stated"},
    ],
    "source_setting": {
        "environment": [{"property": "null share", "value": "98%"}],
        "params": [
            {"name": "downsampling factor", "value": "50", "basis": "derived", "from": "98% null rate"},
            {"name": "share of null", "value": "33%", "basis": "swept", "from": "20%, 33%, 66%"},
            {"name": "batch size", "value": "32", "basis": "unstated", "from": ""},
        ],
    },
    "limits_stated": [{"text": "hurts answerable questions at 66%", "quote": "q"}],
    "action": "Measure the null share, then keep a fraction of null windows.",
    "effect": {"compared": "20% vs 33% vs 66% null", "benefit": "F1 87.3 at 66%", "cost": "answerable F1 falls"},
    "comparison": "controlled",
    "locator": "Table 1",
}

LEGACY = {
    "domain": "qa",
    "task": "windowing",
    "statement": "A long paragraph about windows.",
    "action": "Split documents.",
    "effect": "F1 rose.",
    "applicable_when": ["long documents"],
    "not_applicable_when": ["short documents"],
    "evidence": [{"section": "method", "quote": "we split"}],
    "evidence_scope": {"has_ablation": False, "verification_strength": "single-setting"},
    "bindings": [{"name": "BERT", "kind": "model"}],
}


def _doc(item, *, level="l1", record_id="E-abc"):
    return {"id": record_id, "record_kind": "experience", "experience_level": level,
            "paper": {"paper_id": "1901.08634", "title": "A BERT Baseline", "full_text_available": True},
            "experience": item}


def test_current_record_is_text_with_each_field_explained():
    out = render_record(_doc(CURRENT))
    assert not out.lstrip().startswith("{") and '"experience"' not in out
    assert "controlled comparison: one factor was changed" in out
    assert "Method: null downsampling; also called: negative subsampling, balancing null instances" in out
    assert "a per-window null label (inferred)" in out and "windowed documents (inferred)" not in out
    assert "downsampling factor = 50  [derived from: 98% null rate]" in out
    assert "share of null = 33%  [swept, tried: 20%, 33%, 66%]" in out
    assert "batch size = 32" in out and "batch size = 32  [" not in out
    assert "Limits the authors themselves state" in out
    assert "see Table 1" in out and "search_paper_text" in out


def test_no_cost_line_when_there_is_none():
    item = {**CURRENT, "effect": {**CURRENT["effect"], "cost": None}}
    assert "Cost:" not in render_record(_doc(item))


def test_each_comparison_kind_says_what_it_means():
    for kind, phrase in (("vs_baseline", "other methods"), ("absolute_only", "nothing was compared")):
        assert phrase in render_record(_doc({**CURRENT, "comparison": kind}))


def test_older_records_are_still_readable_and_say_who_judged_the_conditions():
    out = render_record(_doc(LEGACY))
    assert "A long paragraph about windows." in out and "bindings" not in out
    assert "the extractor's judgement, not the paper's" in out
    assert "single-setting (ablation: no)" in out
    assert not out.lstrip().startswith("{")


def test_a_json_string_is_accepted_and_garbage_passes_through():
    assert render_record(json.dumps(_doc(CURRENT))).startswith("E-abc")
    assert render_record("not json") == "not json"


def test_search_results_are_a_ranked_text_list():
    text = render_search([(_doc(CURRENT), {"observation_id": "E-abc"}), (None, {"observation_id": "E-zzz", "summary": "gone"})])
    assert text.startswith("2 experience record(s), best match first:")
    assert "1. E-abc [controlled]" in text and "2. E-zzz gone" in text
    assert "read_memory" in text
    assert render_search([]).startswith("No experience matched.")


def test_browse_is_text():
    out = render_browse({"facet": "domain", "total_records": 9, "total_values": 3, "offset": 0,
                         "values": [{"value": "qa", "records": 6}, {"value": "cv", "records": 3}]})
    assert out.startswith("Experience library: 9 records. By domain: qa (6), cv (3).")


def _library(tmp_path: Path, items: list[dict]) -> Path:
    directory = _paper_dir(tmp_path, project_id=PROJECT, paper_id="1901.08634", url="https://arxiv.org/abs/1901.08634")
    atomic_write_json(directory / "l1.json", {"paper_id": "1901.08634", "experiences": items})
    atomic_write_json(directory / "metadata.json", {"project_id": PROJECT, "paper_id": "1901.08634", "title": "T", "url": "u"})
    return tmp_path


def test_records_of_one_paper_with_the_same_task_get_different_ids(tmp_path):
    other = {**CURRENT, "problem": "Another difficulty.", "action": "Do another thing."}
    docs = list_experience_documents(memory_dir=_library(tmp_path, [CURRENT, other]), project_id=PROJECT)
    assert len({d.observation_id for d in docs}) == 2


def test_the_search_summary_carries_the_method_aliases(tmp_path):
    docs = list_experience_documents(memory_dir=_library(tmp_path, [CURRENT]), project_id=PROJECT)
    assert "negative subsampling" in docs[0].summary and "Most training windows" in docs[0].summary


def test_the_tools_return_text_end_to_end(tmp_path):
    from EvoScientist.memory.observations.tools import create_read_memory_tool
    from EvoScientist.tools.experience_search import create_list_experience_tool, create_search_experience_tool

    memory = _library(tmp_path, [CURRENT])
    found = create_search_experience_tool(memory_dir=memory, project_id=PROJECT).func(method="null downsampling")
    assert "[controlled]" in found and not found.lstrip().startswith("{")
    record_id = found.split("1. ")[1].split()[0]
    full = create_read_memory_tool(memory_dir=memory, project_id=PROJECT).func(observation_id=record_id)
    assert "What to do:" in full and not full.lstrip().startswith("{")
    assert "Experience library: 1 records" in create_list_experience_tool(memory_dir=memory, project_id=PROJECT).func(facet="domain")
