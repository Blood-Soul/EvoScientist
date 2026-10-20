"""The optional graph preserves evidence, boundaries and project isolation."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from EvoScientist.memory.experiences import graph as graph_module
from EvoScientist.memory.experiences.graph import (
    _validate_decision,
    build_experience_graph,
    expand_graph,
    experience_graph_path,
    graph_is_current,
)
from EvoScientist.memory.experiences.retrieval import related_experience_records
from EvoScientist.memory.experiences.store import store_paper_experiences
from EvoScientist.tools.experience_search import create_search_experience_tool


def _document(record_id, paper, level, original, action, sources=None):
    item = {
        "id": original,
        "task": "forecasting",
        "action": action,
        "source_l1_ids": sources or [],
    }
    payload = {
        "id": record_id,
        "experience_level": level,
        "paper": {"paper_id": paper, "paper_key": paper},
        "experience": item,
    }
    return SimpleNamespace(
        observation_id=record_id,
        text=json.dumps(payload),
        path=f"/{paper}/{level}.json",
    )


def _relation(a, b, kind="complements"):
    return {
        "relation": kind,
        "from": a["id"],
        "to": b["id"],
        "claim": "forecasting decision",
        "reason": "compare the two actions",
        "conditions": ["both settings are relevant"],
        "limitations": [],
        "unresolved": ["target fit not checked"],
        "evidence": [
            {"id": a["id"], "field": "action", "quote": a["content"]["action"]},
            {"id": b["id"], "field": "action", "quote": b["content"]["action"]},
        ],
    }


class _LinkModel:
    def __init__(self, invalid=False):
        self.calls = 0
        self.invalid = invalid

    async def ainvoke(self, messages):
        self.calls += 1
        a, b = json.loads(messages[-1].content)
        relation = _relation(a, b)
        if self.invalid:
            relation["evidence"][0]["quote"] = "not in source"
        return AIMessage(
            content=json.dumps({"decision": "related", "relations": [relation]})
        )


def _mock_documents(monkeypatch, docs):
    monkeypatch.setattr(graph_module, "list_experience_documents", lambda **kw: docs)


@pytest.mark.asyncio
async def test_all_level_pairs_source_edges_and_cache(monkeypatch):
    docs = [
        _document("E-a1", "a", "l1", "a1", "assess trend validation"),
        _document("E-a2", "a", "l2", "a2", "select trend validation", ["a1"]),
        _document("E-b1", "b", "l1", "b1", "validate trend validation"),
        _document("E-b2", "b", "l2", "b2", "trend validation forecasts"),
    ]
    _mock_documents(monkeypatch, docs)
    model = _LinkModel()
    args = {
        "memory_dir": ".",
        "project_id": "p",
        "prompt": "rules",
        "model": model,
        "max_pairs": 10,
    }
    first = await build_experience_graph(**args)
    assert model.calls == 6
    uncapped = await build_experience_graph(
        **{**args, "model": None, "max_pairs": None}
    )
    assert len(uncapped["decisions"]) == 6
    assert {row["group"] for row in first["decisions"]} == {
        "same-paper:l1-l2",
        "cross-paper:l1-l1",
        "cross-paper:l1-l2",
        "cross-paper:l2-l2",
    }
    derived = [edge for edge in first["edges"] if edge["relation"] == "derived_from"]
    assert [(e["from"], e["to"]) for e in derived] == [("E-a2", "E-a1")]
    assert all(
        edge["status"] == "unreviewed"
        for edge in first["edges"]
        if edge["relation"] != "derived_from"
    )
    assert (
        next(e for e in first["edges"] if e["relation"] == "complements")["provenance"][
            "source_check"
        ]
        == "experience_record_only"
    )
    second = await build_experience_graph(**args, previous=first)
    assert model.calls == 6
    assert second["history"] == []
    docs[-1] = _document("E-b2", "b", "l2", "b2", "changed trend validation")
    third = await build_experience_graph(**args, previous=second)
    assert model.calls == 9
    assert len(third["history"]) == 3
    assert not graph_is_current(first, memory_dir=".")


@pytest.mark.asyncio
async def test_bad_quote_does_not_create_a_link(monkeypatch):
    docs = [
        _document("E-a", "a", "l2", "a", "choose trend validation"),
        _document("E-b", "b", "l2", "b", "validate trend validation"),
    ]
    _mock_documents(monkeypatch, docs)
    graph = await build_experience_graph(
        memory_dir=".", project_id="p", prompt="rules", model=_LinkModel(invalid=True)
    )
    assert graph["edges"] == []
    assert "quote not found" in graph["decisions"][0]["error"]


@pytest.mark.asyncio
async def test_ambiguous_source_is_not_silently_linked(monkeypatch):
    docs = [
        _document("E-a1", "a", "l1", "same", "one"),
        _document("E-a2", "a", "l1", "same", "two"),
        _document("E-a3", "a", "l2", "three", "three", ["same"]),
    ]
    _mock_documents(monkeypatch, docs)
    graph = await build_experience_graph(memory_dir=".", project_id="p", prompt="rules")
    assert not graph["edges"]
    assert "unresolved L1 source" in graph["warnings"][0]


@pytest.mark.asyncio
async def test_legacy_judgments_are_archived_not_reused(monkeypatch):
    docs = [
        _document("E-a", "a", "l2", "a", "choose model"),
        _document("E-b", "b", "l2", "b", "validate model"),
    ]
    _mock_documents(monkeypatch, docs)
    old_row = {"ids": ["E-a", "E-b"], "result": {"relation": "complements"}}
    old = {"version": 1, "project_id": "p", "decisions": [old_row]}
    graph = await build_experience_graph(
        memory_dir=".", project_id="p", prompt="rules", previous=old
    )
    assert graph["history"][0]["decision"] == old_row
    assert graph["history"][0]["graph_version"] == 1
    assert graph["edges"] == []
    assert "must be reclassified" in graph["warnings"][0]
    with pytest.raises(ValueError, match="another project"):
        await build_experience_graph(
            memory_dir=".", project_id="other", prompt="rules", previous=old
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["none", "uncertain"])
async def test_negative_judgments_reused_even_after_pair_leaves_active_set(
    monkeypatch, decision
):
    docs = [
        _document(f"E-{key}", key, "l2", key, "interval calibration reserve sizing")
        for key in "abc"
    ]
    _mock_documents(monkeypatch, docs)

    class Model:
        calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            return AIMessage(
                content=json.dumps(
                    {"decision": decision, "reason": "insufficient concrete link"}
                )
            )

    model = Model()
    args = {"memory_dir": ".", "project_id": "p", "prompt": "rules", "model": model}
    first = await build_experience_graph(**args)
    assert model.calls == 3
    limited = await build_experience_graph(**args, previous=first, max_pairs=1)
    assert model.calls == 3
    restored = await build_experience_graph(**args, previous=limited)
    assert model.calls == 3
    assert len(restored["decisions"]) == 3
    assert restored["candidate_stats"]["judgments_reused"] == 3
    assert all(row["result"]["decision"] == decision for row in restored["decisions"])


def test_multiple_relations_and_uncertain_are_distinct():
    a = {"id": "E-a", "content": {"action": "select trend"}}
    b = {"id": "E-b", "content": {"action": "test trend"}}
    raw = {
        "decision": "related",
        "relations": [_relation(a, b), _relation(a, b, "qualifies")],
    }
    assert len(_validate_decision(json.dumps(raw), a, b)["relations"]) == 2
    assert (
        _validate_decision('{"decision":"uncertain","reason":"missing scope"}', a, b)[
            "decision"
        ]
        == "uncertain"
    )
    raw["relations"].append(_relation(b, a))
    with pytest.raises(ValueError, match="duplicate"):
        _validate_decision(json.dumps(raw), a, b)
    raw["relations"] = [_relation(a, b)]
    raw["relations"][0]["evidence"][0]["field"] = "missing"
    with pytest.raises(ValueError, match="invalid evidence"):
        _validate_decision(json.dumps(raw), a, b)


def _store(root, paper, project="P-a", action="forecast using trend"):
    common = {
        "task": "forecast trend",
        "domain": "forecast",
        "action": action,
        "statement": action,
        "applicable_when": ["short series"],
    }
    store_paper_experiences(
        memory_dir=root,
        project_id=project,
        paper_id=paper,
        url=f"https://example.com/{paper}",
        title=paper,
        paper_text="paper text",
        prompts={"l1": "l1", "l2": "l2"},
        payloads={
            "l1": {"experiences": [{**common, "id": "l1"}]},
            "l2": {"experiences": [{**common, "id": "l2", "source_l1_ids": ["l1"]}]},
        },
    )


@pytest.mark.asyncio
async def test_field_strategy_build_cache_and_real_search_tool(tmp_path):
    _store(tmp_path, "a", action="calibrated predictive intervals")
    _store(tmp_path, "b", action="calibrated predictive intervals")
    model = _LinkModel()
    args = {
        "memory_dir": tmp_path,
        "project_id": "P-a",
        "prompt": "rules",
        "candidate_strategy": "field",
        "model": model,
    }
    graph = await build_experience_graph(**args)
    assert graph["candidate_strategy"] == "field"
    assert all(
        record["content"]["domain"] == "forecast"
        for record in graph["records"].values()
    )
    assert graph["candidate_evidence"]
    assert all(row["routes"] for row in graph["candidate_evidence"])
    calls = model.calls
    assert calls > 0
    repeated = await build_experience_graph(
        **args, previous=json.loads(json.dumps(graph))
    )
    assert model.calls == calls
    assert repeated["candidate_stats"]["sources_searched"] == 0
    assert repeated["candidate_stats"]["judgments_reused"] == calls
    path = experience_graph_path(tmp_path, "P-a")
    path.write_text(json.dumps(repeated), encoding="utf-8")
    tool = create_search_experience_tool(memory_dir=tmp_path, project_id="P-a")
    result = json.loads(
        tool.invoke(
            {
                "topic": "calibrated intervals",
                "limit": 1,
                "include_related": True,
                "related_types": ["complements"],
            }
        )
    )
    assert result["graph"]["related"]


@pytest.mark.asyncio
async def test_opt_in_tool_expansion_facets_and_stale_graph(tmp_path):
    _store(tmp_path, "a")
    _store(tmp_path, "b")
    graph = await build_experience_graph(
        memory_dir=tmp_path,
        project_id="P-a",
        prompt="rules",
        model=_LinkModel(),
        max_pairs=8,
    )
    path = experience_graph_path(tmp_path, "P-a")
    path.write_text(json.dumps(graph), encoding="utf-8")
    tool = create_search_experience_tool(memory_dir=tmp_path, project_id="P-a")
    default = json.loads(tool.invoke({"topic": "trend", "limit": 1}))
    assert "graph" not in default
    expanded = json.loads(
        tool.invoke(
            {
                "topic": "trend",
                "limit": 1,
                "include_related": True,
                "related_types": ["derived_from"],
                "related_limit": 2,
            }
        )
    )
    assert expanded["graph"]["related"]
    assert all(
        e["relation"] == "derived_from"
        for r in expanded["graph"]["related"]
        for e in r["edges"]
    )
    only_l2 = related_experience_records(
        memory_dir=tmp_path,
        project_id="P-a",
        ids=[next(iter(graph["records"]))],
        level="l2",
    )
    assert all(row["record"]["level"] == "l2" for row in only_l2["related"])
    assert expand_graph(graph, [next(iter(graph["records"]))], limit=0) == []
    _store(tmp_path, "c")
    stale = json.loads(tool.invoke({"topic": "trend", "include_related": True}))
    assert stale["graph"]["related"] == []
    assert "stale" in stale["graph"]["warning"]
    assert (
        related_experience_records(memory_dir=tmp_path, project_id="P-other", ids=[])[
            "related"
        ]
        == []
    )
