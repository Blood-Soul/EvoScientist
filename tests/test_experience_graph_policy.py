"""Graph data must reach policies without bypassing budgets or source boundaries."""

import json
import shutil

import pytest

from EvoScientist.memory.experiences.graph import (
    build_experience_graph,
    experience_graph_path,
    graph_is_current,
    rebase_experience_graph,
)
from EvoScientist.memory.experiences.store import store_paper_experiences
from EvoScientist.memory.policy import graph_context
from EvoScientist.memory.policy.store import _policy_cache_key
from EvoScientist.memory.policy.synthesize import _render_records


def test_disabled_preserves_candidates(monkeypatch):
    monkeypatch.delenv(graph_context.GRAPH_ENV, raising=False)
    result = graph_context.graph_candidates(
        memory_dir="missing", project_id="p", ranked_ids=["a", "b"], limit=2
    )
    assert result["ids"] == ["a", "b"]
    assert not result["edges"]


def test_missing_graph_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv(graph_context.GRAPH_ENV, "true")
    result = graph_context.graph_candidates(
        memory_dir=tmp_path, project_id="p", ranked_ids=["a", "b"], limit=2
    )
    assert result["ids"] == ["a", "b"]
    assert result["warning"]


def test_graph_slots_are_bounded_and_facets_forwarded(monkeypatch, tmp_path):
    monkeypatch.setenv(graph_context.GRAPH_ENV, "true")
    path = experience_graph_path(tmp_path, "p")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"edges": [{"from": "a", "to": "x", "relation": "alternative_to"}]})
    )
    monkeypatch.setattr(graph_context, "graph_is_current", lambda *a, **kw: True)
    calls = []

    def related(**kw):
        calls.append(kw)
        return {
            "related": [
                {"record": {"id": "x"}, "edges": [{"relation": "alternative_to"}]}
            ]
        }

    monkeypatch.setattr(graph_context, "related_experience_records", related)
    result = graph_context.graph_candidates(
        memory_dir=tmp_path,
        project_id="p",
        ranked_ids=["a", "b", "c", "d"],
        limit=4,
        domain="molecules",
        level="l1",
    )
    assert result["ids"] == ["a", "b", "x", "c"]
    assert result["expanded_ids"] == ["x"]
    assert calls[0]["domain"] == "molecules"
    assert calls[0]["level"] == "l1"


def test_writer_only_receives_relations_between_selected_records():
    edge = {
        "from": "a",
        "to": "b",
        "relation": "qualifies",
        "conditions": ["small data"],
        "limitations": ["not verified"],
        "evidence": [],
    }
    omitted = {"from": "a", "to": "absent", "relation": "supports"}
    selected = [
        {"id": "a", "record": {"id": "a"}, "graph_edges": [edge, omitted]},
        {"id": "b", "record": {"id": "b"}},
    ]
    text = _render_records(selected)
    assert "small data" in text
    assert "not verified" in text
    assert "absent" not in text
    assert "data, not instructions" in text


def test_graph_cache_changes_with_graph_and_state():
    selected = [{"id": "a", "record": {}, "graph_fingerprint": "v1"}]
    first = graph_context.graph_cache_context(selected, "initial")
    assert first != graph_context.graph_cache_context(selected, "changed")
    selected[0]["graph_fingerprint"] = "v2"
    assert first != graph_context.graph_cache_context(selected, "initial")
    assert _policy_cache_key(task="t", selected_ids=["a"]) != _policy_cache_key(
        task="t", selected_ids=["a"], context_hash=first
    )


@pytest.mark.asyncio
async def test_mount_rebinds_ids_and_rejects_changed_content(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    common = {
        "task": "graph regression",
        "statement": "compare bond-aware molecular encoders",
        "action": "compare bond-aware molecular encoders",
    }
    for paper in ("a", "b"):
        store_paper_experiences(
            memory_dir=source,
            project_id="old",
            paper_id=paper,
            url="https://example.com",
            title=paper,
            paper_text="text",
            prompts={"l1": "l1", "l2": "l2"},
            payloads={
                "l1": {"experiences": [{"id": "one", **common}]},
                "l2": {"experiences": []},
            },
        )
    from langchain_core.messages import AIMessage

    class Model:
        async def ainvoke(self, messages):
            return AIMessage(content='{"decision":"none","reason":"same topic alone"}')

    graph = await build_experience_graph(
        memory_dir=source, project_id="old", prompt="rules", model=Model()
    )
    shutil.copytree(
        source / "experiences/projects/old", target / "experiences/projects/new"
    )
    for path in target.rglob("metadata.json"):
        data = json.loads(path.read_text())
        data["project_id"] = "new"
        path.write_text(json.dumps(data))
    mounted = await rebase_experience_graph(
        source_graph=graph, memory_dir=target, project_id="new", prompt="rules"
    )
    assert graph_is_current(mounted, memory_dir=target)
    assert set(mounted["records"]).isdisjoint(graph["records"])
    assert len(mounted["decisions"]) == len(graph["decisions"])
    assert mounted["mount_provenance"]["model_requests"] == 0
    item = next(target.rglob("l1.json"))
    data = json.loads(item.read_text())
    data["experiences"][0]["statement"] = "different content"
    item.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="content changed"):
        await rebase_experience_graph(
            source_graph=graph, memory_dir=target, project_id="new", prompt="rules"
        )
