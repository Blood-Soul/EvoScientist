"""Cross-field search: same-field records never reach the gate, and only a written transfer path is kept.

The point of the path is the record nobody would have read -- the one from another field. That makes two failure modes
worth pinning: a similarity ranking that quietly returns the caller's own field, and a gate whose unparsable reply is
treated as permission.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from EvoScientist.memory.experiences.cross_domain import (
    bm25,
    record_view,
    search_cross_domain,
)
from EvoScientist.memory.experiences.render import render_cross_domain
from EvoScientist.memory.experiences.store import store_paper_experiences

PROJECT = "P-cross"


def l1(**over):
    base = {
        "domain": "optimisation", "task": "training a large model",
        "problem": "Most training windows hold no answer, so the model learns to predict no answer.",
        "method": ["null downsampling"],
        "requires": [{"text": "a per-window null label", "basis": "inferred"}],
        "source_setting": {"environment": [{"property": "null share", "value": "98%"}], "params": []},
        "limits_stated": [], "action": "Measure the null share, then keep a fraction of null windows.",
        "effect": {"compared": "20% vs 66% null", "benefit": "F1 87.3", "cost": None},
        "mechanism": {"text": "A null window teaches the model that empty is the safe answer.", "basis": "inferred"},
        "comparison": "controlled", "locator": "Table 1",
    }
    base.update(over)
    return base


def store(tmp_path, papers):
    for key, (domain, discipline, record) in papers.items():
        store_paper_experiences(
            memory_dir=tmp_path, project_id=PROJECT, paper_id=key, url=f"https://example.org/{key}",
            title=f"Paper {key}", paper_text=f"text {key}", prompts={"l1": "p"},
            payloads={"l1": {"experiences": [dict(record, domain=domain, discipline=discipline)]}},
        )


class ScriptedModel:
    """Restates the need on the abstract prompt; on the gate prompt keeps the candidates whose field is ecology."""

    def __init__(self, gate_reply: str | None = None, mappings: int = 2):
        self.gate_reply = gate_reply
        self.mappings = mappings
        self.seen: list[str] = []

    async def ainvoke(self, messages):
        system, human = messages[0].content, messages[1].content
        if "practice FROM ANOTHER FIELD" in system:
            return AIMessage(content=json.dumps({"goal": "How to keep a model from learning the empty answer",
                                                 "passages": ["[ecology] survey the empty patches first"]}))
        self.seen.append(human)
        if self.gate_reply is not None:
            return AIMessage(content=self.gate_reply)
        judgements = []
        for line in human.splitlines():
            if not line.startswith("["):
                continue
            index = int(line[1:line.index("]")])
            keep = "ecology" in line
            pairs = [{"source": "empty training window", "target": "empty survey patch",
                      "rationale": "both are the majority class the model is trained around"},
                     {"source": "null downsampling", "target": "stratified plot selection",
                      "rationale": "both decide how much of the majority class enters the sample"}][:self.mappings]
            judgements.append({
                "i": index, "transferable": keep,
                "object_mappings": pairs if keep else [],
                "shared_relations": [{"relation": "OUTNUMBER", "problem": "OUTNUMBER(empty window, filled window)",
                                      "solution": "OUTNUMBER(empty plot, occupied plot)"}] if keep else [],
                "path": "measure the empty share first, then fix the sampling fraction -> same measure before "
                        "deciding how much of the majority class to keep" if keep else "",
                "drop_reason": None if keep else "surface"})
        return AIMessage(content=json.dumps({"judgements": judgements}))


@pytest.mark.asyncio
async def test_same_field_records_never_reach_the_gate(tmp_path):
    store(tmp_path, {
        "p1": ("optimisation", "cs", l1()),
        "p2": ("ecology", "biology", l1(task="surveying sparse habitats",
                                        problem="Most survey plots are empty, so a model learns to predict empty.")),
    })
    model = ScriptedModel()
    result = await search_cross_domain(memory_dir=tmp_path, project_id=PROJECT, design_point="most windows are empty",
                                       model=model, exclude_domain="optimisation")
    assert [item["domain"] for item in result["kept"]] == ["ecology"]
    assert model.seen
    assert all("optimisation" not in line for line in model.seen[0].splitlines() if line.startswith("["))


@pytest.mark.asyncio
async def test_a_transfer_path_is_what_survives(tmp_path):
    store(tmp_path, {
        "p1": ("optimisation", "cs", l1()),
        "p2": ("ecology", "biology", l1(task="surveying sparse habitats")),
        "p3": ("hydrology", "earth", l1(task="flood forecasting")),
    })
    result = await search_cross_domain(memory_dir=tmp_path, project_id=PROJECT, design_point="most windows are empty",
                                       model=ScriptedModel(), exclude_domain="optimisation")
    assert len(result["kept"]) == 1
    kept = result["kept"][0]
    assert len(kept["mappings"]) == 2
    assert kept["relations"]
    assert kept["path"]
    assert result["dropped"].get("surface") == result["judged"] - 1
    text = render_cross_domain(result)
    assert "hypothesis, not evidence" in text
    assert "Object mappings (your problem -> the record):" in text
    assert "OUTNUMBER(empty plot, occupied plot)" in text
    assert "Transfer path:" in text


@pytest.mark.asyncio
async def test_an_unparsable_gate_keeps_nothing(tmp_path):
    store(tmp_path, {"p2": ("ecology", "biology", l1(task="surveying sparse habitats"))})
    result = await search_cross_domain(memory_dir=tmp_path, project_id=PROJECT, design_point="most windows are empty",
                                       model=ScriptedModel(gate_reply="I could not decide."), exclude_domain="optimisation")
    assert result["kept"] == []
    assert result["gate_failed"] == 1
    assert "No record passed the transfer check" in render_cross_domain(result)


def test_bm25_prefers_the_record_that_shares_the_need():
    documents = ["sparse survey plots empty habitat", "flood forecasting rainfall gauge"]
    scores = bm25(documents, "empty plots")
    assert scores[0] > scores[1] == 0.0


def test_record_view_reads_both_stored_shapes():
    current = {"problem": "windows are empty", "mechanism": {"text": "the model learns the safe answer"}}
    old = {"statement": "windows are empty", "mechanism": "the model learns the safe answer"}
    assert "learns the safe answer" in record_view(current)
    assert "learns the safe answer" in record_view(old)
    assert "mechanism" not in record_view(current)   # field names must not enter the ranking text


@pytest.mark.asyncio
async def test_a_transferable_verdict_without_a_mapping_is_not_shown(tmp_path):
    store(tmp_path, {"p2": ("ecology", "biology", l1(task="surveying sparse habitats"))})
    result = await search_cross_domain(memory_dir=tmp_path, project_id=PROJECT, design_point="most windows are empty",
                                       model=ScriptedModel(mappings=1), exclude_domain="optimisation")
    assert result["kept"] == []
    assert result["dropped"].get("no_mapping") == 1


class BlockModel(ScriptedModel):
    """The same scripted answers, wrapped the way Anthropic-style providers return them (thinking + text blocks)."""

    async def ainvoke(self, messages):
        reply = await super().ainvoke(messages)
        return AIMessage(content=[{"type": "thinking", "thinking": "judge the candidates"},
                                  {"type": "text", "text": reply.content}])


@pytest.mark.asyncio
async def test_a_gate_reply_in_content_blocks_is_parsed(tmp_path):
    store(tmp_path, {"p2": ("ecology", "biology", l1(task="surveying sparse habitats"))})
    result = await search_cross_domain(memory_dir=tmp_path, project_id=PROJECT, design_point="most windows are empty",
                                       model=BlockModel(), exclude_domain="optimisation")
    assert result["gate_failed"] == 0
    assert [item["domain"] for item in result["kept"]] == ["ecology"]


@pytest.mark.asyncio
async def test_the_current_situation_reaches_the_gate(tmp_path):
    store(tmp_path, {"p2": ("ecology", "biology", l1(task="surveying sparse habitats"))})
    model = ScriptedModel()
    await search_cross_domain(memory_dir=tmp_path, project_id=PROJECT, design_point="most windows are empty",
                              model=model, exclude_domain="optimisation",
                              context="one 24GB GPU, labels for 2% of windows, metric is F1 on answered questions")
    assert model.seen
    assert "F1 on answered questions" in model.seen[0]
