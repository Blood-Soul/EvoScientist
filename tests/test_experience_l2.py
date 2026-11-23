"""The second level after the switch to cluster induction: papers yield L1 only; L2 records (induced, or read from
a paper by an older extractor) are still stored, retrieved and rendered."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from EvoScientist.memory.experiences.extraction import (
    load_experience_prompts,
    parse_experience_v2,
    run_experience_extraction,
)
from EvoScientist.memory.experiences.render import render_hit, render_record
from EvoScientist.memory.experiences.store import (
    list_experience_documents,
    store_paper_experiences,
)

PAPER = "The verifier without head features gained 1.2 and 0.7 F1; with head features and initialization it gained 2.9 and 3.1. " * 5
QUOTE_L2 = "The verifier without head features gained 1.2 and 0.7 F1"


def l1_record(**over):
    base = {
        "domain": "qa", "task": "answer verification", "problem": "Heuristic confidence is unreliable.",
        "method": ["answer verifier", "confidence model"],
        "requires": [{"text": "a trained reader", "basis": "stated"}],
        "source_setting": {"environment": [{"property": "dataset", "value": "NQ"}], "params": []},
        "limits_stated": [], "action": "Train a second-stage verifier.",
        "effect": {"compared": "with vs without head features", "benefit": "2.9 and 3.1 F1", "cost": None},
        "comparison": "controlled", "locator": "Table 4",
    }
    base.update(over)
    return base


def l2_record(**over):
    """An L2 record as a library stores it (the shape both the old paper reader and cluster induction produce)."""
    base = {
        "domain": "qa", "task": "answer verification", "problem": "When is a second verifier better than a confidence score?",
        "method": ["answer verifier", "reflection net"],
        "action": "Feed the verifier the reader's head features and initialise from the reader; a text-only verifier gains little.",
        "compared_options": [
            {"option": "text-only verifier", "result": "+1.2 / +0.7 F1"},
            {"option": "head features + initialisation", "result": "+2.9 / +3.1 F1"},
        ],
        "mechanism": "The reader's logit margin is not comparable across questions.",
        "requires": [{"text": "the reader exposes top-layer logits", "basis": "stated"}],
        "limits_stated": [{"text": "a text-only verifier gains little", "quote": QUOTE_L2}],
        "origin": "paper", "locator": "Section 4.2", "source_l1_ids": [],
        "evidence": [{"supports": "claim", "quote": QUOTE_L2}], "layer": "l2",
    }
    base.update(over)
    return base


def reply(*records):
    return "```json\n" + json.dumps({"experiences": list(records)}) + "\n```"


def l2_payload(*records):
    return {"paper_id": "p1", "domain_arxiv": None, "experiences": list(records)}


class PromptModel:
    """Answers every call with one reply and records which prompts it saw."""

    def __init__(self, text):
        self.text, self.seen = text, []

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage

        self.seen.append(messages[0].content.splitlines()[0])
        return AIMessage(content=self.text)


@pytest.mark.asyncio
async def test_a_paper_yields_l1_only():
    assert set(load_experience_prompts()) == {"l1"}
    model = PromptModel(reply(l1_record()))
    out = await run_experience_extraction(paper_id="p1", paper_text=PAPER, prompts=load_experience_prompts(), model=model)
    assert set(out) == {"l1"} and len(out["l1"]["experiences"]) == 1
    assert model.seen == ["# Experience Extraction (v2: one level, comparison-centred)"]



def test_an_l2_record_renders_its_options_mechanism_and_conditions():
    text = render_record({"id": "E-1", "experience_level": "l2", "experience": l2_record()})
    assert "Options compared:" in text and "head features + initialisation" in text
    assert "Why it works:" in text and "Holds when" in text
    assert "Stops holding when (stated by the authors):" in text and "a text-only verifier gains little" in text
    # an older record's ungrounded limit (basis only, no quote, no source) is not shown
    old = l2_record(limits_stated=[{"text": "made up boundary", "basis": "inferred"}])
    assert "made up boundary" not in render_record({"id": "E-1", "experience_level": "l2", "experience": old})
    hit = render_hit(1, {"id": "E-1", "experience_level": "l2", "experience": l2_record()})
    assert "[L2]" in hit and "Claim:" in hit


def test_the_render_says_how_much_stands_behind_a_finding():
    paper = render_record({"id": "E-1", "experience_level": "l2", "experience": l2_record()})
    assert "Evidence: one paper's own sentences" in paper and QUOTE_L2 in paper
    old = l2_record()
    del old["evidence"]
    assert "no sentence of it was recorded" in render_record({"id": "E-2", "experience_level": "l2", "experience": old})


def test_plan_review_retrieval_keeps_a_share_of_l1(tmp_path: Path):
    from EvoScientist.memory.policy.select import gather_candidates

    # one paper with 3 L1 and 6 L2 records, all about the verifier: a pure L2-first fill would return no L1
    l1s = [l1_record(problem=f"verifier problem {i}", action=f"Train the answer verifier, variant {i}.") for i in range(3)]
    l2s = [l2_record(problem=f"When does the answer verifier help, case {i}?") for i in range(6)]
    store_paper_experiences(
        memory_dir=tmp_path, project_id="P-x", paper_id="p1", url="https://arxiv.org/abs/p1", title="T",
        paper_text=PAPER, prompts={"l1": "a", "l2": "b"},
        payloads={"l1": parse_experience_v2(reply(*l1s), paper_id="p1"), "l2": l2_payload(*l2s)},
    )
    got = gather_candidates(memory_dir=tmp_path, project_id="P-x", query="answer verifier", limit=6)
    levels = [c["level"] for c in got]
    assert len(got) == 6 and levels.count("l1") == 2 and levels.count("l2") == 4


def test_storing_resolves_positional_citations_to_the_stored_l1_ids(tmp_path: Path):
    """Libraries written by the in-paper induction cite `l1-N`; storing still resolves them to the stored ids."""
    lim = [{"text": "fails on long answers", "quote": "q"}]
    l1 = parse_experience_v2(reply(l1_record(limits_stated=lim), l1_record(limits_stated=lim, problem="Second.")),
                             paper_id="p1")
    induced = l2_record(origin="induced", locator="", source_l1_ids=["l1-1", "l1-2"], evidence=[],
                        limits_stated=[{"text": "stops helping on long answers", "from": ["l1-1", "l1-2"]}])
    store_paper_experiences(
        memory_dir=tmp_path, project_id="P-x", paper_id="p1", url="https://arxiv.org/abs/p1", title="T",
        paper_text=PAPER, prompts={"l1": "a", "l2": "b"}, payloads={"l1": l1, "l2": l2_payload(induced)},
    )
    docs = list_experience_documents(memory_dir=tmp_path, project_id="P-x")
    l1_ids = [d.observation_id for d in docs if d.experience_level == "l1"]
    stored = json.loads((tmp_path / "experiences" / "projects" / "P-x" / docs[0].paper_key / "l2.json").read_text())
    assert stored["experiences"][0]["source_l1_ids"] == l1_ids[:2]
    assert stored["experiences"][0]["limits_stated"][0]["from"] == l1_ids[:2]
