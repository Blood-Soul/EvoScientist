"""The second level: mechanism-level findings read from a paper, and induction over the paper's own L1 records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from EvoScientist.memory.experiences.extraction import (
    ExperienceOutputError,
    _merge_induced,
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
        "limits_stated": [{"text": "the reader's [CLS] is trained for answer type, not for wrongness", "basis": "stated"}],
        "origin": "paper", "locator": "Section 4.2", "source_l1_ids": [],
    }
    base.update(over)
    return base


def reply(*records):
    return "```json\n" + json.dumps({"experiences": list(records)}) + "\n```"


def test_a_paper_finding_is_parsed_and_marked_as_l2():
    out = parse_experience_v2(reply(l2_record()), paper_id="p1", paper_text=PAPER, level="l2")
    record = out["experiences"][0]
    assert record["layer"] == "l2" and record["origin"] == "paper"
    assert len(record["compared_options"]) == 2


def test_a_paper_finding_without_a_locator_is_dropped():
    with pytest.raises(ExperienceOutputError, match="locator"):
        parse_experience_v2(reply(l2_record(locator="")), paper_id="p1", level="l2")


def test_an_induced_finding_can_only_cite_the_records_it_was_given():
    induced = l2_record(origin="induced", locator="", source_l1_ids=["l1-2"])
    out = parse_experience_v2(reply(induced), paper_id="p1", level="l2", source_ids={"l1-1", "l1-2"})
    assert out["experiences"][0]["source_l1_ids"] == ["l1-2"]
    with pytest.raises(ExperienceOutputError, match="not given"):
        parse_experience_v2(reply(l2_record(origin="induced", locator="", source_l1_ids=["l1-9"])),
                            paper_id="p1", level="l2", source_ids={"l1-1"})


def test_an_induced_finding_must_cite_something():
    with pytest.raises(ExperienceOutputError, match="source_l1_ids"):
        parse_experience_v2(reply(l2_record(origin="induced", locator="", source_l1_ids=[])), paper_id="p1", level="l2")


def test_an_induced_record_loses_a_locator_it_should_not_have():
    out = parse_experience_v2(
        reply(l2_record(origin="induced", source_l1_ids=["l1-1"])), paper_id="p1", level="l2", source_ids={"l1-1"}
    )
    assert out["experiences"][0]["locator"] == ""


def test_a_finding_whose_numbers_the_paper_never_writes_is_dropped():
    invented = l2_record(compared_options=[{"option": "x", "result": "91.11 vs 92.22 vs 93.33 vs 94.44"}])
    out = parse_experience_v2(reply(l2_record(), invented), paper_id="p1", paper_text=PAPER, level="l2")
    assert len(out["experiences"]) == 1


def test_an_induced_record_is_dropped_when_it_restates_a_direct_one():
    direct = l2_record()
    same = l2_record(origin="induced", locator="", source_l1_ids=["l1-1"])
    assert _merge_induced([direct], [same]) == [direct]
    other = l2_record(
        origin="induced", locator="", source_l1_ids=["l1-1"], problem="A different question.",
        compared_options=[{"option": "frozen encoder", "result": "saves a pass"}],
    )
    assert len(_merge_induced([direct], [other])) == 2


def test_an_induced_record_comparing_the_same_options_in_other_words_is_dropped():
    direct = l2_record()
    reworded = l2_record(
        origin="induced", locator="", source_l1_ids=["l1-1"],
        problem="Which design makes a verifier work?",
        action="Use the reader's head features and start from the reader.",
    )
    assert _merge_induced([direct], [reworded]) == [direct]


class LevelModel:
    """Answer each prompt with the reply meant for it."""

    def __init__(self, *, l1=None, l2=None, induced=None):
        self.replies = {"Experience Extraction": l1, "L2 Extraction": l2, "L2 Induction": induced}
        self.seen: list[str] = []

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage

        prompt = messages[0].content
        key = next(k for k in self.replies if k in prompt)
        self.seen.append(key)
        return AIMessage(content=self.replies[key])


@pytest.mark.asyncio
async def test_the_pipeline_reads_l1_and_l2_and_merges_the_induction():
    model = LevelModel(
        l1=reply(l1_record()),
        l2=reply(l2_record()),
        induced=reply(l2_record(
            origin="induced", locator="", source_l1_ids=["l1-1"], problem="Induced question.",
            compared_options=[{"option": "frozen encoder", "result": "saves a pass"}],
        )),
    )
    out = await run_experience_extraction(paper_id="p1", paper_text=PAPER, prompts=load_experience_prompts(), model=model)
    assert set(out) == {"l1", "l2"}
    assert [r["origin"] for r in out["l2"]["experiences"]] == ["paper", "induced"]
    assert sorted(model.seen) == ["Experience Extraction", "L2 Extraction", "L2 Induction"]


@pytest.mark.asyncio
async def test_a_failed_l2_does_not_fail_the_paper():
    model = LevelModel(l1=reply(l1_record()), l2="nope", induced=None)
    model.replies["L2 Induction"] = reply()
    out = await run_experience_extraction(paper_id="p1", paper_text=PAPER, prompts=load_experience_prompts(), model=model)
    assert len(out["l1"]["experiences"]) == 1 and out["l2"]["experiences"] == []


@pytest.mark.asyncio
async def test_an_induction_failure_is_skipped_and_l1_still_lands():
    model = LevelModel(l1=reply(l1_record()), l2=reply(l2_record()), induced="nope")
    out = await run_experience_extraction(paper_id="p1", paper_text=PAPER, prompts=load_experience_prompts(), model=model)
    assert [r["origin"] for r in out["l2"]["experiences"]] == ["paper"]


def test_storing_resolves_the_citation_to_the_stored_l1_id(tmp_path: Path):
    l1 = parse_experience_v2(reply(l1_record(), l1_record(problem="Second.", action="Second action.")), paper_id="p1")
    induced = l2_record(origin="induced", locator="", source_l1_ids=["l1-2"])
    l2 = parse_experience_v2(reply(induced), paper_id="p1", level="l2", source_ids={"l1-1", "l1-2"})
    store_paper_experiences(
        memory_dir=tmp_path, project_id="P-x", paper_id="p1", url="https://arxiv.org/abs/p1", title="T",
        paper_text=PAPER, prompts={"l1": "a", "l2": "b"}, payloads={"l1": l1, "l2": l2},
    )
    docs = list_experience_documents(memory_dir=tmp_path, project_id="P-x")
    l1_ids = [d.observation_id for d in docs if d.experience_level == "l1"]
    stored = json.loads((tmp_path / "experiences" / "projects" / "P-x" / docs[0].paper_key / "l2.json").read_text())
    assert stored["experiences"][0]["source_l1_ids"] == [l1_ids[1]]


def test_an_l2_record_renders_its_options_mechanism_and_conditions():
    text = render_record({"id": "E-1", "experience_level": "l2", "experience": l2_record()})
    assert "Options compared:" in text and "head features + initialisation" in text
    assert "Why it works:" in text and "Holds when" in text and "Stops holding when:" in text
    hit = render_hit(1, {"id": "E-1", "experience_level": "l2", "experience": l2_record()})
    assert "[L2]" in hit and "Claim:" in hit
