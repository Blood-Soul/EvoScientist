"""The single-level extraction: parse, per-record validation, number grounding, storage and the runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from EvoScientist.memory.experiences.extraction import (
    ExperienceOutputError,
    load_experience_prompts,
    parse_experience_v2,
    run_experience_extraction,
)
from EvoScientist.memory.experiences.store import (
    list_experience_documents,
    load_paper_experiences,
    store_paper_experiences,
)

PAPER = "Training with 33% unanswerable examples gives F1 83.05; 66% gives 87.32 and 20% gives 76.93. " * 6


def record(**over):
    base = {
        "domain": "qa", "task": "negative ratio", "problem": "Nobody knows the share of unanswerable examples.",
        "method": ["unanswerable ratio", "negative share"],
        "requires": [{"text": "a no-answer model", "basis": "inferred"}],
        "source_setting": {"environment": [{"property": "model", "value": "roberta"}],
                           "params": [{"name": "share", "value": "33%", "basis": "swept", "from": "20, 33, 66"}]},
        "limits_stated": [], "action": "Sweep the share.",
        "effect": {"compared": "20 vs 33 vs 66", "benefit": "F1 83.05, 87.32, 76.93", "cost": None},
        "comparison": "controlled", "locator": "Table 1",
    }
    base.update(over)
    return base


def reply(*records):
    return "```json\n" + json.dumps({"experiences": list(records)}) + "\n```"


def test_a_valid_reply_is_parsed_and_marked_as_one_level():
    out = parse_experience_v2(reply(record()), paper_id="p1", paper_text=PAPER, domain_arxiv="cs.CL")
    assert out["paper_id"] == "p1" and out["domain_arxiv"] == "cs.CL"
    assert out["experiences"][0]["layer"] == "l1"


def test_a_missing_limits_key_is_tolerated():
    r = record()
    del r["limits_stated"]
    assert parse_experience_v2(reply(r), paper_id="p1")["experiences"][0]["limits_stated"] == []


def test_bad_records_are_dropped_one_by_one():
    out = parse_experience_v2(reply(record(), record(comparison="maybe"), record(task="")), paper_id="p1")
    assert len(out["experiences"]) == 1


def test_no_survivor_is_an_error_but_an_empty_list_is_fine():
    with pytest.raises(ExperienceOutputError, match="no record survived"):
        parse_experience_v2(reply(record(comparison="maybe")), paper_id="p1")
    assert parse_experience_v2('{"experiences": []}', paper_id="p1")["experiences"] == []


def test_numbers_the_paper_never_writes_cost_the_record():
    invented = record(effect={"compared": "x", "benefit": "F1 91.11, 92.22, 93.33, 94.44", "cost": None})
    out = parse_experience_v2(reply(record(), invented), paper_id="p1", paper_text=PAPER)
    assert len(out["experiences"]) == 1
    # PDF spacing in the paper does not count against a real number
    spaced = "F1 83 .05 and 87 .32 and 76 .93 " * 3
    assert parse_experience_v2(reply(record()), paper_id="p1", paper_text=spaced)["experiences"]


def test_truncated_json_is_reported_as_cut_off():
    with pytest.raises(ExperienceOutputError, match="malformed JSON"):
        parse_experience_v2('{"experiences": [{"domain": "qa"', paper_id="p1")


class FakeModel:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        return AIMessage(content=self.replies.pop(0))


@pytest.mark.asyncio
async def test_an_l1_only_prompt_mapping_still_selects_the_single_level_runner_and_retries_once():
    prompts = {"l1": load_experience_prompts()["l1"]}
    model = FakeModel('{"experiences": [{"domain"', reply(record()))
    out = await run_experience_extraction(paper_id="p1", paper_text=PAPER, prompts=prompts, model=model)
    assert set(out) == {"l1"} and len(out["l1"]["experiences"]) == 1
    assert model.calls == 2


@pytest.mark.asyncio
async def test_it_gives_up_after_three_bad_replies():
    model = FakeModel("nope", "nope", "nope")
    with pytest.raises(ExperienceOutputError):
        await run_experience_extraction(
            paper_id="p1", paper_text=PAPER, prompts={"l1": load_experience_prompts()["l1"]}, model=model
        )
    assert model.calls == 3


def test_one_level_is_stored_listed_and_reloaded(tmp_path: Path):
    prompts = load_experience_prompts()
    payload = parse_experience_v2(reply(record(), record(problem="Another.", action="Another action.")), paper_id="p1")
    kwargs = dict(memory_dir=tmp_path, project_id="P-x", paper_id="p1", url="https://arxiv.org/abs/p1")
    store_paper_experiences(**kwargs, title="T", paper_text=PAPER, prompts=prompts, payloads={"l1": payload})
    docs = list_experience_documents(memory_dir=tmp_path, project_id="P-x")
    assert len(docs) == 2 and {d.experience_level for d in docs} == {"l1"}
    cached = load_paper_experiences(**kwargs)
    assert cached is not None and cached["l2"] == {"experiences": []}
    assert cached["metadata"]["extraction_schema"] == "v2"
