"""Cross-paper L1 -> L2 induction: candidate grouping, validation rules, state and storage, with a scripted model."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from EvoScientist.memory.experiences import cross_paper as cp
from EvoScientist.memory.experiences.extraction import parse_experience_v2
from EvoScientist.memory.experiences.render import render_record
from EvoScientist.memory.experiences.store import list_experience_documents, store_paper_experiences

PROJECT = "P-x"


def l1(problem: str, method: str, *, limits: bool = False, gain: str = "1.5") -> dict[str, Any]:
    return {
        "domain": "qa", "task": "extractive question answering", "problem": problem, "method": [method, f"{method} alias"],
        "requires": [], "source_setting": {"environment": [], "params": []},
        "limits_stated": [{"text": f"{method} fails on long answers", "quote": "q"}] if limits else [],
        "action": f"Use {method} for {problem}", "effect": {"compared": "with vs without", "benefit": f"+{gain} F1", "cost": None},
        "comparison": "controlled", "locator": "Table 1",
    }


def store(mem: Path, paper: str, *records: dict[str, Any]) -> None:
    text = "+1.5 F1 +2.5 F1 " * 50
    payload = parse_experience_v2(json.dumps({"experiences": list(records)}), paper_id=paper)
    store_paper_experiences(memory_dir=mem, project_id=PROJECT, paper_id=paper, url=f"https://arxiv.org/abs/{paper}",
                            title=paper, paper_text=text, prompts={"l1": "a"}, payloads={"l1": payload})


class Scripted:
    """Answers by the prompt's title line; records what it was asked."""

    def __init__(self, replies: dict[str, Any]) -> None:
        self.replies, self.asked = replies, []

    async def ainvoke(self, messages: Any) -> AIMessage:
        title = messages[0].content.splitlines()[0]
        self.asked.append((title, messages[1].content))
        reply = self.replies[title]
        return AIMessage(content=json.dumps(reply(messages[1].content) if callable(reply) else reply))


def library(tmp_path: Path) -> Path:
    mem = tmp_path / "mem"
    store(mem, "p1", l1("most windows hold no answer", "null-window downsampling", limits=True))
    store(mem, "p2", l1("most windows hold no answer so the model abstains", "null-window downsampling", limits=True,
                        gain="2.5"))
    store(mem, "p3", l1("images are blurry", "deblurring"))
    return mem


def test_candidate_groups_pair_similar_records_across_papers(tmp_path: Path) -> None:
    recs = cp.load_l1_records(library(tmp_path), PROJECT)
    groups = cp.candidate_groups(recs)
    assert len(groups) == 1 and {recs[i]["paper"] for i in groups[0]} == {"p1", "p2"}


def test_validation_enforces_cross_paper_support_and_grounded_limits() -> None:
    members = {"E-1": {"paper": "p1", "item": {"limits_stated": [{"text": "a"}], "effect": {"benefit": "+1.5 F1"}}},
               "E-2": {"paper": "p2", "item": {"limits_stated": [{"text": "b"}], "effect": {"benefit": "+2.5 F1"}}},
               "E-3": {"paper": "p2", "item": {"limits_stated": [], "effect": {}}}}
    local = {"r1": "E-1", "r2": "E-2", "r3": "E-3"}
    raw = {"domain": "qa", "task": "t", "problem": "p", "method": ["m"], "action": "do X: +1.5 and +2.5 F1",
           "support": ["r1", "r2", "rX"],
           "limits_stated": [{"text": "both fail on long answers", "from": ["r1", "r2"]},
                             {"text": "one record only", "from": ["r1", "r3"]}],
           "conflicts": [{"text": "they disagree", "from": ["r1", "r2"]}, {"text": "single", "from": ["r1"]}]}
    rec = cp.validate_finding(raw, local, members)
    assert rec["source_l1_ids"] == ["E-1", "E-2"] and rec["origin"] == "cross_paper"
    assert [x["text"] for x in rec["limits_stated"]] == ["both fail on long answers"]
    assert [x["text"] for x in rec["conflicts"]] == ["they disagree"]
    assert cp.validate_finding({**raw, "support": ["r2", "r3"]}, local, members) is None  # one paper only
    assert cp.validate_finding({**raw, "action": "gain of 7.7 and 9.9 and 3.3"}, local, members) is None  # invented numbers


def _group_reply(human: str) -> dict[str, Any]:
    ids = [r["id"] for r in json.loads(human)]
    return {"groups": [{"members": ids, "context": "most windows have no answer"}]}


def _induce_reply(human: str) -> dict[str, Any]:
    ids = [r["id"] for r in json.loads(human.split("\n\n", 1)[1])]
    return {"experiences": [{"domain": "qa", "task": "extractive qa", "problem": "too many empty windows",
                             "method": ["null-window downsampling"], "action": "Downsample empty windows (+1.5, +2.5 F1)",
                             "support": ids, "limits_stated": [{"text": "fails on long answers", "from": ids}]}]}


def test_a_rebuild_stores_findings_that_listing_and_rendering_understand(tmp_path: Path) -> None:
    mem = library(tmp_path)
    model = Scripted({"# Cross-paper grouping of experience records": _group_reply,
                      "# Cross-paper L2 induction": _induce_reply})
    out = asyncio.run(cp.run_cross_paper_induction(memory_dir=mem, project_id=PROJECT, model=model, force="rebuild"))
    assert out["status"] == "ok" and out["findings"] == 1 and out["groups"] == 1
    docs = [d for d in list_experience_documents(memory_dir=mem, project_id=PROJECT) if d.paper_key == cp.CROSS_DIR]
    assert len(docs) == 1 and docs[0].experience_level == "l2"
    text = render_record(docs[0].text)
    assert "induced across 2 papers" in text and "Applies in this situation: most windows have no answer" in text
    assert "Stops holding when (stated by the authors):" in text
    # the cross-paper findings are not themselves read back as L1 input
    assert {r["paper"] for r in cp.load_l1_records(mem, PROJECT)} == {"p1", "p2", "p3"}


def test_below_the_trigger_nothing_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EVOSCIENTIST_CROSS_PAPER_EVERY", "50")
    model = Scripted({})
    out = asyncio.run(cp.run_cross_paper_induction(memory_dir=library(tmp_path), project_id=PROJECT, model=model))
    assert out["status"] == "waiting" and model.asked == []


def test_incremental_assigns_and_revises_with_an_operation(tmp_path: Path, monkeypatch) -> None:
    mem = library(tmp_path)
    model = Scripted({"# Cross-paper grouping of experience records": _group_reply,
                      "# Cross-paper L2 induction": _induce_reply})
    asyncio.run(cp.run_cross_paper_induction(memory_dir=mem, project_id=PROJECT, model=model, force="rebuild"))
    store(mem, "p4", l1("most windows hold no answer", "negative window sampling", limits=True))
    gid = next(iter(cp.load_state(mem, PROJECT)["groups"]))
    model.replies.update({
        "# Cross-paper assignment of new experience records": {"assign": {"n1": gid}},
        "# Cross-paper L2 revision": lambda human: {
            "op": "ADD_CONFLICT", "add_support": ["r3"],
            "conflict": {"text": "a third paper samples negatives instead and finds no gain", "from": ["r1", "r3"]},
            "reason": "disagrees"},
    })
    out = asyncio.run(cp.run_cross_paper_induction(memory_dir=mem, project_id=PROJECT, model=model, force="incremental"))
    assert out["status"] == "ok" and out["assigned"] == 1 and out["ops"] == {"ADD_CONFLICT": 1}
    state = cp.load_state(mem, PROJECT)
    (finding,) = state["findings"].values()
    assert len(finding["record"]["source_l1_ids"]) == 3 and finding["record"]["source_papers"] == ["p1", "p2", "p4"]
    assert finding["record"]["conflicts"][0]["text"].startswith("a third paper")
    assert finding["version"] == 2


def test_a_revision_that_shrinks_the_finding_is_rejected(tmp_path: Path) -> None:
    mem = library(tmp_path)
    model = Scripted({"# Cross-paper grouping of experience records": _group_reply,
                      "# Cross-paper L2 induction": _induce_reply})
    asyncio.run(cp.run_cross_paper_induction(memory_dir=mem, project_id=PROJECT, model=model, force="rebuild"))
    before = next(iter(cp.load_state(mem, PROJECT)["findings"].values()))["record"]["action"]
    store(mem, "p4", l1("most windows hold no answer", "negative window sampling"))
    gid = next(iter(cp.load_state(mem, PROJECT)["groups"]))
    model.replies.update({
        "# Cross-paper assignment of new experience records": {"assign": {"n1": gid}},
        "# Cross-paper L2 revision": {"op": "REVISE", "changes": {"action": "x", "limits_stated": []}, "reason": "r"},
    })
    out = asyncio.run(cp.run_cross_paper_induction(memory_dir=mem, project_id=PROJECT, model=model, force="incremental"))
    assert out["ops"] == {"KEEP": 1}
    assert next(iter(cp.load_state(mem, PROJECT)["findings"].values()))["record"]["action"] == before


def test_a_failing_model_never_raises(tmp_path: Path) -> None:
    class Broken:
        async def ainvoke(self, _messages: Any) -> Any:
            raise RuntimeError("gateway down")

    out = asyncio.run(cp.run_cross_paper_induction(memory_dir=library(tmp_path), project_id=PROJECT, model=Broken(),
                                                   force="rebuild"))
    assert out["status"] == "ok" and out["findings"] == 0


@pytest.mark.asyncio
async def test_the_switch_is_off_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("EVOSCIENTIST_MEMORY_CROSS_PAPER_ENABLED", raising=False)
    assert await cp.maybe_run_cross_paper(memory_dir=library(tmp_path), project_id=PROJECT, model=Scripted({})) is None


def test_failed_calls_are_retried_on_the_next_trigger(tmp_path: Path) -> None:
    mem = library(tmp_path)

    class Flaky(Scripted):
        fail = True

        async def ainvoke(self, messages: Any) -> AIMessage:
            if self.fail and messages[0].content.startswith("# Cross-paper L2 induction"):
                raise TimeoutError()
            return await super().ainvoke(messages)

    model = Flaky({"# Cross-paper grouping of experience records": _group_reply,
                   "# Cross-paper L2 induction": _induce_reply})
    out = asyncio.run(cp.run_cross_paper_induction(memory_dir=mem, project_id=PROJECT, model=model, force="rebuild"))
    assert out["groups"] == 1 and out["findings"] == 0
    model.fail = False
    out = asyncio.run(cp.run_cross_paper_induction(memory_dir=mem, project_id=PROJECT, model=model, force="incremental"))
    assert out["reinduced"] == 1 and out["findings"] == 1
