"""L1 -> L2 induction over clusters of L1 records: assignment, splitting, grounding, state and storage."""

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
INDUCE_TITLE = "# L2 induction from a cluster of L1 records"


def l1(problem: str, method: str, *, limits: bool = False, gain: str = "1.5") -> dict[str, Any]:
    return {
        "domain": "qa", "task": "extractive question answering", "problem": problem, "method": [method, f"{method} alias"],
        "requires": [], "source_setting": {"environment": [], "params": []},
        "limits_stated": [{"text": f"{method} fails on long answers", "quote": "q"}] if limits else [],
        "action": f"Use {method} for {problem}", "effect": {"compared": "with vs without", "benefit": f"+{gain} F1", "cost": None},
        "mechanism": {"text": "", "basis": "stated"}, "comparison": "controlled", "locator": "Table 1",
    }


NULL = "most training windows hold no answer so the reader learns to abstain"


def store(mem: Path, paper: str, *records: dict[str, Any]) -> None:
    payload = parse_experience_v2(json.dumps({"experiences": list(records)}), paper_id=paper)
    store_paper_experiences(memory_dir=mem, project_id=PROJECT, paper_id=paper, url=f"https://arxiv.org/abs/{paper}",
                            title=paper, paper_text="+1.5 F1 +2.5 F1 " * 50, prompts={"l1": "a"}, payloads={"l1": payload})


OTHER = [("images are blurry after compression", "deblurring"), ("speech recognizers drift across accents", "accent adaptation"),
         ("graph embeddings collapse on dense subgraphs", "edge dropout"), ("time series forecasts lag sudden shifts", "patching"),
         ("tabular models overfit rare categories", "target encoding"), ("agents loop on repeated tool errors", "retry budgets")]


def library(tmp_path: Path) -> Path:
    """Three records on the empty-window problem (two papers) and six unrelated ones (one paper each).

    The unrelated records give the TF-IDF view a realistic vocabulary: in a library of only the related records, the
    words they share would appear in almost every record and carry no weight."""
    mem = tmp_path / "mem"
    store(mem, "p1", l1(NULL, "null-window downsampling", limits=True),
          l1(NULL + " on long documents", "null-window downsampling", limits=True, gain="2.5"))
    store(mem, "p2", l1(NULL + " during training", "null-window downsampling"))
    for k, (problem, method) in enumerate(OTHER):
        store(mem, f"o{k}", l1(problem, method))
    return mem


class Scripted:
    """Answers the induction prompt with `reply(human)`; records what it was asked."""

    def __init__(self, reply: Any) -> None:
        self.reply, self.asked = reply, []

    async def ainvoke(self, messages: Any) -> AIMessage:
        assert messages[0].content.splitlines()[0] == INDUCE_TITLE
        self.asked.append(messages[1].content)
        return AIMessage(content=json.dumps(self.reply(messages[1].content)))


def finding(ids: list[str], **over: Any) -> dict[str, Any]:
    base = {"domain": "qa", "task": "extractive qa", "problem": "too many empty windows",
            "method": ["null-window downsampling"], "action": "Downsample empty windows (+1.5, +2.5 F1)",
            "support": ids, "limits_stated": [{"text": "fails on long answers", "from": ids}]}
    base.update(over)
    return base


def induce_all(human: str) -> dict[str, Any]:
    return {"experiences": [finding([r["id"] for r in json.loads(human)["records"]])]}


def run(mem: Path, model: Any, **kw: Any) -> dict[str, Any]:
    return asyncio.run(cp.run_cross_paper_induction(memory_dir=mem, project_id=PROJECT, model=model, **kw))


def test_similar_records_share_a_centre_and_an_unrelated_one_starts_its_own(tmp_path: Path) -> None:
    recs = cp.load_l1_records(library(tmp_path), PROJECT)
    vecs = dict(zip([r["id"] for r in recs], cp.tfidf_vectors(recs), strict=True))
    state = cp.load_state(tmp_path / "none", PROJECT)
    cp.assign(state, [r["id"] for r in recs], vecs)
    sizes = sorted(len(c["members"]) for c in state["clusters"].values())
    assert sizes == [1] * len(OTHER) + [3]


def test_a_cluster_too_large_or_too_loose_is_split(monkeypatch) -> None:
    vecs = {f"a{i}": {"alpha": 1.0} for i in range(4)} | {f"b{i}": {"beta": 1.0} for i in range(4)}
    monkeypatch.setenv("EVOSCIENTIST_L2_CLUSTER_MAX_SIZE", "5")
    parts = cp.split(list(vecs), vecs)
    assert sorted(sorted(p) for p in parts) == [[f"a{i}" for i in range(4)], [f"b{i}" for i in range(4)]]
    # loose but within the cap: two directions in one cluster sit 0.71 from its centroid, below a 0.8 floor
    monkeypatch.setenv("EVOSCIENTIST_L2_CLUSTER_MAX_SIZE", "12")
    monkeypatch.setenv("EVOSCIENTIST_L2_CLUSTER_COHESION", "0.8")
    assert len(cp.split(list(vecs), vecs)) == 2
    # tight: never split
    tight = {f"a{i}": {"alpha": 1.0} for i in range(6)}
    assert cp.split(list(tight), tight) == [list(tight)]


def test_a_cluster_with_findings_is_due_once_new_members_reach_half_its_size() -> None:
    def cluster(size: int, induced: int, **extra: Any) -> dict[str, Any]:
        return {"members": [f"m{i}" for i in range(size)], "induced_size": induced, **extra}

    assert not cp.is_due(cluster(2, 0), has_findings=False)              # below the minimum: a centre, no findings
    assert cp.is_due(cluster(3, 0), has_findings=False)                  # first induction
    assert not cp.is_due(cluster(4, 3), has_findings=True)               # 1 new < max(2, ceil(0.5 * 3))
    assert cp.is_due(cluster(5, 3), has_findings=True)
    assert not cp.is_due(cluster(12, 9), has_findings=True)              # 3 new < ceil(0.5 * 9)
    assert cp.is_due(cluster(12, 8), has_findings=True)                  # 4 new = half of 8
    assert cp.is_due(cluster(4, 3, split=True), has_findings=True)       # a split part is revised at once


def test_validation_needs_three_members_and_grounded_limits() -> None:
    members = {"E-1": {"paper": "p1", "item": {"limits_stated": [{"text": "a"}], "effect": {"benefit": "+1.5 F1"}}},
               "E-2": {"paper": "p1", "item": {"limits_stated": [{"text": "b"}], "effect": {"benefit": "+2.5 F1"}}},
               "E-3": {"paper": "p2", "item": {"limits_stated": [], "effect": {}}}}
    local = {"r1": "E-1", "r2": "E-2", "r3": "E-3"}
    raw = finding(["r1", "r2", "r3", "rX"], action="do X: +1.5 and +2.5 F1",
                  limits_stated=[{"text": "both fail on long answers", "from": ["r1", "r2"]},
                                 {"text": "one limited record only", "from": ["r1", "r3"]}],
                  conflicts=[{"text": "they disagree", "from": ["r1", "r2"]}, {"text": "single", "from": ["r1"]}])
    rec = cp.validate_finding(raw, local, members)
    assert rec["source_l1_ids"] == ["E-1", "E-2", "E-3"] and rec["origin"] == "cross_paper"
    assert [x["text"] for x in rec["limits_stated"]] == ["both fail on long answers"]
    assert [x["text"] for x in rec["conflicts"]] == ["they disagree"]
    assert cp.validate_finding({**raw, "support": ["r1", "r2"]}, local, members) is None  # two members only
    assert cp.validate_finding({**raw, "action": "gain of 7.7 and 9.9 and 3.3"}, local, members) is None  # invented


def test_one_paper_alone_can_support_a_finding() -> None:
    members = {f"E-{i}": {"paper": "p1", "item": {"limits_stated": [], "effect": {}}} for i in range(3)}
    local = {f"r{i}": f"E-{i}" for i in range(3)}
    rec = cp.validate_finding(finding(list(local), action="Downsample empty windows", limits_stated=[]), local, members)
    assert rec is not None and rec["source_papers"] == ["p1"]


def test_a_trigger_induces_only_clusters_that_reach_the_minimum(tmp_path: Path) -> None:
    mem = library(tmp_path)
    model = Scripted(induce_all)
    out = run(mem, model, force=True)
    assert out["status"] == "ok" and out["clusters"] == 1 + len(OTHER) and out["clusters_eligible"] == 1
    asked = json.loads(model.asked[0])
    assert out["findings"] == 1 and len(model.asked) == 1 and len(asked["records"]) == 3
    assert "current_findings" not in asked
    docs = [d for d in list_experience_documents(memory_dir=mem, project_id=PROJECT) if d.paper_key == cp.CROSS_DIR]
    assert len(docs) == 1 and docs[0].experience_level == "l2"
    text = render_record(docs[0].text)
    assert "induced from a cluster of experimental records" in text and "3 experimental records from 2 papers" in text
    assert "Stops holding when (stated by the authors):" in text
    # findings are not read back as L1 input
    assert {r["paper"] for r in cp.load_l1_records(mem, PROJECT)} == {"p1", "p2"} | {f"o{k}" for k in range(len(OTHER))}


def test_below_the_trigger_nothing_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EVOSCIENTIST_CROSS_PAPER_EVERY", "50")
    model = Scripted(induce_all)
    assert run(library(tmp_path), model)["status"] == "waiting" and model.asked == []


def test_one_new_member_waits_and_the_findings_stay_as_they_are(tmp_path: Path) -> None:
    mem = library(tmp_path)
    run(mem, Scripted(induce_all), force=True)
    before = cp.load_state(mem, PROJECT)["findings"]
    store(mem, "p4", l1(NULL + " at inference", "null-window downsampling"))
    model = Scripted(induce_all)
    out = run(mem, model, force=True)
    assert out["new"] == 1 and out["induced"] == 0 and out["waiting_for_more"] == 1 and model.asked == []
    state = cp.load_state(mem, PROJECT)
    assert state["findings"] == before
    (cluster,) = [c for c in state["clusters"].values() if c["findings"]]
    assert len(cluster["members"]) == 4 and cluster["induced_size"] == 3


def test_enough_new_members_update_the_findings_from_the_current_ones(tmp_path: Path) -> None:
    mem = library(tmp_path)
    run(mem, Scripted(induce_all), force=True)
    store(mem, "p4", l1(NULL + " at inference", "null-window downsampling"),
          l1(NULL + " at evaluation", "null-window downsampling"))
    model = Scripted(induce_all)
    out = run(mem, model, force=True)
    assert out["new"] == 2 and out["induced"] == 1 and out["carried_over"] == 1
    asked = json.loads(model.asked[0])
    (current,) = asked["current_findings"]
    assert current["action"] == "Downsample empty windows (+1.5, +2.5 F1)" and current["support"] == ["r1", "r2", "r3"]
    assert len(asked["records"]) == 5
    (f,) = cp.load_state(mem, PROJECT)["findings"].values()
    assert len(f["record"]["source_l1_ids"]) == 5 and f["record"]["source_papers"] == ["p1", "p2", "p4"]


def test_a_finding_nothing_takes_over_is_retired_not_deleted(tmp_path: Path) -> None:
    mem = library(tmp_path)
    run(mem, Scripted(induce_all), force=True)
    store(mem, "p4", l1(NULL + " at inference", "null-window downsampling"),
          l1(NULL + " at evaluation", "null-window downsampling"))
    out = run(mem, Scripted(lambda _human: {"experiences": []}), force=True)
    state = cp.load_state(mem, PROJECT)
    assert out["retired"] == 1 and state["findings"] == {} and len(state["retired"]) == 1


def test_a_failed_call_is_retried_on_the_next_trigger_even_below_the_threshold(tmp_path: Path, monkeypatch) -> None:
    class Broken:
        async def ainvoke(self, _messages: Any) -> Any:
            raise TimeoutError()

    mem = library(tmp_path)
    out = run(mem, Broken(), force=True)
    assert out["status"] == "ok" and out["failed"] == 1 and out["findings"] == 0
    monkeypatch.setenv("EVOSCIENTIST_CROSS_PAPER_EVERY", "50")
    out = run(mem, Scripted(induce_all))
    assert out["status"] == "ok" and out["induced"] == 1 and out["findings"] == 1


def test_a_state_from_the_old_grouping_scheme_is_retired_and_reclustered(tmp_path: Path) -> None:
    mem = library(tmp_path)
    root = mem / "experiences" / "projects" / PROJECT / cp.CROSS_DIR
    root.mkdir(parents=True)
    (root / "state.json").write_text(json.dumps({"version": 1, "groups": {}, "findings": {"E-old": {"record": {"x": 1}}}}))
    out = run(mem, Scripted(induce_all), force=True)
    state = cp.load_state(mem, PROJECT)
    assert out["findings"] == 1 and [r["id"] for r in state["retired"]] == ["E-old"]


@pytest.mark.asyncio
async def test_the_switch_is_on_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("EVOSCIENTIST_MEMORY_CROSS_PAPER_ENABLED", raising=False)
    out = await cp.maybe_run_cross_paper(memory_dir=library(tmp_path), project_id=PROJECT, model=Scripted(induce_all))
    assert out is not None and out["status"] == "waiting"
