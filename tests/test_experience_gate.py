"""The extraction quality gate: grade each record, keep the admitted ones, and never let an ungraded record in."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from EvoScientist.config.settings import (
    _ENV_MAPPINGS,
    EvoScientistConfig,
    get_effective_config,
)
from EvoScientist.memory.agents import paper_experience_worker
from EvoScientist.memory.experiences import (
    GateError,
    enqueue_paper,
    gate_extraction,
    list_experience_documents,
    list_tasks,
)
from EvoScientist.memory.experiences.gate import (
    dimension_score,
    evaluate_record,
    gate_mode,
    layer_checks,
    synthesize,
)
from EvoScientist.memory.experiences.store import _paper_dir

PAPER = (
    "Table 4 reports the verifier ablation. The verifier without head features gained 1.2 and 0.7 F1; with head "
    "features and initialization it gained 2.9 and 3.1 F1. The reader's [CLS] is trained for answer type, not for "
    "wrongness. Section 4.2 explains why a text-only verifier gains little. " * 3
)
QUOTE = "The reader's [CLS] is trained for answer type, not for wrongness."


def l1(**over):
    base = {
        "domain": "qa", "task": "answer verification", "problem": "Heuristic confidence is unreliable.",
        "method": ["answer verifier", "confidence model"],
        "requires": [{"text": "a trained reader", "basis": "stated"}],
        "source_setting": {"environment": [{"property": "dataset", "value": "NQ"}], "params": []},
        "limits_stated": [{"text": "the reader is not trained for wrongness", "quote": QUOTE}],
        "action": "Train a second-stage verifier.",
        "effect": {"compared": "with vs without head features", "benefit": "2.9 and 3.1 F1", "cost": None},
        "comparison": "controlled", "locator": "Table 4", "layer": "l1",
    }
    base.update(over)
    return base


def l2(**over):
    base = {
        "domain": "qa", "task": "answer verification", "problem": "When is a second verifier better than a score?",
        "method": ["answer verifier"], "action": "Feed the verifier the reader's head features.",
        "compared_options": [{"option": "text-only verifier", "result": "+1.2 F1"},
                             {"option": "head features", "result": "+2.9 F1"}],
        "mechanism": "The reader's logit margin is not comparable across questions.",
        "requires": [{"text": "the reader exposes logits", "basis": "stated"}],
        "limits_stated": [], "origin": "paper", "locator": "Section 4.2", "source_l1_ids": [], "layer": "l2",
    }
    base.update(over)
    return base


def payloads(l1_records=(), l2_records=()):
    return {"l1": {"paper_id": "p1", "domain_arxiv": None, "experiences": list(l1_records)},
            "l2": {"paper_id": "p1", "domain_arxiv": None, "experiences": list(l2_records)}}


class Reply:
    def __init__(self, text):
        self.content = text


class ScriptedJudge:
    """Answers each dimension prompt from `fails`: {check id: note}; everything else passes with a real span."""

    def __init__(self, fails=None, junk_for=(), span=None):
        self.fails, self.junk_for, self.calls = dict(fails or {}), set(junk_for), []
        self.span = span or "Table 4 reports the verifier ablation."

    async def ainvoke(self, messages):
        system, user = messages[0].content, messages[1].content
        self.calls.append(system.splitlines()[0])
        name = next(n for n in ("faithfulness", "validity", "scope", "inference") if n in system.split("\n", 1)[0].lower()
                    or {"faithfulness": "faithfulness", "validity": "comparison", "scope": "scope",
                        "inference": "claims that go beyond"}[n] in system.split("\n", 1)[0].lower())
        if name in self.junk_for:
            return Reply("I cannot grade this.")
        ids = user.rsplit("Grade exactly these checks: ", 1)[1].split(", ")
        checks = [{"id": i, "verdict": "fail" if i in self.fails else "pass", "evidence_span": self.span,
                   "note": self.fails.get(i, "ok")} for i in ids]
        return Reply(json.dumps({"checks": checks}))


@pytest.fixture
def mode(monkeypatch):
    def set_mode(value):
        monkeypatch.setenv("EVOSCIENTIST_MEMORY_EXPERIENCE_GATE", value)
    return set_mode


async def run(judge, l1_records=(), l2_records=()):
    return await gate_extraction(paper_id="p1", paper_text=PAPER, payloads=payloads(l1_records, l2_records), model=judge)


# --------------------------------------------------------------------- config

def test_the_gate_is_off_by_default_and_has_an_env_switch(mode):
    assert EvoScientistConfig().memory_experience_gate == "off"
    assert _ENV_MAPPINGS["memory_experience_gate"] == "EVOSCIENTIST_MEMORY_EXPERIENCE_GATE"
    assert gate_mode() == "off"
    mode("enforce")
    assert get_effective_config().memory_experience_gate == "enforce" and gate_mode() == "enforce"
    mode("strict")
    with pytest.raises(ValueError, match="off, audit, enforce"):
        gate_mode()


async def test_off_returns_the_payloads_untouched_and_calls_no_model():
    judge = ScriptedJudge()
    original = payloads([l1()], [l2()])
    out, report = await gate_extraction(paper_id="p1", paper_text=PAPER, payloads=original, model=judge)
    assert out is original and report is None and judge.calls == []


# ------------------------------------------------------------ status synthesis

def check(cid, verdict, *, dim, hard=False):
    return {"id": cid, "verdict": verdict, "dimension": dim, "hard": hard, "note": ""}


def test_status_rules_follow_the_rubric():
    clean = [check("F5", "pass", dim="faithfulness", hard=True), check("V1", "pass", dim="validity", hard=True),
             check("S2", "pass", dim="scope", hard=True)]
    assert synthesize(clean)["status"] == "ACCEPT"
    assert synthesize([*clean, check("I1", "fail", dim="inference")])["status"] == "PROVISIONAL"
    assert synthesize([*clean, check("I3", "fail", dim="inference")])["status"] == "REVISE"
    assert synthesize([*clean, check("I1", "fail", dim="inference"), check("I3", "fail", dim="inference")])["status"] == "REVISE"
    two_hard = [check("F5", "fail", dim="faithfulness", hard=True), check("F2", "fail", dim="faithfulness", hard=True)]
    assert synthesize([*two_hard, *clean[1:]])["status"] == "REJECT"
    assert synthesize([check("F5", "pass", dim="faithfulness"), check("V1", "pass", dim="validity")])["status"] == "INSUFFICIENT"
    assert dimension_score([check("A", "fail", dim="x", hard=True), check("B", "fail", dim="x")]) == 0
    assert dimension_score([check("A", "na", dim="x")]) is None


def test_l2_has_no_numeric_or_validity_gate():
    assert "V1" not in layer_checks("l2") and "F2" not in layer_checks("l2") and "S1" not in layer_checks("l2")
    only = [check("F5", "pass", dim="faithfulness", hard=True), check("S5", "pass", dim="scope", hard=True),
            check("I2", "pass", dim="inference")]
    assert synthesize(only, layer="l2")["status"] == "ACCEPT"


# ---------------------------------------------------------------- the judge

async def test_a_clean_record_is_accepted_and_every_program_check_runs():
    result = await evaluate_record(l1(), PAPER, paper_id="p1", layer="l1", model=ScriptedJudge(),
                                   sem=__import__("asyncio").Semaphore(2))
    assert result["status"] == "ACCEPT"
    by_id = {c["id"]: c for c in result["checks"]}
    assert by_id["F1"]["verdict"] == by_id["F2"]["verdict"] == by_id["F3"]["verdict"] == "pass"
    assert len(result["checks"]) == 18   # 3 program + 15 judge, A1-A3 are not graded


async def test_a_quote_the_paper_never_wrote_fails_in_code_without_asking_the_model():
    bad = l1(limits_stated=[{"text": "x", "quote": "The authors proved that every verifier collapses."}])
    result = await evaluate_record(bad, PAPER, paper_id="p1", layer="l1", model=ScriptedJudge(),
                                   sem=__import__("asyncio").Semaphore(2))
    f1 = next(c for c in result["checks"] if c["id"] == "F1")
    assert f1["verdict"] == "fail" and f1["hard"] and result["status"] != "ACCEPT"


async def test_a_number_missing_from_the_paper_and_a_missing_locator_fail():
    result = await evaluate_record(l1(effect={"compared": "a vs b", "benefit": "improves by 87.3%", "cost": None},
                                      locator="Table 9"), PAPER, paper_id="p1", layer="l1",
                                   model=ScriptedJudge(), sem=__import__("asyncio").Semaphore(2))
    by_id = {c["id"]: c for c in result["checks"]}
    assert by_id["F2"]["verdict"] == "fail" and by_id["F3"]["verdict"] == "fail"
    assert result["status"] == "REJECT"


async def test_a_pass_whose_span_is_not_in_the_paper_is_downgraded():
    judge = ScriptedJudge(span="a sentence the model made up about verifiers")
    result = await evaluate_record(l1(), PAPER, paper_id="p1", layer="l1", model=judge,
                                   sem=__import__("asyncio").Semaphore(2))
    judged = [c for c in result["checks"] if c["mode"] == "judge" and c["evidence"] == "paper"]
    assert judged and all(c["verdict"] == "fail" and "evidence not verifiable" in c["note"] for c in judged)
    assert result["status"] in {"REJECT", "REVISE"}
    assert len(judge.calls) == 8   # four dimensions, each asked once and repaired once


# ------------------------------------------------------------------ admission

async def test_enforce_keeps_admitted_records_and_drops_the_rest_but_reports_them(mode):
    mode("enforce")
    good, weak = l1(task="good one"), l1(task="weak one", problem="A different problem.")
    # the judge fails S5 for every record, so the weak one is told apart in code by its unverifiable locator
    weak["locator"] = "Table 9"
    out, report = await run(ScriptedJudge(), [good, weak])
    assert [r["task"] for r in out["l1"]["experiences"]] == ["good one"]
    assert report["graded"] == 2 and report["admitted"] == 1 and report["rejected"] == 1
    kept_out = [r for r in report["records"] if not r["admitted"]]
    assert kept_out[0]["record"]["task"] == "weak one" and kept_out[0]["status"] in {"REVISE", "REJECT"}
    assert kept_out[0]["reasons"]


async def test_audit_grades_and_reports_but_stores_everything(mode):
    mode("audit")
    original = payloads([l1(), l1(locator="Table 9", task="weak")])
    out, report = await gate_extraction(paper_id="p1", paper_text=PAPER, payloads=original, model=ScriptedJudge())
    assert out is original and len(out["l1"]["experiences"]) == 2
    assert report["mode"] == "audit" and report["rejected"] == 1


async def test_provisional_is_stored_with_the_mechanism_marked_unverified(mode):
    mode("enforce")
    out, report = await run(ScriptedJudge(fails={"I1": "mechanism not supported"}), [], [l2()])
    (record,) = out["l2"]["experiences"]
    assert record["gate"] == "PROVISIONAL" and record["mechanism"].startswith("UNVERIFIED_MECHANISM: ")
    assert report["records"][0]["status"] == "PROVISIONAL" and report["records"][0]["admitted"]


async def test_an_induced_finding_that_cites_a_dropped_l1_record_is_dropped_and_citations_are_renumbered(mode):
    mode("enforce")
    first, second, third = l1(task="one"), l1(task="two", locator="Table 9"), l1(task="three")
    cites_third = l2(origin="induced", locator="", source_l1_ids=["l1-3"], task="cites three")
    cites_second = l2(origin="induced", locator="", source_l1_ids=["l1-2"], task="cites two")
    out, report = await run(ScriptedJudge(), [first, second, third], [cites_third, cites_second])
    assert [r["task"] for r in out["l1"]["experiences"]] == ["one", "three"]
    assert [r["task"] for r in out["l2"]["experiences"]] == ["cites three"]
    assert out["l2"]["experiences"][0]["source_l1_ids"] == ["l1-2"]   # "three" is now the second kept record
    dropped = [r for r in report["records"] if r["level"] == "l2" and not r["admitted"]]
    assert dropped and "dropped" in dropped[0]["reasons"][-1]


async def test_an_unusable_judge_reply_raises_instead_of_admitting_the_record(mode):
    mode("enforce")
    with pytest.raises(GateError, match="no usable verdicts"):
        await run(ScriptedJudge(junk_for={"scope"}), [l1()])


async def test_records_in_the_old_shape_are_refused_not_waved_through(mode):
    mode("enforce")
    with pytest.raises(GateError, match="current"):
        await run(ScriptedJudge(), [{"statement": "an old v1 record", "layer": "l1"}])


# ------------------------------------------------------- through the worker

async def test_the_worker_stores_only_admitted_records_and_writes_gate_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode
):
    mode("enforce")
    enqueue_paper(memory_dir=tmp_path, project_id="P-alpha", paper_id="paper-1", url="https://example.invalid/p1",
                  title="Paper")

    async def download(_url):
        return PAPER

    async def extract(**_kwargs):
        return payloads([l1(task="kept"), l1(task="dropped", locator="Table 9")], [])

    async def grade(**kwargs):
        return await gate_extraction(**{**kwargs, "model": ScriptedJudge()})

    monkeypatch.setattr(paper_experience_worker, "download_paper_text", download)
    monkeypatch.setattr(paper_experience_worker, "load_experience_prompts", lambda: {"l1": "a", "l2": "b"})
    monkeypatch.setattr(paper_experience_worker, "run_experience_extraction", extract)
    monkeypatch.setattr(paper_experience_worker, "gate_extraction", grade)

    result = await paper_experience_worker.drain_paper_experience_queue(
        memory_dir=tmp_path, project_id="P-alpha", model=object())
    assert result == {"processed": 1, "failed": 0}
    documents = list_experience_documents(memory_dir=tmp_path, project_id="P-alpha")
    assert len(documents) == 1 and "kept" in documents[0].text and "dropped" not in documents[0].text
    gate_file = _paper_dir(tmp_path, project_id="P-alpha", paper_id="paper-1", url="https://example.invalid/p1") / "gate.json"
    report = json.loads(gate_file.read_text())
    assert report["admitted"] == 1 and report["rejected"] == 1


async def test_a_gate_failure_fails_the_task_and_stores_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    enqueue_paper(memory_dir=tmp_path, project_id="P-alpha", paper_id="paper-1", url="https://example.invalid/p1")

    async def download(_url):
        return PAPER

    async def extract(**_kwargs):
        return payloads([l1()], [])

    async def broken(**_kwargs):
        raise GateError("the judge gave no usable verdicts for scope of p1")

    monkeypatch.setattr(paper_experience_worker, "download_paper_text", download)
    monkeypatch.setattr(paper_experience_worker, "load_experience_prompts", lambda: {"l1": "a", "l2": "b"})
    monkeypatch.setattr(paper_experience_worker, "run_experience_extraction", extract)
    monkeypatch.setattr(paper_experience_worker, "gate_extraction", broken)

    result = await paper_experience_worker.drain_paper_experience_queue(
        memory_dir=tmp_path, project_id="P-alpha", model=object())
    assert result == {"processed": 0, "failed": 1}
    assert list_experience_documents(memory_dir=tmp_path, project_id="P-alpha") == []
    assert "no usable verdicts" in list_tasks(memory_dir=tmp_path, project_id="P-alpha", status="failed")[0].error


async def test_the_active_tool_gates_before_storing_and_says_what_it_kept_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode
):
    from EvoScientist.tools import paper_experience_active as active

    mode("enforce")

    async def run_sync(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    async def download(_url):
        return PAPER

    async def extract(**_kwargs):
        return payloads([l1(task="kept"), l1(task="dropped", locator="Table 9")], [])

    async def grade(**kwargs):
        return await gate_extraction(**{**kwargs, "model": ScriptedJudge()})

    monkeypatch.setattr(active, "download_paper_text", download)
    monkeypatch.setattr(active.asyncio, "to_thread", run_sync)
    monkeypatch.setattr(active, "load_experience_prompts", lambda: {"l1": "a", "l2": "b"})
    monkeypatch.setattr(active, "run_experience_extraction", extract)
    monkeypatch.setattr(active, "gate_extraction", grade)
    tool = active.create_extract_paper_experiences_tool(memory_dir=tmp_path, project_id="P-alpha")

    result = await tool.arun({"papers": [{"url": "https://example.invalid/p1", "paper_id": "paper-1", "title": "Paper"}]})

    assert "quality gate (enforce): 1 of 2 admitted, 1 kept out" in result
    documents = list_experience_documents(memory_dir=tmp_path, project_id="P-alpha")
    assert len(documents) == 1
    assert "kept" in documents[0].text
    assert "dropped" not in documents[0].text
    assert (_paper_dir(tmp_path, project_id="P-alpha", paper_id="paper-1",
                       url="https://example.invalid/p1") / "gate.json").is_file()
