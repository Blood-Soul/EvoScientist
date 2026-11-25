"""Plan review: a finished plan is read against the library, and every suggestion must rest on a record it was shown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from EvoScientist.memory.policy.review import (
    parse_review,
    plan_queries,
    render_review,
    review_plan,
)

PLAN = """# Plan: windowed reader

## Direction and hypothesis
A sliding-window span reader with a CLS answerability head should beat the single-pass baseline on long plots.

## What to build
1. Window the plot into overlapping segments and train a span head with hard negatives sampled from the same plot.
2. Decide has_answer from the best window's null margin, tuned on validation with a grid over thresholds.

## Evidence
- measured on validation
"""

ALLOWED = {"E-aaaaaaaa", "E-bbbbbbbb"}


def _item(**over: Any) -> dict[str, Any]:
    base = {"about": "the null margin rule", "suggestion": "add the span's own score", "because": "in the source ...",
            "why_applicable": "the record's null-margin condition holds here",
            "sources": ["E-aaaaaaaa"], "kind": "add"}
    base.update(over)
    return base


class TestParse:
    def test_a_suggestion_with_a_valid_source_is_kept(self) -> None:
        out = parse_review(json.dumps({"suggestions": [_item()]}), ALLOWED, 6)
        assert [s["sources"] for s in out["suggestions"]] == [["E-aaaaaaaa"]]
        assert out["dropped_without_source"] == 0

    def test_a_suggestion_citing_a_record_it_was_not_shown_is_dropped(self) -> None:
        out = parse_review(json.dumps({"suggestions": [_item(sources=["E-ffffffff"]), _item()]}), ALLOWED, 6)
        assert len(out["suggestions"]) == 1 and out["dropped_without_source"] == 1

    def test_no_source_or_a_missing_field_is_dropped(self) -> None:
        raw = json.dumps({"suggestions": [_item(sources=[]), _item(because=""), _item(about=" "),
                                          _item(why_applicable=""), "not a dict"]})
        out = parse_review(raw, ALLOWED, 6)
        assert out["suggestions"] == [] and out["dropped_without_source"] == 5

    def test_only_the_valid_ids_of_a_mixed_citation_survive(self) -> None:
        out = parse_review(json.dumps({"suggestions": [_item(sources=["E-ffffffff", "E-bbbbbbbb", "E-bbbbbbbb"])]}), ALLOWED, 6)
        assert out["suggestions"][0]["sources"] == ["E-bbbbbbbb"]

    def test_the_cap_and_an_unknown_kind(self) -> None:
        out = parse_review(json.dumps({"suggestions": [_item(kind="weird") for _ in range(9)]}), ALLOWED, 3)
        assert len(out["suggestions"]) == 3 and out["suggestions"][0]["kind"] == "check"

    def test_fenced_json_and_trailing_prose(self) -> None:
        raw = "```json\n" + json.dumps({"suggestions": [_item()]}) + "\n```"
        assert len(parse_review(raw, ALLOWED, 6)["suggestions"]) == 1
        assert len(parse_review("Here: " + json.dumps({"suggestions": [_item()]}) + " done", ALLOWED, 6)["suggestions"]) == 1

    def test_unreadable_output_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_review("no json here", ALLOWED, 6)


class TestRender:
    def test_each_item_names_its_records_and_the_reader_may_refuse(self) -> None:
        review = parse_review(json.dumps({"suggestions": [_item()]}), ALLOWED, 6)
        text = render_review(review, {"E-aaaaaaaa": "Read + Verify"})
        assert "E-aaaaaaaa (Read + Verify)" in text and "refuse it" in text and "[add]" in text
        assert "Why applicable here" in text and "Falsifier" not in text

    def test_an_empty_review_says_so(self) -> None:
        assert "no suggestions" in render_review({"suggestions": [], "no_suggestion_reason": "nothing bears on it"})


def test_queries_are_drawn_from_the_plans_own_words() -> None:
    qs = plan_queries(PLAN)
    assert any("sliding-window span reader" in q for q in qs)
    assert any("hard negatives" in q for q in qs)
    assert len(qs) == len({q.lower() for q in qs})


class _Model:
    def __init__(self, reply: str) -> None:
        self.reply, self.calls = reply, 0

    async def ainvoke(self, _messages: Any) -> Any:
        self.calls += 1
        return AIMessage(content=self.reply)


@pytest.mark.asyncio
async def test_an_empty_plan_costs_no_model_call(tmp_path: Path) -> None:
    model = _Model("{}")
    out = await review_plan(memory_dir=tmp_path, project_id="P-x", task="t", plan="   ", model=model)
    assert out["status"] == "empty_plan" and model.calls == 0


@pytest.mark.asyncio
async def test_an_empty_library_costs_no_model_call(tmp_path: Path) -> None:
    model = _Model("{}")
    out = await review_plan(memory_dir=tmp_path, project_id="P-x", task="t", plan=PLAN, model=model)
    assert out["status"] == "no_records" and model.calls == 0 and "no stored experience" in out["text"]


def test_the_review_prompt_formats() -> None:
    from EvoScientist.memory.policy.prompts import load_policy_prompt
    from EvoScientist.memory.policy.review import REVIEW_PROMPT_FILENAME

    text = load_policy_prompt(REVIEW_PROMPT_FILENAME).format(task="T", plan="P", records="R", max_suggestions=6)
    assert "T" in text and "at most 6" in text


def test_plan_design_point_reads_the_plans_own_sections() -> None:
    from EvoScientist.memory.policy.review import plan_design_point

    point = plan_design_point(PLAN, "predict a property")
    assert "sliding-window span reader" in point and "hard negatives" in point
    assert "predict a property" in point


class _RetryModel:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, _messages: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="I could not decide.")
        return AIMessage(content=json.dumps({"suggestions": [{
            "about": "the null margin rule", "suggestion": "check the threshold grid",
            "because": "in the source the threshold was measured", "why_applicable": "the same validation split exists",
            "sources": ["E-aaaaaaaa"], "kind": "check"}]}))


@pytest.mark.asyncio
async def test_an_unparsable_review_is_retried_once(tmp_path: Path, monkeypatch) -> None:
    import EvoScientist.memory.policy.review as review_mod

    candidate = {"id": "E-aaaaaaaa", "record": {"experience": {"domain": "x", "task": "t", "problem": "p"},
                                                "paper": {}}}
    monkeypatch.setattr(review_mod, "gather_candidates", lambda **_kw: [candidate])
    monkeypatch.setattr(review_mod, "fetch_candidates", lambda **_kw: [])
    model = _RetryModel()
    out = await review_mod.review_plan(memory_dir=tmp_path, project_id="P-x", task="t", plan=PLAN, model=model)
    assert out["status"] == "ok" and len(out["suggestions"]) == 1 and model.calls == 2


def _store_cross_field(tmp_path: Path) -> None:
    from EvoScientist.memory.experiences.store import store_paper_experiences

    record = {
        "domain": "ecology", "task": "surveying sparse habitats",
        "problem": "Most survey plots are empty, so a model learns to predict empty.",
        "method": ["stratified plot selection"], "requires": [{"text": "a plot map", "basis": "inferred"}],
        "source_setting": {"environment": [{"property": "empty share", "value": "90%"}], "params": []},
        "limits_stated": [], "action": "Measure the empty share, then keep a fraction of empty plots.",
        "effect": {"compared": "20% vs 66% empty", "benefit": "F1 87.3", "cost": None},
        "mechanism": {"text": "An empty plot teaches the model the safe answer.", "basis": "inferred"},
        "comparison": "controlled", "locator": "Table 1",
    }
    store_paper_experiences(
        memory_dir=tmp_path, project_id="P-x", paper_id="p1", url="https://example.org/p1",
        title="Ecology paper", paper_text="text", prompts={"l1": "p"},
        payloads={"l1": {"experiences": [record]}},
    )


class _CrossOnlyModel:
    """Answers the abstract and gate prompts of the cross-field leg; it is never asked a same-field review."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages: Any) -> Any:
        self.calls += 1
        system = str(messages[0].content)
        if "practice FROM ANOTHER FIELD" in system:
            return AIMessage(content=json.dumps({"goal": "how to keep a model from learning the empty answer",
                                                 "passages": ["[ecology] survey the empty patches first"]}))
        return AIMessage(content=json.dumps({"judgements": [{
            "i": 1, "transferable": True,
            "object_mappings": [
                {"source": "empty training window", "target": "empty survey plot", "rationale": "same majority class"},
                {"source": "null downsampling", "target": "stratified plot selection", "rationale": "same sampling decision"},
            ],
            "shared_relations": [{"relation": "OUTNUMBER", "problem": "OUTNUMBER(empty window, filled window)",
                                  "solution": "OUTNUMBER(empty plot, occupied plot)"}],
            "path": "measure the empty share, then fix the sampling fraction", "drop_reason": None,
        }]}))


@pytest.mark.asyncio
async def test_the_review_appends_a_cross_field_section(tmp_path: Path, monkeypatch) -> None:
    import EvoScientist.memory.policy.review as review_mod

    _store_cross_field(tmp_path)
    # No same-field matches: the review should still return the cross-field leg.
    monkeypatch.setattr(review_mod, "gather_candidates", lambda **_kw: [])
    monkeypatch.setattr(review_mod, "fetch_candidates", lambda **_kw: [])
    model = _CrossOnlyModel()
    out = await review_mod.review_plan(
        memory_dir=tmp_path, project_id="P-x", task="molecular property prediction",
        plan="# Plan: x\n\n## Direction and hypothesis\nA model must decide when evidence is missing.\n",
        model=model, cross_domain=True,
    )
    assert out["status"] == "no_records"
    assert out["cross_domain"]["kept"] == 1
    assert "Cross-field transfer hypotheses (hypotheses, not evidence)" in out["text"]
    assert "OUTNUMBER(empty plot, occupied plot)" in out["text"]
    assert model.calls == 2  # abstract + one gate batch


@pytest.mark.asyncio
async def test_no_cross_field_section_without_the_switch(tmp_path: Path, monkeypatch) -> None:
    import EvoScientist.memory.policy.review as review_mod

    _store_cross_field(tmp_path)
    monkeypatch.setattr(review_mod, "gather_candidates", lambda **_kw: [])
    monkeypatch.setattr(review_mod, "fetch_candidates", lambda **_kw: [])
    model = _CrossOnlyModel()
    out = await review_mod.review_plan(
        memory_dir=tmp_path, project_id="P-x", task="t",
        plan="# Plan: x\n\n## Direction and hypothesis\nA model must decide when evidence is missing.\n",
        model=model,
    )
    assert out["cross_domain"] is None and "Cross-field transfer hypotheses" not in out["text"]
    assert model.calls == 0


def test_a_revised_plan_from_the_reviewer_is_ignored() -> None:
    # The reviewer only advises; the writer revises the file itself.
    raw = json.dumps({"suggestions": [_item()], "revised_plan": "# Plan: x\n\nbody"})
    assert "revised_plan" not in parse_review(raw, ALLOWED, 6)


def test_the_review_text_shows_the_applicability_and_leaves_the_decision_to_the_writer() -> None:
    review = parse_review(json.dumps({"suggestions": [_item()]}), ALLOWED, 6)
    text = render_review(review)
    assert "Why applicable here:" in text and "Falsifier" not in text and "You decide" in text


def test_the_prompt_is_advice_only() -> None:
    from EvoScientist.memory.policy.prompts import load_policy_prompt
    from EvoScientist.memory.policy.review import REVIEW_PROMPT_FILENAME

    text = load_policy_prompt(REVIEW_PROMPT_FILENAME).format(task="T", plan="P", records="R", max_suggestions=6)
    assert "revised_plan" not in text and "why_applicable" in text and "falsifier" not in text
    assert "takes or refuses each suggestion" in text


@pytest.mark.asyncio
async def test_the_tool_never_rewrites_the_plan_file(tmp_path: Path, monkeypatch) -> None:
    from EvoScientist.tools import plan_review as tool_mod

    async def fake_review(**_kw):
        return {"status": "ok", "text": "Plan review: 1 suggestion", "suggestions": [], "revised_plan": "# overwritten"}

    monkeypatch.setattr(tool_mod, "review_plan", fake_review)
    import EvoScientist.EvoScientist as evo

    monkeypatch.setattr(evo, "_ensure_auxiliary_chat_model", lambda: object())
    plan = tmp_path / "reports" / "plan.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Plan: original\n", encoding="utf-8")
    tool = tool_mod.create_review_plan_tool(memory_dir=tmp_path / "mem", project_id="P-x", workspace_root=tmp_path)
    out = await tool.coroutine(path="reports/plan.md", task="t")
    assert out == "Plan review: 1 suggestion"
    assert plan.read_text(encoding="utf-8") == "# Plan: original\n"


def test_round_robin_gives_every_query_its_best_record_first() -> None:
    from EvoScientist.memory.policy.review import merge_round_robin

    # query 0 scored low across the board, query 1 high: scores must not be compared across queries
    q0 = [{"id": "a1", "condition": 0.1}, {"id": "a2", "condition": 0.05}]
    q1 = [{"id": "b1", "condition": 1.0}, {"id": "b2", "condition": 0.9}, {"id": "a1", "condition": 0.8}]
    assert list(merge_round_robin([q0, q1])) == ["a1", "b1", "a2", "b2"]
    assert merge_round_robin([]) == {}


def test_stopwords_and_template_words_do_not_count_as_a_condition_match() -> None:
    from EvoScientist.memory.policy.select import _condition_score, _tokens_of

    assert _tokens_of("2. What the model sees") == {"sees"}
    record = {"experience": {"task": "what the model does", "domain": "", "problem": "the data"}}
    assert _condition_score("What the model sees", record) == 0.0
