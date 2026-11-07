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
        raw = json.dumps({"suggestions": [_item(sources=[]), _item(because=""), _item(about=" "), "not a dict"]})
        out = parse_review(raw, ALLOWED, 6)
        assert out["suggestions"] == [] and out["dropped_without_source"] == 4

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
