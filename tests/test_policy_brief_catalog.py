"""The task brief (verified facts) and the library catalog (fixed-size view)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from EvoScientist.memory.policy.brief import (
    build_brief,
    fallback_facts,
    quote_in_statement,
    verify_facts,
)
from EvoScientist.memory.policy.catalog import (
    library_map,
    normalise_topic,
    record_line,
    situation_terms,
)

STATEMENT = """
# Overview
Answer questions about movie plots. The evaluation will be performed on the
`submission.csv` file using the Accuracy metric. If `has_answer` is false the
prediction is only correct if `no_answer` is true.
You have one RTX 4090 and 4 hours. Use only ./models/roberta-base.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
def evaluate(a, b):
    return a == b
"""


class TestQuotationCheck:
    def test_exact_quote_is_found(self) -> None:
        assert quote_in_statement("Use only ./models/roberta-base.", STATEMENT)

    def test_case_whitespace_and_backticks_are_ignored(self) -> None:
        assert quote_in_statement(
            "the evaluation will be performed on the submission.csv   file", STATEMENT
        )

    def test_an_invented_sentence_is_not_found(self) -> None:
        assert not quote_in_statement(
            "External LLM APIs are forbidden entirely.", STATEMENT
        )

    def test_a_very_short_quote_proves_nothing(self) -> None:
        assert not quote_in_statement("the", STATEMENT)


class TestVerifiedFacts:
    def test_unsupported_facts_are_dropped_and_counted(self) -> None:
        raw = [
            {
                "kind": "metric",
                "fact": "scored by accuracy",
                "quote": "using the Accuracy metric",
            },
            {
                "kind": "restriction",
                "fact": "no GPU compute allowed",
                "quote": "extra GPU compute is forbidden",
            },
            {"kind": "restriction", "fact": "no quote at all", "quote": ""},
        ]
        facts, dropped = verify_facts(raw, STATEMENT)
        assert [f.text for f in facts] == ["scored by accuracy"]
        assert dropped == 2

    def test_duplicates_collapse(self) -> None:
        item = {
            "kind": "metric",
            "fact": "scored by accuracy",
            "quote": "using the Accuracy metric",
        }
        facts, _ = verify_facts([item, dict(item)], STATEMENT)
        assert len(facts) == 1

    def test_unknown_kind_is_coerced(self) -> None:
        item = {
            "kind": "weird",
            "fact": "scored by accuracy",
            "quote": "using the Accuracy metric",
        }
        facts, _ = verify_facts([item], STATEMENT)
        assert facts[0].kind in {"goal", "data"}


class TestBuildBrief:
    @pytest.mark.asyncio
    async def test_model_facts_are_verified(self) -> None:
        payload = {
            "facts": [
                {
                    "kind": "metric",
                    "fact": "scored by accuracy",
                    "quote": "using the Accuracy metric",
                },
                {
                    "kind": "restriction",
                    "fact": "invented rule",
                    "quote": "this sentence is not in the statement",
                },
            ]
        }
        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=MagicMock(content=json.dumps(payload)))
        brief = await build_brief(STATEMENT, model)
        assert brief.source == "model"
        assert brief.dropped == 1
        assert "invented rule" not in brief.render()
        assert "scored by accuracy" in brief.render()

    @pytest.mark.asyncio
    async def test_a_failing_model_falls_back_without_licence_or_code(self) -> None:
        model = MagicMock()
        model.ainvoke = AsyncMock(side_effect=RuntimeError("down"))
        brief = await build_brief(STATEMENT, model)
        assert brief.source == "fallback"
        text = brief.render().lower()
        assert "copyright" not in text
        assert "def evaluate" not in text
        assert "roberta-base" in text

    @pytest.mark.asyncio
    async def test_unparseable_output_falls_back(self) -> None:
        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=MagicMock(content="no json here"))
        assert (await build_brief(STATEMENT, model)).source == "fallback"

    @pytest.mark.asyncio
    async def test_empty_statement(self) -> None:
        brief = await build_brief("", MagicMock())
        assert brief.facts == ()
        assert "no task brief" in brief.render()

    def test_fallback_skips_licence_and_code(self) -> None:
        texts = [f.text.lower() for f in fallback_facts(STATEMENT)]
        assert not any("copyright" in t or "rights reserved" in t for t in texts)
        assert not any(t.startswith("def ") for t in texts)


class TestCatalog:
    def test_domain_spellings_fold_together(self) -> None:
        variants = [
            "machine_reading_comprehension",
            "machine reading comprehension",
            "Machine Reading Comprehensions",
        ]
        assert len({normalise_topic(v) for v in variants}) == 1

    def test_missing_domain(self) -> None:
        assert normalise_topic(None) == "unknown"

    @staticmethod
    def _doc(
        i: int, domain: str, task: str = "a task", summary: str = ""
    ) -> SimpleNamespace:
        return SimpleNamespace(
            observation_id=f"E-{i}",
            domain=domain,
            task=task,
            summary=summary,
            experience_level="l1",
        )

    def test_map_size_does_not_grow_with_the_library(self) -> None:
        small = library_map([self._doc(i, f"topic{i % 5}") for i in range(20)])
        big = library_map([self._doc(i, f"topic{i % 5}") for i in range(20000)])
        assert len(big) < len(small) * 2
        assert big.count("\n") == small.count("\n")

    def test_map_is_capped_and_says_how_much_was_left_out(self) -> None:
        docs = [self._doc(i, f"topic number {i}") for i in range(60)]
        text = library_map(docs, limit=10)
        assert text.count("\n") == 10
        assert "and 50 more topics" in text

    def test_empty_library(self) -> None:
        assert "empty" in library_map([])

    def test_record_line_is_one_short_line(self) -> None:
        line = record_line(self._doc(7, "nlp", task="x " * 300, summary="y " * 300))
        assert "\n" not in line
        assert len(line) <= 170
        assert line.startswith("E-7 [L1]")

    def test_situation_terms_come_from_the_text_and_skip_filler(self) -> None:
        terms = situation_terms(
            "threshold calibration for unanswerable questions",
            "execute python train.py output result file error",
        ).split()
        assert "threshold" in terms
        assert "unanswerable" in terms
        assert "output" not in terms
        assert "result" not in terms
