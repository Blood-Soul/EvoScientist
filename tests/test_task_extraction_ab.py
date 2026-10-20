"""Check isolation of the two arms without treating replay as fresh generation."""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from scripts.prepare_task_extraction_ab import digest, prepare
from scripts.replay_task_extraction_comparison import SuppliedExtractionResponses


def test_exported_requests_differ_only_by_task_block(tmp_path):
    text = "A paper body with its own evidence and [paper_id] inside the text."
    paper_path = tmp_path / "paper.md"
    paper_path.write_text(text, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "paper_id": "p1",
                    "text_path": str(paper_path),
                    "paper_sha256": digest(text),
                }
            ]
        ),
        encoding="utf-8",
    )
    context = tmp_path / "task.json"
    context.write_text('{"task":"forecast five weeks"}', encoding="utf-8")
    output = tmp_path / "requests"
    prepare(manifest, context, output)
    for level in ("l1", "l2"):
        neutral = json.loads(
            (output / "neutral" / f"p1.{level}.request.json").read_text()
        )
        task = json.loads((output / "task" / f"p1.{level}.request.json").read_text())
        assert neutral["system"] == task["system"]
        assert neutral["paper_sha256"] == task["paper_sha256"]
        assert neutral["human"] == "[paper_id] p1\n\n" + text
        assert "[task_context]" not in neutral["human"]
        assert "[task_context]" in task["human"]
        assert task["human"].endswith("\n\n" + text)
        assert "strict task-selection overlay" not in task["system"]
    assert not list(output.rglob("*.response.json"))
    audit = json.loads((output / "input-audit.json").read_text())
    assert audit["status"] == "inputs_prepared_no_extraction_results"


def test_neutral_replay_accepts_no_context_and_records_actual_input(tmp_path):
    (tmp_path / "p1.l1.json").write_text('{"experiences": []}', encoding="utf-8")
    model = SuppliedExtractionResponses(
        tmp_path,
        {"l1": "BASE PROMPT"},
        expect_task_context=False,
        messages_dir=tmp_path / "actual-messages",
    )
    messages = [
        SystemMessage(content="BASE PROMPT"),
        HumanMessage(content="[paper_id] p1\n\nBODY"),
    ]
    response = asyncio.run(model.ainvoke(messages))
    assert json.loads(response.content) == {"experiences": []}
    assert model.calls[0]["external_model_call"] is False
    saved = json.loads((tmp_path / "actual-messages/p1.l1.json").read_text())
    assert saved["human"] == messages[-1].content


def test_neutral_replay_rejects_task_contamination(tmp_path):
    (tmp_path / "p1.l1.json").write_text('{"experiences": []}', encoding="utf-8")
    model = SuppliedExtractionResponses(
        tmp_path, {"l1": "BASE"}, expect_task_context=False
    )
    with pytest.raises(AssertionError):
        asyncio.run(
            model.ainvoke(
                [
                    SystemMessage(content="BASE"),
                    HumanMessage(content="[paper_id] p1\n\n[task_context]\nBODY"),
                ]
            )
        )
