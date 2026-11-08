"""The usage ledger records every LLM call the model factory builds."""

from __future__ import annotations

import json
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from EvoScientist.llm import get_chat_model
from EvoScientist.llm.usage_ledger import (
    LEDGER_ENV,
    UsageLedgerHandler,
    attach_usage_ledger,
    summarize_ledger,
)


def _usage_message(**details) -> AIMessage:
    usage = {
        "input_tokens": 100,
        "output_tokens": 5,
        "total_tokens": 105,
        "input_token_details": {"cache_read": 80, "cache_creation": 10},
    }
    return AIMessage(content="hi", usage_metadata=usage)


def _records(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_ledger_is_off_without_the_environment_variable(monkeypatch):
    monkeypatch.delenv(LEDGER_ENV, raising=False)
    model = GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))
    attach_usage_ledger(model, provider="anthropic", model_name="fake")
    assert model.callbacks in (None, [])


def test_attach_records_tokens_and_cache(monkeypatch, tmp_path):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv(LEDGER_ENV, str(path))
    model = GenericFakeChatModel(messages=iter([_usage_message()]))
    attach_usage_ledger(model, provider="anthropic", model_name="fake-model")

    model.invoke("hello")

    records = [r for r in _records(path) if r["kind"] == "usage"]
    assert len(records) == 1
    record = records[0]
    assert record["provider"] == "anthropic"
    assert record["model"] == "fake-model"
    assert record["input_tokens"] == 100
    assert record["output_tokens"] == 5
    assert record["cache_read"] == 80
    assert record["cache_creation"] == 10
    assert record["cache_hit"] is True
    assert "error" not in record
    assert "test_usage_ledger" in record["source"]


def test_async_calls_are_recorded_too(monkeypatch, tmp_path):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv(LEDGER_ENV, str(path))
    model = GenericFakeChatModel(messages=iter([_usage_message()]))
    attach_usage_ledger(model, provider="anthropic", model_name="fake-model")

    import asyncio

    asyncio.run(model.ainvoke("hello"))

    records = [r for r in _records(path) if r["kind"] == "usage"]
    assert len(records) == 1
    assert records[0]["input_tokens"] == 100


def test_errors_are_recorded_without_usage(monkeypatch, tmp_path):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv(LEDGER_ENV, str(path))
    handler = UsageLedgerHandler(path, provider="anthropic", model="fake-model")

    handler.on_llm_error(RuntimeError("boom"), run_id=uuid4())

    records = [r for r in _records(path) if r["kind"] == "usage"]
    assert records[0]["error"] == "RuntimeError"
    assert "boom" in records[0]["error_message"]
    assert records[0]["input_tokens"] == 0


def test_unwritable_ledger_never_breaks_the_call(monkeypatch, tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv(LEDGER_ENV, str(blocker / "ledger.jsonl"))
    model = GenericFakeChatModel(messages=iter([_usage_message()]))
    attach_usage_ledger(model, provider="anthropic", model_name="fake-model")

    assert model.invoke("hello").content == "hi"


def test_get_chat_model_attaches_the_handler(monkeypatch, tmp_path):
    monkeypatch.setenv(LEDGER_ENV, str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    model = get_chat_model("claude-sonnet-5-5")

    assert any(isinstance(cb, UsageLedgerHandler) for cb in model.callbacks or [])


def test_summarize_ledger_aggregates_sources_and_cache(tmp_path):
    path = tmp_path / "ledger.jsonl"
    rows = [
        {"kind": "ledger_start", "pid": 1, "cwd": "/tmp"},
        {
            "kind": "usage",
            "langgraph_node": "agent",
            "model": "m",
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_read": 30,
            "cache_creation": 0,
            "cache_hit": True,
        },
        {
            "kind": "usage",
            "langgraph_node": "coach",
            "model": "m",
            "input_tokens": 20,
            "output_tokens": 4,
            "cache_read": 0,
            "cache_creation": 0,
            "cache_hit": False,
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    summary = summarize_ledger(path)

    assert summary["calls"] == 2
    assert summary["input_tokens"] == 30
    assert summary["output_tokens"] == 6
    assert summary["cache_read"] == 30
    assert summary["calls_with_cache_hit"] == 1
    assert summary["cache_hit_rate"] == 0.5
    assert summary["prompt_tokens_anthropic_style"] == 60
    assert summary["by_source"]["agent"]["calls"] == 1
    assert summary["by_source"]["coach"]["calls"] == 1
