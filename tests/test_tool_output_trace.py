"""The experience tools leave their full, untruncated output in the policy trace."""

from __future__ import annotations

import json

from EvoScientist.memory.policy.review import MAX_RECORD_CHARS, record_sizes
from EvoScientist.memory.policy.trace import emit_tool_output, trace_path


def _lines(memory_dir):
    return [json.loads(line) for line in trace_path(memory_dir).read_text(encoding="utf-8").splitlines()]


def test_full_output_is_recorded_with_its_length(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOSCIENTIST_POLICY_TRACE", "1")
    monkeypatch.delenv("EVOSCIENTIST_POLICY_TRACE_PATH", raising=False)
    long_text = "x" * 50_000
    emit_tool_output(tmp_path, "read_memory", {"id": "E-1"}, long_text)
    (line,) = _lines(tmp_path)
    assert line["event"] == "tool_output" and line["tool"] == "read_memory"
    assert line["chars"] == 50_000 and line["output"] == long_text and line["args"] == {"id": "E-1"}


def test_nothing_is_written_when_tracing_is_off(tmp_path, monkeypatch):
    monkeypatch.delenv("EVOSCIENTIST_POLICY_TRACE", raising=False)
    emit_tool_output(tmp_path, "search_experience", {}, "{}")
    assert not trace_path(tmp_path).exists()


def test_record_sizes_flag_what_the_reviewer_got_cut_short():
    small = {"id": "E-a", "record": {"statement": "short"}}
    big = {"id": "E-b", "record": {"statement": "y" * (MAX_RECORD_CHARS + 10)}}
    sizes = {s["id"]: s for s in record_sizes([small, big])}
    assert sizes["E-a"]["truncated"] is False and sizes["E-b"]["truncated"] is True
