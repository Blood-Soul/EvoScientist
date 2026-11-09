"""Structured data reaches a model as readable text; the memory worker's transcript keeps who said what."""

from __future__ import annotations

from EvoScientist.memory.source_context import _trajectory_for_prompt
from EvoScientist.render_text import render_data


def test_mappings_become_key_value_lines_and_empty_values_are_dropped():
    out = render_data({"name": "x", "empty": "", "none": None, "items": [], "ok": True, "nested": {"a": 1, "b": []}})
    assert out == "name: x\nok: yes\nnested:\n  a: 1"


def test_lists_become_dash_items_and_dict_items_stay_together():
    out = render_data({"candidates": [{"id": "c1", "n": 2}, {"id": "c2"}]})
    assert out == "candidates:\n  - id: c1\n    n: 2\n  - id: c2"


def test_multiline_strings_keep_their_line_breaks():
    out = render_data({"body": "line one\nline two"})
    assert out == "body:\n  line one\n  line two"
    assert "\\n" not in out


def test_the_worker_transcript_shows_roles_calls_and_results():
    text = _trajectory_for_prompt(
        [
            {"role": "human", "content": "Fix the bug."},
            {
                "role": "ai",
                "content": "Looking.",
                "tool_calls": [
                    {"name": "read_file", "args": {"path": "utils.py"}, "id": "c1"},
                    {"name": "execute", "args": {"command": "pytest -q\ngrep x y"}, "id": "c2"},
                ],
            },
            {"role": "tool", "content": "def f(): pass", "name": "read_file", "tool_call_id": "c1", "status": "success"},
            {"role": "tool", "content": "boom", "name": "execute", "tool_call_id": "c2", "status": "error"},
        ]
    )
    assert "[human]\nFix the bug." in text
    assert "-> read_file(path='utils.py')" in text
    assert "    command:\n      pytest -q\n      grep x y" in text
    assert "[tool result (read_file)]\ndef f(): pass" in text
    assert "[tool result (execute) [error]]\nboom" in text
    assert not text.lstrip().startswith(("[{", "{"))
