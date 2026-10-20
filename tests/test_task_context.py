"""Task context rendering and its effect on the extraction message."""

from __future__ import annotations

import json

import pytest

from EvoScientist.memory.experiences.extraction import build_extraction_message
from EvoScientist.memory.experiences.task_context import (
    TASK_CONTEXT_BLOCK_LABEL,
    TaskContext,
    TaskContextError,
)


def test_no_context_message_is_byte_identical_to_the_bench_blind_form():
    """The already-extracted corpus must stay comparable: no context, no change."""
    message = build_extraction_message(paper_id="p1", paper_text="BODY")
    assert message == "[paper_id] p1\n\nBODY"


def test_context_block_precedes_the_paper():
    context = TaskContext(task="forecast a fixed horizon")
    message = build_extraction_message(
        paper_id="p1", paper_text="BODY", task_context=context
    )
    assert message.index(TASK_CONTEXT_BLOCK_LABEL) < message.index("BODY")
    assert message.startswith("[paper_id] p1")


def test_empty_fields_are_omitted_rather_than_rendered_blank():
    context = TaskContext(task="t")
    payload = context.to_dict()
    assert payload == {"task": "t"}


def test_render_emits_parsable_json_under_the_label():
    context = TaskContext(
        task="t",
        objective="mae",
        direction="minimize",
        data_characteristics=("short history",),
        reference_points={"sota": 576.35},
        bottlenecks=("flat forecasts win by default",),
        extraction_focus=("when to extrapolate",),
    )
    block = context.render()
    assert block.startswith(f"[{TASK_CONTEXT_BLOCK_LABEL}]\n```json\n")
    body = block.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
    assert json.loads(body) == {
        "task": "t",
        "objective": "mae",
        "direction": "minimize",
        "data_characteristics": ["short history"],
        "bottlenecks": ["flat forecasts win by default"],
        "extraction_focus": ["when to extrapolate"],
        "reference_points": {"sota": 576.35},
    }


def test_blank_entries_inside_a_list_are_dropped():
    context = TaskContext(task="t", bottlenecks=("real", "   ", ""))
    assert context.to_dict()["bottlenecks"] == ["real"]


def test_task_is_required():
    with pytest.raises(TaskContextError, match="task must be a non-empty string"):
        TaskContext(task="  ")


def test_direction_is_a_closed_vocabulary():
    with pytest.raises(TaskContextError, match="direction must be one of"):
        TaskContext(task="t", objective="mae", direction="lower")


def test_direction_without_objective_is_rejected():
    with pytest.raises(TaskContextError, match="direction requires objective"):
        TaskContext(task="t", direction="minimize")


def test_from_mapping_rejects_unknown_fields():
    """A typo in a bench's context file must fail loudly, not be ignored."""
    with pytest.raises(TaskContextError, match="unknown task context fields: focus"):
        TaskContext.from_mapping({"task": "t", "focus": ["x"]})


def test_from_mapping_round_trips_to_dict():
    payload = {
        "task": "t",
        "objective": "mae",
        "direction": "minimize",
        "data_characteristics": ["a"],
        "reference_points": {"sota": 1},
        "bottlenecks": ["b"],
        "extraction_focus": ["c"],
    }
    assert TaskContext.from_mapping(payload).to_dict() == payload


def test_from_mapping_rejects_a_string_where_a_list_belongs():
    with pytest.raises(TaskContextError, match="bottlenecks must be an array"):
        TaskContext.from_mapping({"task": "t", "bottlenecks": "not a list"})


def test_from_json_file_reads_a_bench_definition(tmp_path):
    path = tmp_path / "bench.json"
    path.write_text(json.dumps({"task": "t", "bottlenecks": ["b"]}), encoding="utf-8")
    context = TaskContext.from_json_file(path)
    assert context.task == "t"
    assert context.bottlenecks == ("b",)


def test_from_json_file_reports_malformed_json(tmp_path):
    path = tmp_path / "bench.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(TaskContextError, match="not valid JSON"):
        TaskContext.from_json_file(path)


def test_from_json_file_reports_a_missing_file(tmp_path):
    with pytest.raises(TaskContextError, match="cannot read task context"):
        TaskContext.from_json_file(tmp_path / "absent.json")


def test_shipped_example_is_valid():
    """The solar example doubles as the reference for writing a new bench file."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    path = root / "prompt" / "task_context.examples" / "solar-weekly-mae.json"
    context = TaskContext.from_json_file(path)
    assert context.direction == "minimize"
    assert context.reference_points
    assert context.extraction_focus
