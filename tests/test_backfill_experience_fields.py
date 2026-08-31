"""Tests for scripts/backfill_experience_fields.py.

The script edits the live memory store in place, and the store *is* the library
-- a half-written `l1.json` loses every record in it, and there is no upstream to
re-derive them from without re-downloading papers. So the properties worth
testing are the safety ones: never destructive, idempotent, resumable, and
per-record failures that do not take the rest of the file down with them.

Loaded by path because the script is deliberately standalone, not an installed
module.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "backfill_experience_fields.py"


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("backfill_experience_fields", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution because the module defines dataclasses, and
    # `dataclasses` resolves annotations through `sys.modules[cls.__module__]`.
    sys.modules["backfill_experience_fields"] = module
    spec.loader.exec_module(module)
    return module


bf = _load_script()


STATEMENT = (
    "The practice is to convert task feedback into written reflections and store "
    "them in an episodic buffer, rather than updating model weights."
)


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


class _Model:
    """Records how many times it was called, so idempotency is observable."""

    def __init__(self, *outputs: str) -> None:
        self._outputs = list(outputs)
        self.calls = 0

    async def ainvoke(self, messages: Any) -> _Response:
        self.calls += 1
        index = min(self.calls - 1, len(self._outputs) - 1)
        return _Response(self._outputs[index])


def _good_output(core: str = "when X, doing Y yields Z") -> str:
    return json.dumps(
        {
            "transferable_core": core,
            "bindings": [{"name": "GPT-4", "kind": "model"}],
        }
    )


def _record(record_id: str, *, statement: str = STATEMENT) -> dict[str, Any]:
    return {
        "id": record_id,
        "layer": "L1",
        "domain": "agent_learning",
        "task": "reflection",
        "statement": statement,
        "applicable_when": ["multi-trial settings"],
        "not_applicable_when": ["single-shot settings"],
        "scope": "language agents",
        "action": "store reflections",
        "effect": "higher success rate",
        "evidence": [{"source_id": "p1", "section": "experiment", "quote": "q"}],
        "confidence": 0.7,
    }


def _seed(root: Path, *, records: list[dict[str, Any]]) -> Path:
    paper = root / "experiences" / "projects" / "P-alpha" / "2303.11366-abc"
    paper.mkdir(parents=True)
    path = paper / "l1.json"
    path.write_text(
        json.dumps({"paper_id": "2303.11366", "experiences": records}),
        encoding="utf-8",
    )
    return path


def _run(root: Path, model: Any, *extra: str) -> Any:
    args = bf._parse_args(["--apply", "--memory-dir", str(root), *extra])
    import EvoScientist.EvoScientist as entry

    original = entry._ensure_auxiliary_chat_model
    entry._ensure_auxiliary_chat_model = lambda: model  # type: ignore[assignment]
    try:
        return asyncio.run(bf._run(args))
    finally:
        entry._ensure_auxiliary_chat_model = original  # type: ignore[assignment]


def test_dry_run_neither_calls_the_model_nor_writes(tmp_path: Path) -> None:
    """The script edits the live store, so it must be inspectable without risk."""
    path = _seed(tmp_path, records=[_record("l1_01")])
    before = path.read_text(encoding="utf-8")

    args = bf._parse_args(["--dry-run", "--memory-dir", str(tmp_path)])
    stats = asyncio.run(bf._run(args))

    assert stats.records_seen == 1
    assert stats.records_filled == 1
    assert path.read_text(encoding="utf-8") == before


def test_backfill_fills_only_the_two_derived_fields(tmp_path: Path) -> None:
    """Nothing else about a record may change: same statement, same evidence."""
    path = _seed(tmp_path, records=[_record("l1_01")])
    original = json.loads(path.read_text(encoding="utf-8"))

    stats = _run(tmp_path, _Model(_good_output()))

    assert stats.records_filled == 1
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert set(updated) == set(original)
    before, after = original["experiences"][0], updated["experiences"][0]
    changed = {
        key for key in set(before) | set(after) if before.get(key) != after.get(key)
    }
    assert changed == {"transferable_core", "bindings"}
    assert after["transferable_core"] == "when X, doing Y yields Z"
    assert after["bindings"] == [{"name": "GPT-4", "kind": "model"}]


def test_rerunning_is_a_no_op_that_costs_nothing(tmp_path: Path) -> None:
    """Idempotence is what makes an interrupted run resumable by re-running it."""
    path = _seed(tmp_path, records=[_record("l1_01")])
    _run(tmp_path, _Model(_good_output()))
    after_first = path.read_text(encoding="utf-8")

    model = _Model(_good_output("a different core"))
    stats = _run(tmp_path, model)

    assert model.calls == 0
    assert stats.records_already_had == 1
    assert stats.records_filled == 0
    assert path.read_text(encoding="utf-8") == after_first


def test_overwrite_regenerates_populated_fields(tmp_path: Path) -> None:
    path = _seed(tmp_path, records=[_record("l1_01")])
    _run(tmp_path, _Model(_good_output()))

    _run(tmp_path, _Model(_good_output("a regenerated core")), "--overwrite")

    item = json.loads(path.read_text(encoding="utf-8"))["experiences"][0]
    assert item["transferable_core"] == "a regenerated core"


def test_a_thin_statement_is_skipped_rather_than_filled_badly(tmp_path: Path) -> None:
    """A derived field is only as good as its input.

    Writing a core derived from a stub is worse than leaving the documented
    `statement` fallback in place, because a populated field stops the fallback
    from applying.
    """
    path = _seed(tmp_path, records=[_record("l1_01", statement="too short")])
    model = _Model(_good_output())

    stats = _run(tmp_path, model)

    assert model.calls == 0
    assert stats.records_too_thin == 1
    assert "transferable_core" not in json.loads(path.read_text())["experiences"][0]


def test_one_bad_response_does_not_lose_the_other_records(tmp_path: Path) -> None:
    """Failures are per record; the file must keep the records that succeeded."""
    path = _seed(tmp_path, records=[_record("l1_01"), _record("l1_02")])

    stats = _run(tmp_path, _Model(_good_output(), "not json at all"))

    assert stats.records_filled == 1
    assert stats.records_failed == 1
    items = json.loads(path.read_text(encoding="utf-8"))["experiences"]
    assert len(items) == 2
    assert items[0]["transferable_core"] == "when X, doing Y yields Z"
    # The failed one is untouched, so the next pass picks it up.
    assert "transferable_core" not in items[1]


def test_an_all_failing_file_is_left_byte_identical(tmp_path: Path) -> None:
    path = _seed(tmp_path, records=[_record("l1_01")])
    before = path.read_text(encoding="utf-8")

    stats = _run(tmp_path, _Model("not json at all"))

    assert stats.records_failed == 1
    assert path.read_text(encoding="utf-8") == before


def test_limit_bounds_one_invocation_so_passes_compose(tmp_path: Path) -> None:
    project = tmp_path / "experiences" / "projects" / "P-alpha"
    for index in range(3):
        paper = project / f"230{index}.00001-abc"
        paper.mkdir(parents=True)
        (paper / "l1.json").write_text(
            json.dumps({"paper_id": f"230{index}", "experiences": [_record("l1_01")]}),
            encoding="utf-8",
        )

    first = _run(tmp_path, _Model(_good_output()), "--limit", "2")
    assert first.papers_seen == 2
    assert first.records_filled == 2

    second = _run(tmp_path, _Model(_good_output()), "--limit", "2")
    # The first two are already done, so the same bound now reaches the third.
    assert second.records_already_had == 2
    assert second.records_filled == 1


def test_backup_keeps_a_copy_of_the_original(tmp_path: Path) -> None:
    path = _seed(tmp_path, records=[_record("l1_01")])
    before = path.read_text(encoding="utf-8")

    _run(tmp_path, _Model(_good_output()), "--backup")

    backup = path.with_suffix(path.suffix + ".bak")
    assert backup.read_text(encoding="utf-8") == before


def test_a_crash_mid_write_leaves_the_original_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The atomic replace is the reason a crash cannot truncate the library."""
    path = _seed(tmp_path, records=[_record("l1_01")])
    before = path.read_text(encoding="utf-8")

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(bf.os, "replace", _boom)

    with pytest.raises(OSError, match="disk full"):
        _run(tmp_path, _Model(_good_output()))

    assert path.read_text(encoding="utf-8") == before
    # And no temporary file is left lying next to it.
    assert not list(path.parent.glob(".*.tmp"))


def test_empty_core_leaves_the_statement_fallback_in_place(tmp_path: Path) -> None:
    """An untransferable claim must not get a present-but-useless field.

    `transferable_core()` falls back to `statement` only when the field is
    absent, so writing "" would silently make the descriptor empty.
    """
    path = _seed(tmp_path, records=[_record("l1_01")])

    _run(tmp_path, _Model(json.dumps({"transferable_core": "", "bindings": []})))

    item = json.loads(path.read_text(encoding="utf-8"))["experiences"][0]
    assert "transferable_core" not in item


@pytest.mark.parametrize(
    ("name", "usable"),
    [
        ("GPT-4", True),
        ("ImageNet", True),
        ("pass@1", True),
        ("91%", False),
        ("3.2%", False),
        ("80", False),
        ("up", False),
    ],
)
def test_bindings_must_be_names_not_measured_values(name: str, usable: bool) -> None:
    """A binding is matched as a substring by the A/B harness.

    `91%` identifies nothing and fires on any text containing those characters,
    which would inflate the stale-binding rate for every condition equally.
    """
    assert bf._is_usable_binding(name) is usable


def test_binding_shape_is_validated_like_a_fresh_extraction(tmp_path: Path) -> None:
    """A backfilled record must not hold a shape extraction would have rejected."""
    with pytest.raises(bf.ExperienceOutputError):
        bf._parse_response(
            json.dumps({"bindings": [{"name": "GPT-4", "kind": "invented"}]})
        )
