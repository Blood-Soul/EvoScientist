"""Bench-agnostic task context for task-oriented experience extraction.

Extraction is bench-blind by default: the prompts see a paper and nothing
else, so they record what the *paper* emphasises. When the library is being
built for one specific benchmark, that is the wrong emphasis -- the same paper
yields different useful experience depending on what the downstream agent is
trying to beat.

`TaskContext` is the formal description of that downstream task, deliberately
written in bench-independent terms: no field names a dataset, a metric
implementation, or a harness. Switching benchmarks means constructing a
different instance, never editing extraction logic.

Passing a context is optional everywhere. With no context the rendered block is
absent and the model receives byte-identical input to the bench-blind path,
which is what keeps the already-extracted corpus comparable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Rendered as a fenced block with this label so the prompts can refer to it by
# name, and so a paper that happens to contain similar prose cannot be confused
# for it.
TASK_CONTEXT_BLOCK_LABEL = "task_context"

_OBJECTIVE_DIRECTIONS = ("minimize", "maximize")


class TaskContextError(ValueError):
    """Raised when a supplied task context cannot be used as written."""


@dataclass(frozen=True, slots=True)
class TaskContext:
    """What the downstream agent is being asked to do, and what blocks it.

    Only `task` is required. Every other field narrows the extraction further;
    an empty one is simply omitted from the rendered block rather than sent as
    a blank line, so a sparse context never implies "this is known to be empty".
    """

    # What the agent must produce, in task terms rather than dataset terms
    # ("forecast a fixed horizon per univariate series", not "solar_weekly").
    task: str

    # How the attempt is judged, and which direction is better. `objective`
    # names the quantity; `direction` is one of `_OBJECTIVE_DIRECTIONS`.
    objective: str = ""
    direction: str = ""

    # Properties of the input that bear on method choice: scale, length,
    # dimensionality, noise, how much history is available.
    data_characteristics: tuple[str, ...] = ()

    # Reference points on the objective. Free-form labels mapped to values, e.g.
    # {"human SOTA": 576.35, "best constant rule": 875.64}. This is the field
    # that stops extraction from optimising a sub-ceiling strategy: knowing the
    # simple family tops out well short of SOTA redirects extraction toward what
    # would clear it.
    reference_points: Mapping[str, Any] = field(default_factory=dict)

    # What currently limits performance, stated as an obstacle rather than a
    # fix, mirroring how `trigger_context` is phrased in the record schema.
    bottlenecks: tuple[str, ...] = ()

    # What this extraction pass should bias toward, when the operator already
    # knows. Kept separate from `bottlenecks` because a bottleneck is an
    # observation while this is an instruction.
    extraction_focus: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise TaskContextError("task must be a non-empty string")
        if self.direction and self.direction not in _OBJECTIVE_DIRECTIONS:
            raise TaskContextError(
                "direction must be one of "
                + ", ".join(_OBJECTIVE_DIRECTIONS)
                + f"; got {self.direction!r}"
            )
        if self.direction and not self.objective.strip():
            raise TaskContextError("direction requires objective to be set")
        if not isinstance(self.reference_points, Mapping):
            raise TaskContextError("reference_points must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        """Serialise the non-empty fields, in the order the block renders them."""
        payload: dict[str, Any] = {"task": self.task.strip()}
        if self.objective.strip():
            payload["objective"] = self.objective.strip()
            if self.direction:
                payload["direction"] = self.direction
        for name, value in (
            ("data_characteristics", self.data_characteristics),
            ("bottlenecks", self.bottlenecks),
            ("extraction_focus", self.extraction_focus),
        ):
            entries = [str(entry).strip() for entry in value if str(entry).strip()]
            if entries:
                payload[name] = entries
        if self.reference_points:
            payload["reference_points"] = dict(self.reference_points)
        return payload

    def render(self) -> str:
        """Render the block appended to the human message.

        JSON rather than prose: the model must not mistake any of this for the
        paper's own text, and a fenced labelled object is the same shape the
        output contract already uses.
        """
        body = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        return f"[{TASK_CONTEXT_BLOCK_LABEL}]\n```json\n{body}\n```"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TaskContext:
        """Build a context from a plain mapping, as loaded from JSON."""
        if not isinstance(payload, Mapping):
            raise TaskContextError("task context must be a JSON object")
        unknown = sorted(set(payload) - _FIELD_NAMES)
        if unknown:
            raise TaskContextError("unknown task context fields: " + ", ".join(unknown))
        task = payload.get("task")
        if not isinstance(task, str):
            raise TaskContextError("task must be a string")
        return cls(
            task=task,
            objective=_as_str(payload.get("objective"), "objective"),
            direction=_as_str(payload.get("direction"), "direction"),
            data_characteristics=_as_tuple(
                payload.get("data_characteristics"), "data_characteristics"
            ),
            reference_points=_as_mapping(payload.get("reference_points")),
            bottlenecks=_as_tuple(payload.get("bottlenecks"), "bottlenecks"),
            extraction_focus=_as_tuple(
                payload.get("extraction_focus"), "extraction_focus"
            ),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> TaskContext:
        """Load a context from a JSON file, so a bench is a file not a code edit."""
        resolved = Path(path).expanduser()
        try:
            raw = resolved.read_text(encoding="utf-8")
        except OSError as exc:
            raise TaskContextError(f"cannot read task context: {exc}") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TaskContextError(
                f"task context is not valid JSON: {exc.msg}"
            ) from exc
        return cls.from_mapping(payload)


_FIELD_NAMES = {
    "task",
    "objective",
    "direction",
    "data_characteristics",
    "reference_points",
    "bottlenecks",
    "extraction_focus",
}


def _as_str(value: Any, name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TaskContextError(f"{name} must be a string")
    return value


def _as_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TaskContextError(f"{name} must be an array of strings")
    if not all(isinstance(entry, str) for entry in value):
        raise TaskContextError(f"{name} must contain only strings")
    return tuple(value)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TaskContextError("reference_points must be an object")
    return dict(value)


__all__ = [
    "TASK_CONTEXT_BLOCK_LABEL",
    "TaskContext",
    "TaskContextError",
]
