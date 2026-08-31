"""Target-bound reuse policy: the object that replaces raw experience injection.

An `E-*` experience record is *source-bound*: its `statement` runs to ~2.5k
characters describing what one paper's authors did, on their datasets, with
their models and their numbers. Injecting that into an acting context is the
"Full Trajectory" condition from the QCR study -- it delivers the procedure
and the obsolete bindings together, and the actor copies both.

A policy is the same knowledge rewritten *against the current task*: the
procedure that still transfers, the values that must be re-derived rather than
copied, the preconditions that make it valid, and the checks that close it out.
Declining reuse is a first-class outcome, not a failure.

Three fields extend the QCR schema for research reuse:
- `conflicts`: QCR selects exactly one memory, so disagreement cannot arise.
  Research memory routinely holds papers that contradict each other, and a
  writer that silently picks one hides the most useful signal there is.
- `unsupported`: what memory has nothing to say about, so the caller sends
  that part to live search instead of assuming it was covered.
- `sources`: every line stays auditable back to an `E-*` record.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

VERDICTS = ("adopt", "adapt", "decline")

# Kinds of source-fixed value a policy may ask the caller to re-derive. Open
# enough to describe a real paper, closed enough that the A/B harness can group
# stale-binding hits by kind.
BINDING_KINDS = (
    "dataset",
    "model",
    "scale",
    "hyperparam",
    "baseline",
    "metric",
    "toolchain",
    "other",
)

_LIST_FIELDS = (
    "procedure",
    "rebind",
    "preconditions",
    "declines",
    "checks",
    "conflicts",
    "unsupported",
    "sources",
)

_REBIND_REQUIRED = ("name", "why_bound", "how_to_obtain")
_CONFLICT_REQUIRED = ("between", "disagreement", "discriminator")


class PolicyOutputError(ValueError):
    """Raised when the policy writer's JSON does not match this schema."""


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list):
        raise PolicyOutputError(f"policy {field} must be an array")
    cleaned = [" ".join(item.split()) for item in value if isinstance(item, str)]
    return [item for item in cleaned if item]


def _rebind_list(value: Any) -> list[dict[str, str]]:
    """Normalize the anti-copy field, the one part of the schema that carries
    a source value on purpose -- labelled as provenance, never as an answer."""
    if not isinstance(value, list):
        raise PolicyOutputError("policy rebind must be an array")
    rows: list[dict[str, str]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        row = {key: str(raw.get(key) or "").strip() for key in _REBIND_REQUIRED}
        if not row["name"] or not row["how_to_obtain"]:
            # A binding the caller cannot act on is noise; drop it rather than
            # failing the whole policy over one malformed row.
            continue
        kind = str(raw.get("kind") or "other").strip().casefold()
        row["kind"] = kind if kind in BINDING_KINDS else "other"
        row["source_value"] = str(raw.get("source_value") or "").strip()
        rows.append(row)
    return rows


def _conflict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise PolicyOutputError("policy conflicts must be an array")
    rows: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        between = raw.get("between")
        row = {
            "between": [str(item).strip() for item in between if str(item).strip()]
            if isinstance(between, list)
            else [],
            "disagreement": str(raw.get("disagreement") or "").strip(),
            "discriminator": str(raw.get("discriminator") or "").strip(),
        }
        if row["disagreement"]:
            rows.append(row)
    return rows


def normalize_policy(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one policy object written by the model.

    Deliberately lenient about individual malformed rows and strict about
    shape: a reuse policy that loses one binding row is still worth returning,
    while one whose `verdict` or `procedure` cannot be read is not.
    """
    verdict = str(payload.get("verdict") or "").strip().casefold()
    if verdict not in VERDICTS:
        raise PolicyOutputError(
            f"policy verdict must be one of {VERDICTS}, got {verdict!r}"
        )
    normalized: dict[str, Any] = {"verdict": verdict}
    for field in _LIST_FIELDS:
        raw = payload.get(field, [])
        if field == "rebind":
            normalized[field] = _rebind_list(raw)
        elif field == "conflicts":
            normalized[field] = _conflict_list(raw)
        else:
            normalized[field] = _string_list(raw, field=field)
    reason = payload.get("reason")
    normalized["reason"] = " ".join(str(reason).split()) if reason else ""
    if verdict != "decline" and not normalized["procedure"]:
        raise PolicyOutputError(
            "policy procedure cannot be empty unless verdict is decline"
        )
    if verdict == "decline" and not (normalized["reason"] or normalized["declines"]):
        # A bare "no" teaches the caller nothing and cannot be audited later.
        raise PolicyOutputError("a declining policy must say what blocks reuse")
    return normalized


__all__ = [
    "BINDING_KINDS",
    "VERDICTS",
    "PolicyOutputError",
    "normalize_policy",
]
