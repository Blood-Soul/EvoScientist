"""Observation memory tool results as text an agent reads, not as JSON.

An observation is already a short markdown note, so `read_memory` hands it over as written and lists what it links to.
Search results become a numbered list; the two write tools answer in one sentence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _s(value: Any) -> str:
    return " ".join(str(value if value is not None else "").split())


def _clip(text: str, limit: int) -> str:
    text = _s(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _kind(value: Any) -> str:
    return _s(getattr(value, "value", value))


def render_observation_hits(results: Sequence[Mapping[str, Any]]) -> str:
    """Ranked hits from `search_observations`."""
    if not results:
        return "No observation matched."
    lines = [f"{len(results)} observation(s), best match first:", ""]
    for i, r in enumerate(results, 1):
        tag = " · ".join(p for p in (_kind(r.get("memory_type")), _kind(r.get("scope"))) if p)
        lines.append(f"{i}. {_s(r.get('observation_id'))}" + (f" [{tag}]" if tag else "") + f" {_clip(r.get('summary') or '', 200)}")
        for snippet in list(r.get("matches") or [])[:2]:
            if _s(snippet):
                lines.append(f"   … {_clip(snippet, 220)}")
        related = [x for x in (r.get("related_observations") or []) if isinstance(x, Mapping)]
        if related:
            lines.append("   linked: " + ", ".join(_s(x.get("observation_id")) for x in related[:4]))
    lines += ["", "Read one in full with read_memory(<id>)."]
    return "\n".join(lines)


def render_observation(result: Mapping[str, Any]) -> str:
    """One observation note as written, followed by the notes it is linked to."""
    out = [str(result.get("text") or "").rstrip()]
    related = [x for x in (result.get("related_observations") or []) if isinstance(x, Mapping)]
    if related:
        out += ["", "Linked observations:"]
        for x in related:
            relation = _kind(x.get("relation"))
            why = _s(x.get("reason"))
            out.append(
                f"  - {_s(x.get('observation_id'))}" + (f" ({relation})" if relation else "") + f" {_clip(x.get('summary') or '', 140)}"
                + (f" — {_clip(why, 120)}" if why else "")
            )
    return "\n".join(out)


def render_recorded(result: Mapping[str, Any]) -> str:
    """Answer of `record_observation`."""
    oid = _s(result.get("observation_id"))
    note = _s(result.get("note"))
    if result.get("created"):
        return f"Recorded {oid} ({_kind(result.get('memory_type'))}, {_kind(result.get('scope'))})." + (f" {note}" if note else "")
    if oid:
        return f"Nothing new recorded: it matches existing {oid}." + (f" {note}" if note else "")
    return "Not recorded." + (f" {note}" if note else "")


def render_linked(result: Mapping[str, Any]) -> str:
    """Answer of `link_observations`."""
    src, dst, rel = _s(result.get("source_observation_id")), _s(result.get("target_observation_id")), _s(result.get("relation"))
    if result.get("linked"):
        updated = ", ".join(_s(x) for x in result.get("updated_observation_ids") or [])
        return f"Linked {src} -> {dst} ({rel})." + (f" Updated: {updated}." if updated else "")
    missing = ", ".join(_s(x) for x in result.get("missing_observation_ids") or [])
    why = _s(result.get("note") or result.get("error"))
    return f"Not linked {src} -> {dst}." + (f" Missing: {missing}." if missing else "") + (f" {why}" if why else "")


def render_error(message: str) -> str:
    return f"Error: {_s(message)}"


__all__ = ["render_error", "render_linked", "render_observation", "render_observation_hits", "render_recorded"]
