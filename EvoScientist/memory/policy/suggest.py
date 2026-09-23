"""Render a derived policy as the guidance injected at the current step.

The pull path returns the policy report as a JSON tool result and lets the agent
read it. The push path has no tool result to land in, so the policy has to be
rendered as text and injected directly. That difference is worth being explicit
about: what gets injected here is **guidance**, not retrieved records. The
records themselves never reach the agent's context on this path -- a policy is
already the target-bound rewrite, so rendering it as prose adds no model call and
loses nothing.

The rendering is deliberately lossy in one direction only. `procedure`, `rebind`,
`checks`, `conflicts` and `unsupported` are kept because each one changes what
the agent should do next. `sources` is kept as bare IDs so a line can still be
audited through `read_memory`. `preconditions` and `declines` are kept but capped.
What is dropped is everything that would make this block grow with library size.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SUGGESTION_OPEN = "<experience_guidance>"
SUGGESTION_CLOSE = "</experience_guidance>"

# Caps exist so one step's guidance cannot crowd out the trajectory it is meant
# to inform. A policy longer than this is a sign the gate asked too broad a
# question, not a reason to inject more.
MAX_ITEMS = 6
MAX_ITEM_CHARS = 300


def _flatten(value: Any, limit: int = MAX_ITEM_CHARS) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _bullets(values: Any, *, indent: str = "- ") -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        return []
    lines: list[str] = []
    for item in list(values)[:MAX_ITEMS]:
        text = _flatten(item)
        if text:
            lines.append(f"{indent}{text}")
    return lines


def _rebind_lines(values: Any) -> list[str]:
    """Render the rebind list, which is the load-bearing part of the guidance.

    ``source_value`` is rendered last and labelled as provenance. It is the one
    field that deliberately carries a source-side number, and the whole point of
    the layer is that it is an anchor, never the answer -- so the label travels
    with it rather than being left to the reader's memory.
    """
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        return []
    lines: list[str] = []
    for item in list(values)[:MAX_ITEMS]:
        if not isinstance(item, Mapping):
            continue
        name = _flatten(item.get("name") or "", 120)
        if not name:
            continue
        kind = _flatten(item.get("kind") or "", 40)
        head = f"- {name}" + (f" ({kind})" if kind else "")
        obtain = _flatten(item.get("how_to_obtain") or "", 240)
        source = _flatten(item.get("source_value") or "", 120)
        detail = []
        if obtain:
            detail.append(f"derive it here: {obtain}")
        if source:
            detail.append(f"source-side value, provenance only: {source}")
        lines.append(head + (" — " + "; ".join(detail) if detail else ""))
    return lines


def _conflict_lines(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        return []
    lines: list[str] = []
    for item in list(values)[:MAX_ITEMS]:
        if isinstance(item, Mapping):
            issue = _flatten(item.get("issue") or item.get("summary") or "", 200)
            decide = _flatten(item.get("decide_by") or item.get("condition") or "", 200)
            if not issue:
                continue
            lines.append(f"- {issue}" + (f" — decide on: {decide}" if decide else ""))
        else:
            text = _flatten(item)
            if text:
                lines.append(f"- {text}")
    return lines


def render_suggestion(report: Mapping[str, Any]) -> str:
    """Render one `derive_policy` report as an injectable guidance block.

    Returns ``""`` when there is nothing actionable to say. An empty return is
    the normal outcome for ``no_candidates``, ``no_reusable_memory``, and for a
    ``decline`` verdict carrying no conflicts or gaps: the coach then injects
    nothing at all, which is cheaper and less misleading than telling the agent
    that memory had no opinion.
    """
    policy = report.get("policy")
    if not isinstance(policy, Mapping):
        return ""

    verdict = str(policy.get("verdict") or "").strip().lower()
    sections: list[str] = []

    procedure = _bullets(policy.get("procedure"))
    rebind = _rebind_lines(policy.get("rebind"))
    preconditions = _bullets(policy.get("preconditions"))
    declines = _bullets(policy.get("declines"))
    checks = _bullets(policy.get("checks"))
    conflicts = _conflict_lines(policy.get("conflicts"))
    unsupported = _bullets(policy.get("unsupported"))

    if verdict == "decline" and not (conflicts or unsupported or declines):
        return ""
    if not (procedure or rebind or checks or conflicts or unsupported):
        return ""

    if procedure:
        sections.append("What transfers to this step:\n" + "\n".join(procedure))
    if rebind:
        sections.append(
            "Values you must re-derive here rather than copy:\n" + "\n".join(rebind)
        )
    if preconditions:
        sections.append("This only holds if:\n" + "\n".join(preconditions))
    if declines:
        sections.append("What does not transfer:\n" + "\n".join(declines))
    if conflicts:
        sections.append(
            "Papers disagree here — resolve on the condition, do not average:\n"
            + "\n".join(conflicts)
        )
    if checks:
        sections.append("Verify before treating this as settled:\n" + "\n".join(checks))
    if unsupported:
        sections.append(
            "Memory does not cover these — use live search or `search_paper_text`:\n"
            + "\n".join(unsupported)
        )

    sources = policy.get("sources")
    source_ids = [
        str(item.get("id") if isinstance(item, Mapping) else item).strip()
        for item in (sources if isinstance(sources, Sequence) else [])
        if not isinstance(sources, str)
    ]
    source_ids = [sid for sid in source_ids if sid.startswith("E-")][:MAX_ITEMS]
    if not source_ids:
        selected = report.get("selected")
        if isinstance(selected, Sequence) and not isinstance(selected, str):
            source_ids = [str(item) for item in list(selected)[:MAX_ITEMS]]

    header = (
        "This project's stored paper experience bears on the step you are about "
        "to take. This is guidance derived for this decision, not a retrieved "
        f"record; the verdict is `{verdict or 'adapt'}`."
    )
    footer_parts = [
        "Act on this where it fits and ignore it where it does not — it is "
        "advisory, and nothing here overrides what you have observed in this "
        "project."
    ]
    if source_ids:
        footer_parts.append(
            "Audit any line with `read_memory` on: " + ", ".join(source_ids) + "."
        )

    return "\n\n".join(
        [SUGGESTION_OPEN, header, *sections, " ".join(footer_parts), SUGGESTION_CLOSE]
    )


__all__ = [
    "SUGGESTION_CLOSE",
    "SUGGESTION_OPEN",
    "render_suggestion",
]
