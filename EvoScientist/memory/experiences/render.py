"""Experience records as text an agent reads, not as JSON.

Stored records are JSON because search ranks over them and code reads their fields. What the agent sees is built here:
labelled sections in a fixed order, with each field explained where it appears (what "controlled" means, how a knob's
value was chosen), so no legend has to be remembered. Three outputs share this module: one full record (`read_memory`),
a ranked list of hits (`search_experience`) and a library overview (`list_experience`); plan review renders its records
the same way.

Two shapes of record exist. The current one (`problem`, `method`, `source_setting`, `effect` as an object, `comparison`)
is rendered in full. Records extracted before it (`statement`, `applicable_when`, ...) are still in older libraries and
are rendered with the fields they have, so nothing stored becomes unreadable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

COMPARISON_MEANING = {
    "controlled": "controlled comparison: one factor was changed and everything else held fixed",
    "vs_baseline": "compared with other methods or earlier results; no single factor isolated",
    "absolute_only": "absolute result only; nothing was compared",
}

BASIS_MEANING = {
    "swept": "several values were tried and one picked",
    "derived": "computed from a measurement or rule the paper states",
    "prior_work": "taken from earlier work or a standard recipe",
    "default": "a library or framework default",
}

LEVEL_MEANING = {
    "l1": "one practice the paper tested",
    "l2": "a general finding drawn from the paper",
}


def _s(value: Any) -> str:
    return " ".join(str(value if value is not None else "").split())


def _clip(text: str, limit: int) -> str:
    text = _s(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str | bytes) else []


def is_current(item: Mapping[str, Any]) -> bool:
    """True for a current-shape record, either level (L1 carries `source_setting`, L2 `compared_options`)."""
    if not isinstance(item.get("effect"), Mapping) and "compared_options" not in item:
        return False
    return "problem" in item and ("source_setting" in item or "compared_options" in item)


def is_l2_current(item: Mapping[str, Any]) -> bool:
    return "compared_options" in item and "source_setting" not in item


def _paper_line(paper: Mapping[str, Any] | None) -> str:
    if not isinstance(paper, Mapping):
        return ""
    title = _clip(paper.get("title") or "", 90)
    pid = _s(paper.get("paper_id"))
    label = f"{title} ({pid})" if title and pid else title or pid
    if not label:
        return ""
    tail = " · full text stored: use search_paper_text" if paper.get("full_text_available") else ""
    return f"{label}{tail}"


def _method_line(method: Any) -> str:
    names = [n for n in (_s(m) for m in _items(method)) if n]
    if not names:
        return _s(method)
    head = names[0]
    return head if len(names) == 1 else f"{head}; also called: {', '.join(names[1:])}"


def _knob_line(knob: Mapping[str, Any]) -> str:
    name, value = _s(knob.get("name")), _s(knob.get("value"))
    basis, origin = _s(knob.get("basis")), _s(knob.get("from"))
    # New records omit the key when the paper does not say how the value was chosen; older records still carry
    # `"basis": "unstated"`. Both mean the same thing, and neither earns a bracket in the rendered text.
    if basis in {"", "unstated"}:
        return f"{name} = {value}"
    why = BASIS_MEANING.get(basis, basis)
    if basis == "swept":
        detail = f"swept, tried: {origin}" if origin else f"swept ({why})"
    elif basis == "derived":
        detail = f"derived from: {origin}" if origin else f"derived ({why})"
    elif basis == "prior_work":
        detail = f"from prior work: {origin}" if origin else f"from prior work ({why})"
    elif basis == "default":
        detail = f"default ({why})"
    else:
        detail = why or "basis not given"
    return f"{name} = {value}  [{detail}]"


def _render_current(record_id: str, item: Mapping[str, Any], paper: Mapping[str, Any] | None) -> str:
    comparison = _s(item.get("comparison"))
    effect = item.get("effect") if isinstance(item.get("effect"), Mapping) else {}
    setting = item.get("source_setting") if isinstance(item.get("source_setting"), Mapping) else {}
    lines = [f"{record_id} · {COMPARISON_MEANING.get(comparison, comparison or 'comparison not stated')}"]
    task, domain = _s(item.get("task")), _s(item.get("domain"))
    lines.append(f"Task: {task}" + (f"  [{domain}]" if domain else ""))
    if _s(item.get("problem")):
        lines.append(f"Problem it answers: {_s(item.get('problem'))}")
    if item.get("method"):
        lines.append(f"Method: {_method_line(item.get('method'))}")
    if _s(item.get("action")):
        lines.append(f"What to do: {_s(item.get('action'))}")

    lines.append("Result:")
    if _s(effect.get("compared")):
        lines.append(f"  - Compared: {_s(effect.get('compared'))}")
    if _s(effect.get("benefit")):
        lines.append(f"  - Benefit: {_s(effect.get('benefit'))}")
    if _s(effect.get("cost")) and _s(effect.get("cost")).lower() not in {"none", "null"}:
        lines.append(f"  - Cost: {_s(effect.get('cost'))}")

    requires = [r for r in _items(item.get("requires")) if isinstance(r, Mapping) and _s(r.get("text"))]
    if requires:
        lines.append("Needs (what the method requires; '(inferred)' = not stated by the paper):")
        for r in requires:
            lines.append(f"  - {_s(r.get('text'))}" + (" (inferred)" if _s(r.get("basis")) == "inferred" else ""))

    env = [e for e in _items(setting.get("environment")) if isinstance(e, Mapping) and _s(e.get("property"))]
    knobs = [k for k in _items(setting.get("params")) if isinstance(k, Mapping) and _s(k.get("name"))]
    if env or knobs:
        lines.append("Setting in the source experiment (compare it with your own before trusting the result):")
        for e in env:
            lines.append(f"  - {_s(e.get('property'))}: {_s(e.get('value'))}")
        if knobs:
            lines.append("  Values the authors chose (the bracket says how each was chosen):")
            for k in knobs:
                lines.append(f"  - {_knob_line(k)}")

    # `limits_stated.quote` is stored for audit but deliberately not rendered: the agent acts on the authors' stated
    # limit, not on a verbatim excerpt.
    limits = [x for x in _items(item.get("limits_stated")) if isinstance(x, Mapping) and _s(x.get("text"))]
    if limits:
        lines.append("Limits the authors themselves state:")
        for x in limits:
            lines.append(f"  - {_s(x.get('text'))}")

    where = _s(item.get("locator"))
    source = _paper_line(paper)
    if where or source:
        lines.append("Source: " + " · ".join(p for p in (source, f"see {where}" if where else "") if p))
    return "\n".join(lines)


def _render_current_l2(record_id: str, item: Mapping[str, Any], paper: Mapping[str, Any] | None) -> str:
    """An L2 finding: a mechanism-level claim, plus the options and conditions that make it checkable."""
    origin = _s(item.get("origin"))
    label = (
        "L2 finding · induced from this paper's own experimental records"
        if origin == "induced"
        else "L2 finding · mechanism-level claim drawn from the paper"
    )
    lines = [f"{record_id} · {label}"]
    task, domain = _s(item.get("task")), _s(item.get("domain"))
    lines.append(f"Task: {task}" + (f"  [{domain}]" if domain else ""))
    if _s(item.get("problem")):
        lines.append(f"Problem it answers: {_s(item.get('problem'))}")
    if item.get("method"):
        lines.append(f"Method: {_method_line(item.get('method'))}")
    if _s(item.get("action")):
        lines.append(f"What to do: {_s(item.get('action'))}")

    options = [o for o in _items(item.get("compared_options")) if isinstance(o, Mapping) and _s(o.get("option"))]
    if options:
        lines.append("Options compared:")
        for option in options:
            tail = f"  [{_s(option.get('from'))}]" if _s(option.get("from")) else ""
            result = _s(option.get("result"))
            lines.append(f"  - {_s(option.get('option'))}" + (f": {result}" if result else "") + tail)

    if _s(item.get("mechanism")):
        lines.append(f"Why it works: {_s(item.get('mechanism'))}")

    requires = [r for r in _items(item.get("requires")) if isinstance(r, Mapping) and _s(r.get("text"))]
    if requires:
        lines.append("Holds when (what must be true for the finding to apply):")
        for r in requires:
            lines.append(f"  - {_s(r.get('text'))}" + (" (inferred)" if _s(r.get("basis")) == "inferred" else ""))

    limits = [x for x in _items(item.get("limits_stated")) if isinstance(x, Mapping) and _s(x.get("text"))]
    if limits:
        lines.append("Stops holding when:")
        for x in limits:
            lines.append(f"  - {_s(x.get('text'))}" + (" (inferred)" if _s(x.get("basis")) == "inferred" else ""))

    source = _paper_line(paper)
    where = _s(item.get("locator"))
    # `origin` labels the finding above; `source_l1_ids` stays as an audit trail and is shown only as provenance.
    cited = [c for c in _items(item.get("source_l1_ids")) if _s(c)]
    parts = [p for p in (source, f"see {where}" if where else "") if p]
    if cited:
        parts.append("drawn from " + ", ".join(_s(c) for c in cited))
    if parts:
        lines.append("Source: " + " · ".join(parts))
    return "\n".join(lines)


def _render_legacy(record_id: str, level: str, item: Mapping[str, Any], paper: Mapping[str, Any] | None) -> str:
    lines = [f"{record_id} · {LEVEL_MEANING.get(level, level)}"]
    task, domain = _s(item.get("task")), _s(item.get("domain"))
    lines.append(f"Task: {task}" + (f"  [{domain}]" if domain else ""))
    if _s(item.get("trigger_context")):
        lines.append(f"Situation that called for it: {_s(item.get('trigger_context'))}")
    if _s(item.get("statement")):
        lines.append(f"Summary: {_s(item.get('statement'))}")
    if _s(item.get("action")):
        lines.append(f"What to do: {_s(item.get('action'))}")
    if _s(item.get("effect")):
        lines.append(f"Result: {_s(item.get('effect'))}")
    for key, label in (("applicable_when", "Applies when"), ("not_applicable_when", "Does not apply when")):
        values = [v for v in (_s(x) for x in _items(item.get(key))) if v]
        if values:
            lines.append(f"{label} (the extractor's judgement, not the paper's):")
            lines.extend(f"  - {v}" for v in values)
    if _s(item.get("scope")):
        lines.append(f"Scope: {_s(item.get('scope'))}")
    scope = item.get("evidence_scope") if isinstance(item.get("evidence_scope"), Mapping) else {}
    if scope:
        ablation = "yes" if scope.get("has_ablation") else "no"
        lines.append(f"How widely tested in the paper: {_s(scope.get('verification_strength')) or 'not stated'} (ablation: {ablation})")
    quotes = [e for e in _items(item.get("evidence")) if isinstance(e, Mapping) and _s(e.get("quote"))]
    for e in quotes[:2]:
        lines.append(f"Evidence ({_s(e.get('section'))}): \"{_clip(e.get('quote'), 320)}\"")
    source = _paper_line(paper)
    if source:
        lines.append(f"Source: {source}")
    return "\n".join(lines)


def render_record(document: Mapping[str, Any] | str) -> str:
    """One stored record, as the stored document dict or its JSON text, in readable form."""
    if isinstance(document, str):
        try:
            document = json.loads(document)
        except ValueError:
            return document
    if not isinstance(document, Mapping):
        return str(document)
    item = document.get("experience") if isinstance(document.get("experience"), Mapping) else document
    record_id = _s(document.get("id") or item.get("id"))
    paper = document.get("paper") if isinstance(document.get("paper"), Mapping) else None
    if is_l2_current(item):
        return _render_current_l2(record_id, item, paper)
    if is_current(item):
        return _render_current(record_id, item, paper)
    return _render_legacy(record_id, _s(document.get("experience_level") or item.get("layer") or ""), item, paper)


def render_hit(index: int, document: Mapping[str, Any], *, score: Any = None, matches: Sequence[Any] = ()) -> str:
    """One search result: enough to decide whether to read the record."""
    item = document.get("experience") if isinstance(document.get("experience"), Mapping) else document
    record_id = _s(document.get("id") or item.get("id"))
    paper = document.get("paper") if isinstance(document.get("paper"), Mapping) else None
    head = f"{index}. {record_id}"
    if is_l2_current(item):
        label = "L2 induced" if _s(item.get("origin")) == "induced" else "L2"
        lines = [f"{head} [{label}] {_s(item.get('task'))}"]
        if _s(item.get("problem")):
            lines.append(f"   Problem: {_clip(item.get('problem'), 180)}")
        if _s(item.get("action")):
            lines.append(f"   Claim: {_clip(item.get('action'), 220)}")
        if _s(item.get("mechanism")):
            lines.append(f"   Why: {_clip(item.get('mechanism'), 160)}")
    elif is_current(item):
        effect = item.get("effect") if isinstance(item.get("effect"), Mapping) else {}
        label = _s(item.get("comparison"))
        lines = [f"{head} [{label}] {_s(item.get('task'))}"]
        if _s(item.get("problem")):
            lines.append(f"   Problem: {_clip(item.get('problem'), 180)}")
        if item.get("method"):
            lines.append(f"   Method: {_clip(_method_line(item.get('method')), 180)}")
        if _s(effect.get("benefit")):
            lines.append(f"   Benefit: {_clip(effect.get('benefit'), 200)}")
        if _s(effect.get("cost")) and _s(effect.get("cost")).lower() not in {"none", "null"}:
            lines.append(f"   Cost: {_clip(effect.get('cost'), 160)}")
    else:
        level = _s(document.get("experience_level") or item.get("layer") or "").upper()
        lines = [f"{head} [{level}] {_s(item.get('task'))}"]
        core = _s(item.get("transferable_core") or item.get("statement"))
        if core:
            lines.append(f"   {_clip(core, 220)}")
    source = _paper_line(paper)
    if source:
        lines.append(f"   Paper: {_clip(source, 150)}")
    return "\n".join(lines)


def render_search(
    entries: Sequence[tuple[Mapping[str, Any] | None, Mapping[str, Any]]],
    *,
    notes: Sequence[str] = (),
) -> str:
    """Ranked hits as text. Each entry is (stored document or None, raw hit)."""
    if not entries:
        return "No experience matched.\n" + "\n".join(notes)
    blocks = []
    for index, (document, hit) in enumerate(entries, 1):
        if document is None:
            blocks.append(f"{index}. {_s(hit.get('observation_id'))} {_clip(hit.get('summary') or '', 200)}")
        else:
            blocks.append(render_hit(index, document, score=hit.get("score"), matches=hit.get("matches") or ()))
    out = [f"{len(entries)} experience record(s), best match first:", ""] + blocks
    out += ["", "Read one in full with read_memory(<id>); it explains each field where it appears."]
    out += list(notes)
    return "\n".join(out)


def render_browse(result: Mapping[str, Any]) -> str:
    """The library overview from `browse_experience_facets`, as text."""
    total = result.get("total_records", 0)
    facet = _s(result.get("facet"))
    if facet in {"discipline", "domain"}:
        values = result.get("values") or []
        shown = ", ".join(f"{_s(v.get('value'))} ({v.get('records')})" for v in values if isinstance(v, Mapping))
        more = result.get("total_values", len(values)) - result.get("offset", 0) - len(values)
        out = [f"Experience library: {total} records. By {facet}: {shown or 'none'}."]
        if more > 0:
            out.append(f"{more} more {facet} values: call again with offset={result.get('offset', 0) + len(values)}.")
        return "\n".join(out)
    rows = result.get("records") or []
    out = [f"Experience library: {total} records; showing {len(rows)} from offset {result.get('offset', 0)}."]
    for r in rows:
        if isinstance(r, Mapping):
            out.append(f"- {_s(r.get('id'))} [{_s(r.get('level')).upper()}] {_s(r.get('task'))}")
    return "\n".join(out)


__all__ = [
    "BASIS_MEANING",
    "COMPARISON_MEANING",
    "is_current",
    "render_browse",
    "render_hit",
    "render_record",
    "render_search",
]
