"""Shared L1/L2 → EvoMemory observation mapping.

Both importers (offline experience bank, online paper-experience promotion) use
this so the two sources stay schema-identical in the store. The first version kept
only ~6 of L1's 12 fields and ~7 of L2's 19, which silently dropped the fields the
knowledge graph and keyword search need most: `domain*` (no way to cluster by
field), `keywords` (strong TF-IDF signal), `practice_trace` (L1's defining
action→feedback chain), and `r`/`μ_r`/`r_depth` (L2's causal layer).

Layout per observation:
  summary        one indexed line (3× search weight)
  observation    narrative + the structured fields as labelled sections
  why_it_matters applicability / boundary
  evidence       provenance + verbatim quote + classifications + keywords
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def clean(value: Any) -> str:
    """Coerce a field to a stripped string; empty for None/blank."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _mu(rec: Mapping[str, Any], key: str = "μ") -> str:
    """Read a confidence field under either its Greek or ASCII key.

    The extractor emits `μ`/`μ_r` inconsistently — sometimes U+03BC, sometimes
    "mu" — so both spellings must be tried or the value is silently lost.
    """
    ascii_key = key.replace("μ", "mu")
    return clean(rec.get(key) or rec.get(ascii_key))


def _trace(value: Any) -> str:
    """Render practice_trace as numbered action→feedback pairs."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return clean(value)
    if not isinstance(value, list) or not value:
        return ""
    lines = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, Mapping):
            action = clean(item.get("action"))
            feedback = clean(item.get("feedback"))
            lines.append(f"{index}. Action: {action}\n   Feedback: {feedback}")
        else:
            lines.append(f"{index}. {clean(item)}")
    return "\n".join(lines)


def _obj(value: Any) -> dict[str, Any]:
    """Normalize a nested field that may arrive as a dict or a JSON string."""
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def _sections(pairs: list[tuple[str, str]]) -> str:
    """Join non-empty (label, body) pairs as labelled Markdown sections."""
    return "\n\n".join(f"### {label}\n{body}" for label, body in pairs if body)


def _taxonomy(rec: Mapping[str, Any]) -> str:
    """Classification lines — these drive graph clustering, so keep them all."""
    fields = (
        ("domain", "domain"),
        ("arXiv", "domain_arxiv"),
        ("Wikipedia", "domain_wikipedia"),
        ("ACM CCS", "domain_acm_ccs"),
        ("CLC", "domain_clc"),
    )
    lines = [f"{label}: {clean(rec.get(key))}" for label, key in fields if clean(rec.get(key))]
    return "\n".join(lines)


def _provenance(rec: Mapping[str, Any], paper_id: str) -> str:
    pid = clean(rec.get("_paper_id")) or clean(rec.get("paper_id")) or paper_id
    parts = [f"arXiv:{pid}"] if pid else []
    name = clean(rec.get("_paper_name"))
    if name and name != pid:
        parts.append(name)
    if clean(rec.get("_paper_group")):
        parts.append(f"group: {clean(rec.get('_paper_group'))}")
    if _v3_section(rec):
        parts.append(f"section: {_v3_section(rec)}")
    return " | ".join(parts)


def _first_sentence(text: str, limit: int = 200) -> str:
    """First sentence (or clause) of a long statement, for use as summary."""
    text = clean(text)
    if not text:
        return ""
    for sep in ("。", ". ", "! ", "? "):
        idx = text.find(sep)
        if 0 < idx < limit:
            return text[: idx + len(sep)].strip()
    return text[:limit].strip()


def _quote(rec: Mapping[str, Any]) -> str:
    """Verbatim supporting quote from v3 `evidence[]` or legacy `source_quote`.

    v3 emits ``evidence: [{source_id, section, quote}, ...]``; the old schema used
    a flat ``source_quote``. Prefer v3, join multiple quotes.
    """
    evidence = rec.get("evidence")
    if isinstance(evidence, list) and evidence:
        quotes = []
        for item in evidence:
            if isinstance(item, Mapping):
                q = clean(item.get("quote") or item.get("source_quote"))
                sec = clean(item.get("section"))
                if q:
                    quotes.append(f"[{sec}] {q}" if sec else q)
        if quotes:
            return "\n\n".join(quotes)
    return clean(rec.get("source_quote"))


def _v3_section(rec: Mapping[str, Any]) -> str:
    """Best section label for provenance, honoring v3 `evidence[].section`."""
    evidence = rec.get("evidence")
    if isinstance(evidence, list) and evidence and isinstance(evidence[0], Mapping):
        return clean(evidence[0].get("section"))
    return clean(rec.get("source_section"))


def l1_to_observation(rec: Mapping[str, Any], paper_id: str = "") -> dict | None:
    """Map one L1 practical experience. Returns None when unusable.

    Prefers v3 fields (``statement`` / ``task`` / ``applicable_when`` /
    ``evidence[]``), falling back to the legacy schema (``narrative`` / ``t`` /
    ``e`` / ``source_quote``) so the offline bank still imports.
    """
    task_obj = _obj(rec.get("t"))
    # Body: v3 `statement`, legacy `narrative`
    narrative = clean(rec.get("statement")) or clean(rec.get("narrative"))
    # Summary: legacy `t.summary`, else v3 `task`, else first sentence of statement
    summary = (
        clean(task_obj.get("summary"))
        or clean(rec.get("task"))
        or _first_sentence(narrative)
    )
    if not summary or not narrative:
        return None

    granularity = clean(rec.get("granularity"))  # legacy-only; v3 has no grain
    scope = clean(rec.get("scope"))  # v3 applicability scope, not a grain enum
    body = _sections(
        [
            ("Narrative", narrative),
            # legacy nested env; v3 puts detail in practice_trace/task
            ("Practice environment", clean(rec.get("e"))),
            ("Practice trace", _trace(rec.get("practice_trace"))),
            (
                "Task context",
                "\n".join(
                    f"{label}: {clean(task_obj.get(key))}"
                    for label, key in (
                        ("modality", "modality"),
                        ("scale", "scale"),
                        ("constraint", "constraint"),
                    )
                    if clean(task_obj.get(key))
                )
                or clean(rec.get("task")),
            ),
            ("Applicable when", clean(rec.get("applicable_when"))),
            ("Scope", scope),
            ("Utility", clean(rec.get("utility"))),
            ("Extraction rationale", clean(rec.get("extraction_rationale"))),
        ]
    )

    why_parts = []
    if granularity:
        why_parts.append(f"Granularity: {granularity}.")
    if scope:
        why_parts.append(f"Scope: {scope}")
    if clean(task_obj.get("scale")):
        why_parts.append(f"Scale: {clean(task_obj.get('scale'))}")
    boundary = clean(task_obj.get("constraint")) or clean(rec.get("applicable_when"))
    if boundary:
        why_parts.append(f"Applies under: {boundary}")

    evidence = _sections(
        [
            ("Source", _provenance(rec, paper_id)),
            ("Verbatim quote", _quote(rec)),
            ("Classification", _taxonomy(rec)),
            ("Keywords", clean(rec.get("keywords"))),
        ]
    )

    prefix = f"[{granularity}] " if granularity else ""
    return {
        "summary": f"{prefix}{summary}",
        "observation": body,
        "why_it_matters": " ".join(why_parts) or summary,
        "evidence": evidence or None,
    }


def l2_to_observation(rec: Mapping[str, Any], paper_id: str = "") -> dict | None:
    """Map one L2 inductive experience. Returns None when unusable.

    Prefers v3 fields (``statement`` / ``confidence`` / ``rationale`` /
    ``applicable_when``), falling back to legacy (``declaration`` / ``μ`` / ``r`` /
    ``context``) so the offline bank still imports.
    """
    # Declaration: v3 `statement`, legacy `declaration`
    declaration = clean(rec.get("statement")) or clean(rec.get("declaration"))
    if not declaration:
        return None
    summary = declaration if len(declaration) < 220 else _first_sentence(declaration)

    context = _obj(rec.get("context"))
    claim_type = clean(rec.get("claim_type"))
    # Confidence: v3 `confidence`, legacy `μ`/`mu`
    mu = clean(rec.get("confidence")) or _mu(rec)
    # Rationale: v3 `rationale`/`rationale_depth`, legacy `r`/`r_depth`/`μ_r`
    reason = clean(rec.get("rationale")) or clean(rec.get("r"))

    causal = ""
    if reason:
        bits = [reason]
        detail = [
            f"{label}: {value}"
            for label, value in (
                ("confidence (μ_r)", _mu(rec, "μ_r")),
                (
                    "depth",
                    clean(rec.get("rationale_depth")) or clean(rec.get("r_depth")),
                ),
                ("depth rationale", clean(rec.get("r_depth_rationale"))),
            )
            if value
        ]
        if detail:
            bits.append("\n".join(detail))
        causal = "\n\n".join(bits)

    body = _sections(
        [
            ("Declaration", declaration),
            # Only legacy records carry a separate narrative body
            ("Narrative", clean(rec.get("narrative"))),
            ("Causal explanation (rationale)", causal),
            (
                "Applicability context",
                "\n".join(
                    f"{label}: {clean(context.get(key))}"
                    for label, key in (
                        ("summary", "summary"),
                        ("modality", "modality"),
                        ("scale", "scale"),
                        ("constraint", "constraint"),
                    )
                    if clean(context.get(key))
                )
                or "\n".join(
                    f"{label}: {clean(rec.get(key))}"
                    for label, key in (
                        ("applicable when", "applicable_when"),
                        ("not applicable when", "not_applicable_when"),
                    )
                    if clean(rec.get(key))
                ),
            ),
            ("Extraction rationale", clean(rec.get("extraction_rationale"))),
        ]
    )

    why_parts = []
    if claim_type:
        why_parts.append(f"Claim type: {claim_type}.")
    if mu:
        why_parts.append(f"Confidence: {mu}.")
    boundary = clean(context.get("summary")) or clean(rec.get("applicable_when"))
    if boundary:
        why_parts.append(boundary)
    constraint = clean(context.get("constraint")) or clean(rec.get("not_applicable_when"))
    if constraint:
        why_parts.append(f"Boundary: {constraint}")

    evidence = _sections(
        [
            ("Source", _provenance(rec, paper_id)),
            ("Verbatim quote", _quote(rec)),
            ("Classification", _taxonomy(rec)),
            (
                "Keywords",
                clean(rec.get("keywords")) or clean(rec.get("keywords_summary")),
            ),
        ]
    )

    prefix = f"[{claim_type}] " if claim_type else ""
    return {
        "summary": f"{prefix}{summary}",
        "observation": body,
        "why_it_matters": " ".join(why_parts) or declaration,
        "evidence": evidence or None,
    }
