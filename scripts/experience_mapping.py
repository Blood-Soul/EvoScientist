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
    if clean(rec.get("source_section")):
        parts.append(f"section: {clean(rec.get('source_section'))}")
    return " | ".join(parts)


def l1_to_observation(rec: Mapping[str, Any], paper_id: str = "") -> dict | None:
    """Map one L1 practical experience. Returns None when unusable."""
    task = _obj(rec.get("t"))
    summary = clean(task.get("summary"))
    narrative = clean(rec.get("narrative"))
    if not summary or not narrative:
        return None

    granularity = clean(rec.get("granularity"))
    body = _sections(
        [
            ("Narrative", narrative),
            ("Practice environment", clean(rec.get("e"))),
            ("Practice trace", _trace(rec.get("practice_trace"))),
            (
                "Task context",
                "\n".join(
                    f"{label}: {clean(task.get(key))}"
                    for label, key in (
                        ("modality", "modality"),
                        ("scale", "scale"),
                        ("constraint", "constraint"),
                    )
                    if clean(task.get(key))
                ),
            ),
            ("Extraction rationale", clean(rec.get("extraction_rationale"))),
        ]
    )

    why_parts = []
    if granularity:
        why_parts.append(f"Granularity: {granularity}.")
    if clean(task.get("scale")):
        why_parts.append(f"Scale: {clean(task.get('scale'))}")
    if clean(task.get("constraint")):
        why_parts.append(f"Applies under: {clean(task.get('constraint'))}")

    evidence = _sections(
        [
            ("Source", _provenance(rec, paper_id)),
            ("Verbatim quote", clean(rec.get("source_quote"))),
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
    """Map one L2 inductive experience. Returns None when unusable."""
    declaration = clean(rec.get("declaration"))
    narrative = clean(rec.get("narrative"))
    if not declaration or not narrative:
        return None

    context = _obj(rec.get("context"))
    claim_type = clean(rec.get("claim_type"))
    mu = _mu(rec)
    reason = clean(rec.get("r"))

    causal = ""
    if reason:
        bits = [reason]
        detail = [
            f"{label}: {value}"
            for label, value in (
                ("confidence (μ_r)", _mu(rec, "μ_r")),
                ("depth", clean(rec.get("r_depth"))),
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
            ("Narrative", narrative),
            ("Causal explanation (r)", causal),
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
    if clean(context.get("summary")):
        why_parts.append(clean(context.get("summary")))
    if clean(context.get("constraint")):
        why_parts.append(f"Applies under: {clean(context.get('constraint'))}")

    evidence = _sections(
        [
            ("Source", _provenance(rec, paper_id)),
            ("Verbatim quote", clean(rec.get("source_quote"))),
            ("Classification", _taxonomy(rec)),
            (
                "Keywords",
                clean(rec.get("keywords")) or clean(rec.get("keywords_summary")),
            ),
        ]
    )

    prefix = f"[{claim_type}] " if claim_type else ""
    return {
        "summary": f"{prefix}{declaration}",
        "observation": body,
        "why_it_matters": " ".join(why_parts) or declaration,
        "evidence": evidence or None,
    }
