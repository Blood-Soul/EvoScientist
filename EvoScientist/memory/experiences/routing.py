"""Inspectable field views and rule-based facets, separate from semantic edges."""

from __future__ import annotations

import re
from typing import Any

from .candidates import _digest

INDEX_VERSION = 1
TAXONOMY_VERSION = "field-facets-v2"
VIEW_FIELDS = {
    "problem": ("task", "trigger_context"),
    "method": ("action", "transferable_core"),
    "transfer": ("transferable_core",),
    "positive": ("applicable_when",),
    "negative": ("not_applicable_when",),
    "outcome": ("effect", "practice_trace"),
}
# These are mentions useful for indexing, not assertions of causality or fit.
FACET_RULES = {
    "decision_roles": {
        "model_selection": r"model selection|select(?:ing)? (?:a |the )?(?:model|forecast)|choose|routing|selector",
        "validation_design": r"cross.validation|rolling.origin|forward.chaining|validation design",
        "uncertainty_calibration": r"coverage|prediction intervals?|recalibrat\w*|calibrat\w*.{0,35}(?:interval|predictive|probabil|quantile|uncertainty)",
        "representation": r"factorization|latent factors?|low.rank|representation",
        "preprocessing": r"normali[sz]\w*|preprocess\w*|scal(?:e|ing)",
        "resource_decision": r"reserve|capacity|dispatch|resource allocation",
    },
    "problem_patterns": {
        "short_history": r"short (?:histor\w*|series)|limited (?:histor\w*|context)|insufficient (?:histor\w*|data)|less history|data.scarce",
        "distribution_change": r"distribution\w* shift|regime chang\w*|structural breaks?|non.stationar\w*|drift",
        "miscalibration": r"undercover\w*|overconfiden\w*|miscalibrat\w*|coverage|intervals?",
        "validation_bias": r"validation.{0,30}bias|cross.validation|dependent (?:data|observations)|temporal dependence",
        "computation": r"expensive|comput\w* cost|resource.constrain\w*|CPU|latency",
        "interpretability": r"interpretab\w*|transparent|audit\w*",
    },
    "method_patterns": {
        "conditional_routing": r"rout(?:e|es|ing)|selector|selective\w*|difficult (?:instances|inputs|windows)",
        "temporal_validation": r"rolling.origin|forward.chaining|dependence.aware|block cross.validation",
        "reference_normalization": r"normali[sz]\w*.{0,100}(?:reference|baseline)|clear.sky",
        "latent_factorization": r"factorization|latent factors?|low.rank",
        "uncertainty_calibration": r"recalibrat\w*|calibrat\w*.{0,35}(?:interval|predictive|probabil|quantile|uncertainty)|(?:interval|quantile|uncertainty|distribution).{0,35}calibrat\w*",
        "residual_adjustment": r"signed residual|residual adjustments?|residual.{0,25}(?:pool|sampl|add)",
    },
}
FACET_FIELDS = {
    "decision_roles": ("task", "action"),
    "problem_patterns": ("trigger_context", "applicable_when"),
    "method_patterns": ("action", "transferable_core"),
}


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def normalize_label(value):
    """Normalize spelling only; no inferred semantic aliases."""
    return (
        re.sub(r"[\s_-]+", "_", value.strip().casefold())
        if isinstance(value, str)
        else ""
    )


def field_sources(content, fields):
    sources = []
    for field in fields:
        value = content.get(field)
        if field == "practice_trace":
            value = [
                step.get("feedback") for step in value or [] if isinstance(step, dict)
            ]
        sources.extend(
            {"field": field, "quote": text} for text in strings(value) if text.strip()
        )
    return sources


def build_routing_index(records: dict[str, dict[str, Any]], previous=None):
    """Derive views/facets with literal field citations; never mutate records."""
    config = {
        "index_version": INDEX_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "rules_hash": _digest([FACET_RULES, FACET_FIELDS, VIEW_FIELDS]),
    }
    old = previous if previous and previous.get("config") == config else {}
    entries, reused = {}, 0
    for key, record in sorted(records.items()):
        version = _digest(record)
        prior = old.get("entries", {}).get(key)
        if prior and prior.get("record_version") == version:
            entries[key] = prior
            reused += 1
            continue
        content = record["content"]
        views = {
            name: {"sources": field_sources(content, fields)}
            for name, fields in VIEW_FIELDS.items()
        }
        if not views["transfer"]["sources"]:
            views["transfer"] = {
                "sources": field_sources(content, ("action",)),
                "fallback": "action",
            }
        facets = {}
        for axis, rules in FACET_RULES.items():
            facets[axis] = []
            sources = field_sources(content, FACET_FIELDS[axis])
            for label, pattern in rules.items():
                for source in sources:
                    match = re.search(pattern, source["quote"], re.IGNORECASE)
                    if match:
                        text = source["quote"]
                        excerpt = text[max(0, match.start() - 40) : match.end() + 80]
                        facets[axis].append(
                            {
                                "value": label,
                                "basis": [{"field": source["field"], "quote": excerpt}],
                                "origin": "rule_mention",
                            }
                        )
                        break
        entries[key] = {
            "experience_id": key,
            "record_version": version,
            "discipline": normalize_label(content.get("discipline")),
            "domain": normalize_label(content.get("domain")),
            "context": {
                field: content[field]
                for field in (
                    "scope",
                    "bindings",
                    "evidence_scope",
                    "claim_type",
                    "rationale",
                    "rationale_depth",
                )
                if field in content
            },
            "field_views": views,
            "facets": facets,
            "unknown_axes": [axis for axis, values in facets.items() if not values],
        }
    return {
        "config": config,
        "entries": entries,
        "stats": {
            "records": len(entries),
            "entries_reused": reused,
            "entries_built": len(entries) - reused,
        },
    }
