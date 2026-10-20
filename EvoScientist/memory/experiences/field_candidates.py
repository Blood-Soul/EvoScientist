"""Bounded candidates from role-specific field matches, never asserted edges."""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict

from ..search import _tokens
from .candidates import _STOP, _digest, nominate_candidates
from .routing import build_routing_index

ROUTES = {
    "shared_problem": (("problem", "problem"),),
    "shared_method": (("method", "method"),),
    "boundary": (("method", "negative"), ("positive", "negative")),
    "prerequisite": (("outcome", "positive"),),
}
HYPOTHESES = {
    "shared_problem": ["alternative_to", "complements", "overlaps_with"],
    "shared_method": ["overlaps_with", "specializes"],
    "boundary": ["qualifies", "contradicts"],
    "prerequisite": ["complements", "depends_on"],
    "transfer": ["complements", "overlaps_with"],
    "lexical_fallback": [],
}


def nominate_field_candidates(
    records,
    *,
    neighbors=3,
    recall_pool=20,
    min_similarity=0.12,
    min_shared_terms=2,
    max_document_frequency=0.6,
    diversity_ratio=0.8,
    previous=None,
    enabled_routes=None,
):
    """Fuse per-route ranks under ONE per-source K, with literal field reasons.

    A routing change rebuilds candidates. Unchanged corpora reuse nominations;
    library changes currently refresh all route rankings to avoid stale fusion.
    Per-record classification is incremental, pair judgments are cached by graph.
    """
    routes = set(HYPOTHESES) if enabled_routes is None else set(enabled_routes)
    if not routes <= set(HYPOTHESES):
        raise ValueError("unknown candidate route")
    # Reuse the baseline's input validation, even if its fallback is disabled.
    if neighbors < 1 or recall_pool < neighbors:
        raise ValueError("recall_pool must be at least neighbors, both positive")
    if not 0 <= min_similarity <= 1 or min_shared_terms < 1:
        raise ValueError("invalid similarity or shared-term threshold")
    if not 0 < max_document_frequency <= 1 or not 0 <= diversity_ratio <= 1:
        raise ValueError("invalid document-frequency or diversity threshold")
    config = {
        "version": 1,
        "strategy": "field",
        "neighbors": neighbors,
        "recall_pool": recall_pool,
        "min_similarity": min_similarity,
        "min_shared_terms": min_shared_terms,
        "max_document_frequency": max_document_frequency,
        "diversity_ratio": diversity_ratio,
        "routes": sorted(routes),
        "route_definitions_hash": _digest(ROUTES),
    }
    index = build_routing_index(records, (previous or {}).get("routing_index"))
    versions = {key: _digest(value) for key, value in records.items()}
    old = previous if previous and previous.get("config") == config else {}
    if (
        old
        and old.get("record_versions") == versions
        and old.get("routing_index", {}).get("config") == index["config"]
    ):
        pairs = {tuple(row["ids"]): row["evidence"] for row in old["pairs"]}
        state = {**old, "routing_index": index}
        stats = _stats(records, pairs, 0, index)
        return pairs, state, stats

    entries = index["entries"]
    raw_tokens = {
        (key, view): set(
            _tokens(" ".join(source["quote"] for source in data["sources"]))
        )
        - _STOP
        for key, entry in entries.items()
        for view, data in entry["field_views"].items()
    }
    document_tokens = defaultdict(set)
    for (key, _), values in raw_tokens.items():
        document_tokens[key].update(values)
    frequency = Counter(
        token for values in document_tokens.values() for token in values
    )
    tokens = {
        key: {
            token
            for token in values
            if len(records) <= 5
            or frequency[token] / len(records) <= max_document_frequency
        }
        for key, values in raw_tokens.items()
    }
    weights = {
        token: (math.log((len(records) + 1) / (count + 1)) + 1) ** 2
        for token, count in frequency.items()
    }
    norms = {
        key: math.sqrt(sum(weights[token] for token in sorted(values)))
        for key, values in tokens.items()
    }
    postings = defaultdict(set)
    for (key, view), values in tokens.items():
        for token in values:
            postings[(view, token)].add(key)

    def basis(key, view, common):
        for source in entries[key]["field_views"][view]["sources"]:
            words = set(_tokens(source["quote"]))
            if common & words:
                # Keep exact text, not a synthetic explanation.
                return {"id": key, **source}
        return None

    per_source = {key: defaultdict(dict) for key in records}
    for source in sorted(records):
        for route, mappings in ROUTES.items():
            if route not in routes:
                continue
            for left, right in mappings:
                source_tokens = tokens[(source, left)]
                dots, counts = defaultdict(float), Counter()
                for token in sorted(source_tokens):
                    for target in sorted(postings[(right, token)]):
                        if source != target:
                            dots[target] += weights[token]
                            counts[target] += 1
                ranked = heapq.nsmallest(
                    recall_pool,
                    (
                        (
                            -dot / (norms[(source, left)] * norms[(target, right)]),
                            target,
                        )
                        for target, dot in dots.items()
                        if norms[(source, left)]
                        and norms[(target, right)]
                        and counts[target] >= min_shared_terms
                        and dot / (norms[(source, left)] * norms[(target, right)])
                        >= min_similarity
                    ),
                )
                for negative_score, target in ranked:
                    common = source_tokens & tokens[(target, right)]
                    row = {
                        "source": source,
                        "target": target,
                        "route": route,
                        "similarity": round(-negative_score, 6),
                        "shared_terms": sorted(common),
                        "field_matches": [
                            basis(source, left, common),
                            basis(target, right, common),
                        ],
                        "left_view": left,
                        "right_view": right,
                        "condition_state": "unknown",
                        "relation_hypotheses": HYPOTHESES[route],
                    }
                    existing = per_source[source][route].get(target)
                    if existing is None or row["similarity"] > existing["similarity"]:
                        per_source[source][route][target] = row

    if "transfer" in routes:
        facet_postings = defaultdict(set)
        for key, entry in entries.items():
            for axis, values in entry["facets"].items():
                for value in values:
                    facet_postings[(axis, value["value"])].add(key)
        for source, entry in entries.items():
            matches = defaultdict(lambda: defaultdict(list))
            for axis, values in entry["facets"].items():
                for facet in values:
                    for target in facet_postings[(axis, facet["value"])]:
                        if source != target:
                            matches[target][axis].append(facet["value"])
            eligible_targets = [
                target
                for target, axes in matches.items()
                if len(axes) >= 2
                and "method_patterns" in axes
                and entry["domain"]
                and entries[target]["domain"]
                and entry["domain"] != entries[target]["domain"]
                and records[source]["paper_key"] != records[target]["paper_key"]
                and entry["field_views"]["transfer"]["sources"]
                and entries[target]["field_views"]["transfer"]["sources"]
            ]
            shortlist = heapq.nsmallest(
                recall_pool,
                eligible_targets,
                key=lambda target: (
                    -len(matches[target]),
                    -len(tokens[(source, "transfer")] & tokens[(target, "transfer")]),
                    target,
                ),
            )
            for target in shortlist:
                axes = matches[target]
                other = entries[target]
                # Label difference is recorded, never called verified transfer.
                if (
                    len(axes) < 2
                    or "method_patterns" not in axes
                    or not entry["domain"]
                    or not other["domain"]
                    or entry["domain"] == other["domain"]
                    or records[source]["paper_key"] == records[target]["paper_key"]
                    or not entry["field_views"]["transfer"]["sources"]
                    or not other["field_views"]["transfer"]["sources"]
                ):
                    continue
                per_source[source]["transfer"][target] = {
                    "source": source,
                    "target": target,
                    "route": "transfer",
                    "similarity": None,
                    "facet_axes": dict(axes),
                    "field_matches": [
                        {
                            "id": source,
                            **entry["field_views"]["transfer"]["sources"][0],
                        },
                        {
                            "id": target,
                            **other["field_views"]["transfer"]["sources"][0],
                        },
                    ],
                    "domain_difference": {
                        "left": entry["domain"],
                        "right": other["domain"],
                        "status": "label_difference_only",
                    },
                    "condition_state": "unknown",
                    "relation_hypotheses": HYPOTHESES["transfer"],
                    "unresolved": [
                        "Rule mentions do not establish structural equivalence or target applicability."
                    ],
                }

    if "lexical_fallback" in routes:
        baseline, _, _ = nominate_candidates(
            records,
            neighbors=neighbors,
            recall_pool=recall_pool,
            min_similarity=min_similarity,
            min_shared_terms=min_shared_terms,
            max_document_frequency=max_document_frequency,
            diversity_ratio=diversity_ratio,
        )
        for (a, b), evidence in baseline.items():
            for source, target in ((a, b), (b, a)):
                per_source[source]["lexical_fallback"][target] = {
                    "source": source,
                    "target": target,
                    "route": "lexical_fallback",
                    **evidence,
                    "condition_state": "unknown",
                    "relation_hypotheses": [],
                }

    pairs, nominations = {}, {}
    for source, route_rows in per_source.items():
        scores, reasons = defaultdict(float), defaultdict(list)
        for route, rows in sorted(route_rows.items()):
            ranked = sorted(
                rows,
                key=lambda target: (
                    -len(rows[target].get("facet_axes", {}))
                    if route == "transfer"
                    else -(rows[target].get("similarity") or 0),
                    target,
                ),
            )[:recall_pool]
            for rank, target in enumerate(ranked, 1):
                scores[target] += 1 / (60 + rank)
                reasons[target].append(rows[target])
        selected = sorted(scores, key=lambda target: (-scores[target], target))[
            :neighbors
        ]
        nominations[source] = selected
        for target in selected:
            pair = tuple(sorted((source, target)))
            row = pairs.setdefault(
                pair,
                {
                    "similarity": 0,
                    "ranking_kind": "rrf",
                    "routes": [],
                    "matches": [],
                    "status": "pending",
                },
            )
            row["similarity"] = max(row["similarity"], round(scores[target], 8))
            row["routes"] = sorted(
                set(row["routes"]) | {reason["route"] for reason in reasons[target]}
            )
            row["matches"].extend(reasons[target])
    state = {
        "config": config,
        "record_versions": versions,
        "routing_index": index,
        "nominations": nominations,
        "pairs": [
            {"ids": list(pair), "evidence": value}
            for pair, value in sorted(pairs.items())
        ],
    }
    return pairs, state, _stats(records, pairs, len(records), index)


def _stats(records, pairs, searched, index):
    covered = {key for pair in pairs for key in pair}
    return {
        "records": len(records),
        "nominated_pairs": len(pairs),
        "covered_nodes": len(covered),
        "isolated_ids": sorted(set(records) - covered),
        "sources_searched": searched,
        "sources_reused": len(records) - searched,
        "index": index["stats"],
        "route_pair_counts": dict(
            Counter(
                route for evidence in pairs.values() for route in evidence["routes"]
            )
        ),
    }
