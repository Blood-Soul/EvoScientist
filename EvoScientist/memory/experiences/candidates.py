"""Sparse lexical candidate nomination; similarity is not a relation verdict."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from typing import Any

from ..search import _tokens

_STOP = frozenset(
    "the and for with from that this these those into using use used when where "
    "which then than are was were have has had can could should would may might "
    "not only also such each both more most other their there its our your "
    "under over between across through based before after within without "
    "rather instead include including consider ensure compare evaluate select "
    "apply check report paper study result results method methods model models "
    "data dataset datasets task tasks forecasting forecast forecasts".split()
)
_FIELDS = (
    "task",
    "statement",
    "action",
    "trigger_context",
    "applicable_when",
    "not_applicable_when",
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_text(item) for item in value.values())
    return ""


def nominate_candidates(
    records: dict[str, dict[str, Any]],
    *,
    neighbors=3,
    recall_pool=20,
    min_similarity=0.12,
    min_shared_terms=2,
    max_document_frequency=0.6,
    diversity_ratio=0.8,
    previous=None,
):
    """Return per-node nominations and reusable state, using an inverted index.

    Each source nominates at most K destinations. Unchanged sources retain
    eligible nominations; new/changed sources and invalidated neighbors refill
    from a bounded shortlist. Incoming nominations do not count against K.
    Scores and gates are lexical heuristics, not calibrated probabilities.
    """
    if neighbors < 1 or recall_pool < neighbors:
        raise ValueError("recall_pool must be at least neighbors, both positive")
    if not 0 <= min_similarity <= 1 or min_shared_terms < 1:
        raise ValueError("invalid similarity or shared-term threshold")
    if not 0 < max_document_frequency <= 1 or not 0 <= diversity_ratio <= 1:
        raise ValueError("invalid document-frequency or diversity threshold")
    config = {
        "version": 1,
        "neighbors": neighbors,
        "recall_pool": recall_pool,
        "min_similarity": min_similarity,
        "min_shared_terms": min_shared_terms,
        "max_document_frequency": max_document_frequency,
        "diversity_ratio": diversity_ratio,
    }
    versions = {key: _digest(value) for key, value in records.items()}
    tokens = {
        key: set(
            _tokens(" ".join(_text(record["content"].get(field)) for field in _FIELDS))
        )
        - _STOP
        for key, record in records.items()
    }
    frequency = Counter(token for values in tokens.values() for token in values)
    # Do not discard every shared term in a tiny library.
    if len(records) > 5:
        tokens = {
            key: {
                t
                for t in values
                if frequency[t] / len(records) <= max_document_frequency
            }
            for key, values in tokens.items()
        }
    weights = {
        token: (math.log((len(records) + 1) / (count + 1)) + 1) ** 2
        for token, count in frequency.items()
    }
    norms = {
        key: math.sqrt(sum(weights[t] for t in values))
        for key, values in tokens.items()
    }
    postings = defaultdict(list)
    for key, values in tokens.items():
        for token in sorted(values):
            postings[token].append(key)

    def score(source, target):
        common = tokens[source] & tokens[target]
        similarity = (
            sum(weights[t] for t in common) / (norms[source] * norms[target])
            if norms[source] and norms[target]
            else 0
        )
        return similarity, len(common)

    def eligible(source, target):
        similarity, shared = score(source, target)
        return similarity >= min_similarity and shared >= min_shared_terms

    old = previous if previous and previous.get("config") == config else {}
    nominations = {}
    reused_sources = 0
    for source in sorted(records):
        retained = []
        old_choices = old.get("nominations", {}).get(source, [])
        unchanged = old.get("record_versions", {}).get(source) == versions[source]
        if unchanged:
            for target in old_choices:
                if (
                    target in records
                    and target != source
                    and old.get("record_versions", {}).get(target) == versions[target]
                    and eligible(source, target)
                ):
                    retained.append(target)
        # Empty nominations also cache a completed search. New nodes nominate
        # into the old library; unchanged nodes are not exhaustively reranked.
        if unchanged and len(retained) == len(old_choices):
            nominations[source] = retained
            reused_sources += 1
            continue
        dots = defaultdict(float)
        shared_counts = Counter()
        for token in sorted(tokens[source]):
            for target in postings[token]:
                if target != source:
                    dots[target] += weights[token]
                    shared_counts[target] += 1
        ranked = heapq.nsmallest(
            recall_pool,
            (
                (-dot / (norms[source] * norms[target]), target)
                for target, dot in dots.items()
                if norms[source]
                and norms[target]
                and shared_counts[target] >= min_shared_terms
                and dot / (norms[source] * norms[target]) >= min_similarity
            ),
        )
        choices = retained + [target for _, target in ranked if target not in retained]
        selected = choices[:neighbors]
        if selected and all(
            records[target]["paper_key"] == records[source]["paper_key"]
            for target in selected
        ):
            cross = next(
                (
                    target
                    for _, target in ranked
                    if records[target]["paper_key"] != records[source]["paper_key"]
                ),
                None,
            )
            if (
                cross is not None
                and score(source, cross)[0]
                >= diversity_ratio * score(source, selected[-1])[0]
            ):
                selected[-1] = cross
        nominations[source] = selected
    state = {"config": config, "record_versions": versions, "nominations": nominations}
    pairs = {}
    for source, targets in nominations.items():
        for target in targets:
            pair = tuple(sorted((source, target)))
            similarity, shared = score(*pair)
            pairs[pair] = {"similarity": round(similarity, 6), "shared_terms": shared}
    covered = {key for pair in pairs for key in pair}
    stats = {
        "records": len(records),
        "nominated_pairs": len(pairs),
        "covered_nodes": len(covered),
        "isolated_ids": sorted(set(records) - covered),
        "sources_reused": reused_sources,
        "sources_searched": len(records) - reused_sources,
    }
    return pairs, state, stats
