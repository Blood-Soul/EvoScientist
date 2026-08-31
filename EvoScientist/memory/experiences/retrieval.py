"""Retrieval over the isolated paper-experience store.

This module used to host a merged retriever that concatenated observation
documents with experience documents and handed both to one ranker. That merge
was the root of two failures:

1. It forced experience retrieval to accept observation-shaped arguments.
   Because every `E-*` record is stored as ``SEMANTIC``/``PROJECT``, a caller
   passing ``memory_type=procedural`` (the natural pairing for a "how do I do X"
   query, which the tool schema actively encouraged) or ``scope=global`` (the
   natural pairing for "find me general methodology") silently removed the
   entire experience library from the result set, with no signal that it had
   happened.
2. It made the observation vocabulary the vocabulary of experience search, so
   callers phrased process-shaped queries ("what to do when an API call fails")
   against records that only ever hold subject-matter findings.

Experience retrieval is therefore its own entry point with its own arguments:
discipline/domain/level, never memory_type/scope. Observations keep
``observations.store.search_observation_files``. ``read_memory_file`` stays the
single ID-addressed reader for both stores, since the `O-`/`E-` prefix is
already an unambiguous route.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..search import search_documents
from ..types import (
    ExperienceLevel,
    ObservationReadResult,
    ObservationSearchDocument,
    ObservationSearchHit,
    ObservationSearchMode,
)
from .store import list_experience_documents, read_experience_file

# Reciprocal-rank-fusion constant. 60 is the value from the original RRF paper
# and is scale-free: it damps the contribution of deep ranks without depending
# on library size or on how many facets were supplied.
RRF_K = 60


def _facet_documents(
    *,
    memory_dir: str | Path,
    project_id: str,
    discipline: str = "",
    domain: str = "",
    level: ExperienceLevel | None = None,
) -> list[ObservationSearchDocument]:
    """List experience documents narrowed by the structured facets."""
    documents = list_experience_documents(memory_dir=memory_dir, project_id=project_id)
    wanted_discipline = discipline.strip().casefold()
    wanted_domain = domain.strip().casefold()
    return [
        document
        for document in documents
        if (level is None or document.experience_level == level)
        and (
            not wanted_discipline
            or (document.discipline or "").casefold() == wanted_discipline
        )
        and (not wanted_domain or (document.domain or "").casefold() == wanted_domain)
    ]


def search_experience_records(
    *,
    memory_dir: str | Path,
    project_id: str,
    topic: str = "",
    method: str = "",
    task: str = "",
    discipline: str = "",
    domain: str = "",
    level: ExperienceLevel | None = None,
    limit: int = 8,
    mode: ObservationSearchMode = ObservationSearchMode.RANKED,
) -> list[ObservationSearchHit]:
    """Search `E-*` records by subject-matter facets.

    The three text facets are searched *separately* and fused with reciprocal
    rank fusion rather than concatenated into one query string. Concatenation
    made a single low-information token decide the whole ranking: one shared
    term matched every document, every document scored identically, and the
    "ranking" degenerated into directory order while still returning a full,
    confident-looking result page. Fusing per-facet rankings means a document
    must rank well on some facet to surface, and agreement across facets is
    what promotes it.
    """
    facets = [text.strip() for text in (topic, method, task) if text.strip()]
    documents = _facet_documents(
        memory_dir=memory_dir,
        project_id=project_id,
        discipline=discipline,
        domain=domain,
        level=level,
    )
    if not documents:
        return []
    if not facets:
        # Facet-only browse through the search entry point: no text to rank by,
        # so return the filtered set in a stable order instead of nothing.
        return [_hit_from_document(document) for document in documents[:limit]]

    search_mode = ObservationSearchMode(mode)
    hits_by_facet = [
        search_documents(
            documents=documents,
            query=facet,
            limit=max(limit * 2, limit),
            mode=search_mode,
        )
        for facet in facets
    ]
    if len(hits_by_facet) == 1:
        return hits_by_facet[0][:limit]

    scores: Counter[str] = Counter()
    best: dict[str, ObservationSearchHit] = {}
    for hits in hits_by_facet:
        for rank, hit in enumerate(hits):
            record_id = hit["observation_id"]
            scores[record_id] += 1.0 / (RRF_K + rank + 1)
            if record_id not in best:
                best[record_id] = hit
    ordered = sorted(
        best.values(),
        key=lambda hit: (-scores[hit["observation_id"]], hit["observation_id"]),
    )
    fused: list[ObservationSearchHit] = []
    for hit in ordered[:limit]:
        merged = dict(hit)
        merged["score"] = round(scores[hit["observation_id"]], 5)
        fused.append(merged)  # type: ignore[arg-type]
    return fused


def _hit_from_document(document: ObservationSearchDocument) -> ObservationSearchHit:
    """Build a hit for a facet-only listing, with no relevance score."""
    hit: ObservationSearchHit = {
        "observation_id": document.observation_id,
        "path": document.path,
        "memory_type": document.memory_type,
        "scope": document.scope,
        "summary": document.summary,
        "matches": [],
    }
    hit["record_kind"] = document.record_kind
    if document.experience_level is not None:
        hit["experience_level"] = document.experience_level
    return hit


def browse_experience_facets(
    *,
    memory_dir: str | Path,
    project_id: str,
    facet: str = "discipline",
    discipline: str = "",
    domain: str = "",
    level: ExperienceLevel | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """Page through the experience library by facet instead of by keyword.

    Browsing exists because keyword retrieval is lexical: a caller who does not
    already know the library's vocabulary cannot phrase a query that reaches it,
    and for an open-ended task ("what has been found about X") there may be no
    single right phrasing at all. Facets let the caller narrow by structure and
    read titles, which is phrasing-independent.

    Paged and counted at every level on purpose -- ``facet="domain"`` is free
    text and unbounded, so it is ordered by record count and truncated rather
    than returned whole.
    """
    documents = _facet_documents(
        memory_dir=memory_dir,
        project_id=project_id,
        discipline=discipline,
        domain=domain,
        level=level,
    )
    total = len(documents)
    window = slice(max(offset, 0), max(offset, 0) + max(limit, 1))

    if facet in {"discipline", "domain"}:
        counts: Counter[str] = Counter(
            (getattr(document, facet) or "unknown") for document in documents
        )
        # Count first, then name: highest-count buckets are the ones worth
        # drilling into, and the tail is reachable through offset.
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        page = ranked[window]
        return {
            "facet": facet,
            "total_records": total,
            "total_values": len(ranked),
            "offset": max(offset, 0),
            "values": [{"value": value, "records": count} for value, count in page],
        }

    ordered = sorted(documents, key=lambda document: document.observation_id)
    page = ordered[window]
    return {
        "facet": "records",
        "total_records": total,
        "offset": max(offset, 0),
        "records": [
            {
                "id": document.observation_id,
                "level": document.experience_level,
                "discipline": document.discipline,
                "domain": document.domain,
                "task": document.task,
                "summary": document.summary,
            }
            for document in page
        ],
    }


def experience_library_stats(
    *, memory_dir: str | Path, project_id: str, top_domains: int = 8
) -> dict[str, Any]:
    """Summarize the library for the prompt-facing index block.

    Returns counts and top facet values, never a per-record listing: the block
    has to stay a fixed size as the library grows.
    """
    documents = list_experience_documents(memory_dir=memory_dir, project_id=project_id)
    levels: Counter[str] = Counter()
    disciplines: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    papers: set[str] = set()
    for document in documents:
        levels[document.experience_level or "l1"] += 1
        disciplines[(document.discipline or "unknown")] += 1
        domains[(document.domain or "unknown")] += 1
        if document.paper_key:
            papers.add(document.paper_key)
    return {
        "records": len(documents),
        "papers": len(papers),
        "levels": dict(levels),
        "disciplines": sorted(
            disciplines.items(), key=lambda item: (-item[1], item[0])
        ),
        "top_domains": sorted(domains.items(), key=lambda item: (-item[1], item[0]))[
            :top_domains
        ],
    }


def read_memory_file(
    *, memory_dir: str | Path, project_id: str, record_id: str
) -> ObservationReadResult | None:
    """Read either an observation or a project experience by stable ID."""
    from ..observations.store import read_observation_file

    requested = record_id.strip()
    if requested.startswith("E-"):
        return read_experience_file(
            memory_dir=memory_dir,
            project_id=project_id,
            experience_id=requested,
        )
    return read_observation_file(
        memory_dir=memory_dir,
        project_id=project_id,
        observation_id=requested,
    )


__all__ = [
    "browse_experience_facets",
    "experience_library_stats",
    "read_memory_file",
    "search_experience_records",
]
