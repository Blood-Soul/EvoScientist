"""Candidate lookup for plan review.

Retrieval is lexical (`search_experience_records`, the same core behind the `search_experience` tool) and returns the
stored records with a compact descriptor for the trace. Plan review reads them in full; there is no second ranking
stage.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


from ..experiences.retrieval import search_experience_records
from ..experiences.store import list_experience_documents
from ..types import ExperienceLevel, ObservationSearchMode
from .trace import emit_trace

DEFAULT_RETRIEVE_LIMIT = 8


def _experience_record(text: str) -> dict[str, Any] | None:
    """Parse one stored experience document body."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _descriptor(record: Mapping[str, Any]) -> str:
    """Compact one-line candidate for the trace."""
    experience = record.get("experience", {})
    if not isinstance(experience, Mapping):
        experience = {}
    paper = record.get("paper", {})
    title = paper.get("title", "") if isinstance(paper, Mapping) else ""
    methods = experience.get("method")
    method = "; ".join(str(m) for m in methods[:3]) if isinstance(methods, list) else ""
    gist = experience.get("problem") or experience.get("transferable_core") or experience.get("statement") or ""
    return " | ".join(
        part
        for part in (
            f"{record.get('id', '')} [{str(experience.get('comparison') or record.get('experience_level', '')).upper()}]",
            str(experience.get("domain") or ""),
            str(experience.get("task") or ""),
            method,
            " ".join(str(gist).split())[:200],
            f"source: {' '.join(str(title).split())[:80]}",
        )
        if part and not part.endswith(("source: ",))
    )


def gather_candidates(
    *,
    memory_dir: str | Path,
    project_id: str,
    query: str,
    topic: str = "",
    method: str = "",
    discipline: str = "",
    domain: str = "",
    level: ExperienceLevel | None = None,
    limit: int = DEFAULT_RETRIEVE_LIMIT,
    call_id: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve experience candidates for one target task.

    Scoped to `E-*` records on purpose. Observations (`O-*`) are already short
    and target-neutral, so reading one directly costs little and the reuse
    layer adds nothing; this transformation exists for the long, source-bound
    records where direct injection is what fails.

    The optional facets are the same ones `search_experience` exposes. They are
    passed through rather than folded into `query` so that a caller narrowing by
    structure ("materials-science records only") does not have that constraint
    competing for lexical weight against the subject-matter terms.

    ``topic`` is the subject facet in a paper's own vocabulary, which the push
    gate writes and the pull path does not have. When it is given, ``query``
    (the concrete decision sentence) is searched as the separate ``task``
    facet; when it is not, ``query`` stands in as the topic, as before.
    """
    topic = topic.strip()
    hits = search_experience_records(
        memory_dir=memory_dir,
        project_id=project_id,
        topic=topic or query,
        method=method,
        task=query if topic else "",
        discipline=discipline,
        domain=domain,
        level=level,
        limit=limit,
        mode=ObservationSearchMode.RANKED,
    )
    ranked_ids = [hit["observation_id"] for hit in hits][:limit]
    if not ranked_ids:
        return []
    bodies = {
        document.observation_id: document.text
        for document in list_experience_documents(
            memory_dir=memory_dir, project_id=project_id
        )
    }
    candidates: list[dict[str, Any]] = []
    for experience_id in ranked_ids:
        record = _experience_record(bodies.get(experience_id, ""))
        if record is None:
            continue
        candidates.append(
            {
                "id": experience_id,
                "record": record,
                "descriptor": _descriptor(record),
            }
        )
    emit_trace(
        memory_dir,
        "retrieve",
        call_id=call_id,
        project_id=project_id,
        query=query,
        topic=topic,
        method=method,
        candidate_ids=[item["id"] for item in candidates],
        descriptors={item["id"]: item["descriptor"] for item in candidates},
    )
    return candidates


def fetch_candidates(
    *,
    memory_dir: str | Path,
    project_id: str,
    ids: list[str],
    call_id: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch named records as candidates, keeping the caller's order.

    Used when the gate has already chosen from its candidate list: the lexical
    retrieval step is skipped and only those records go on to rerank and writing.
    Unknown ids are ignored.
    """
    bodies = {
        document.observation_id: document.text
        for document in list_experience_documents(
            memory_dir=memory_dir, project_id=project_id
        )
    }
    candidates: list[dict[str, Any]] = []
    for experience_id in dict.fromkeys(ids):
        record = _experience_record(bodies.get(experience_id, ""))
        if record is None:
            continue
        candidates.append(
            {"id": experience_id, "record": record, "descriptor": _descriptor(record)}
        )
    emit_trace(
        memory_dir,
        "retrieve",
        call_id=call_id,
        project_id=project_id,
        query="[picked by the gate]",
        topic="",
        method="",
        candidate_ids=[item["id"] for item in candidates],
        descriptors={item["id"]: item["descriptor"] for item in candidates},
    )
    return candidates


__all__ = ["DEFAULT_RETRIEVE_LIMIT", "fetch_candidates", "gather_candidates"]
