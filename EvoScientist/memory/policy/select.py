"""Candidate lookup for plan review.

Retrieval is lexical (`search_experience_records`, the same core behind the `search_experience` tool) and returns the
stored records with a compact descriptor for the trace. Plan review reads them in full; there is no second ranking
stage.
"""

from __future__ import annotations

import json
import re
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


_WORD = re.compile(r"[a-z0-9]+")
# Function words and plan-template words. Without this a short query such as a
# section heading ("2. What the model sees") matched almost every record on
# "what"/"the"/"model" and scored as high as a real condition match.
_STOP = frozenset(
    """the and for with from that this these those what when where which while whose into onto over under than then
    there their them they its are was were been being have has had does did not but any all each per via how why who
    can could should would may might must will shall use used using our your one two more most less least also only
    such same other about after before between both within without whether model models data task plan build step
    steps section""".split()
)


def _tokens_of(text: Any) -> set[str]:
    return {
        w for w in _WORD.findall(" ".join(str(text or "").lower().split()))
        if len(w) > 2 and w not in _STOP
    }


def _record_experience(record: Mapping[str, Any]) -> Mapping[str, Any]:
    exp = record.get("experience") if isinstance(record, Mapping) else None
    return exp if isinstance(exp, Mapping) else {}


def _record_level(record: Mapping[str, Any]) -> str:
    exp = _record_experience(record)
    return str(record.get("experience_level") or exp.get("layer") or "").lower()


def _condition_score(query: str, record: Mapping[str, Any]) -> float:
    """How well the record's task/conditions match the query's own words.

    L2-first already narrows the level; this score decides the order inside the
    window. Task/domain match counts double; problem, requires and source
    environment count once. Method names are deliberately not scored: a query
    asks which conditions hold, not which method the record used.
    """
    exp = _record_experience(record)
    q = _tokens_of(query)
    if not q:
        return 0.0
    task = _tokens_of(f"{exp.get('task') or ''} {exp.get('domain') or ''}")
    setting_values: list[str] = []
    source = exp.get("source_setting")
    if isinstance(source, Mapping):
        for entry in source.get("environment") or []:
            if isinstance(entry, Mapping):
                setting_values.append(f"{entry.get('property') or ''} {entry.get('value') or ''}")
        for entry in source.get("params") or []:
            if isinstance(entry, Mapping):
                setting_values.append(f"{entry.get('name') or ''} {entry.get('value') or ''}")
    requires = " ".join(str(r.get("text") or "") for r in (exp.get("requires") or []) if isinstance(r, Mapping))
    cond = _tokens_of(f"{exp.get('problem') or ''} {' '.join(setting_values)} {requires}")
    return 2.0 * (len(q & task) / max(1, len(q))) + 1.0 * (len(q & cond) / max(1, len(q)))


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
    prefer_l2: bool = True,
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

    def search(level_arg: ExperienceLevel | None, n: int) -> list[dict[str, Any]]:
        return search_experience_records(
            memory_dir=memory_dir,
            project_id=project_id,
            topic=topic or query,
            method=method,
            task=query if topic else "",
            discipline=discipline,
            domain=domain,
            level=level_arg,
            limit=n,
            mode=ObservationSearchMode.RANKED,
        )

    # L2-first, but L1 keeps a share: L2 says when a finding transfers, L1
    # carries the source experiment's setting and numbers. A pure L2-first
    # fill (as in 36e0891) left no L1 at all once the library had enough L2.
    if prefer_l2 and level is None:
        n_l1 = max(1, limit // 3)
        l2_hits = search("l2", limit)
        l1_hits = search("l1", limit)
        hits = l2_hits[: limit - min(n_l1, len(l1_hits))]
        hits += l1_hits[: limit - len(hits)]
        seen = {hit["observation_id"] for hit in hits}
        if len(hits) < limit:  # one level is short: fill from the other, best first
            hits += [h for h in l2_hits + l1_hits if h["observation_id"] not in seen][: limit - len(hits)]
    else:
        hits = search(level, limit)
    hits = hits[:limit]
    ranked_ids = [hit["observation_id"] for hit in hits]
    if not ranked_ids:
        return []
    bodies = {
        document.observation_id: document.text
        for document in list_experience_documents(
            memory_dir=memory_dir, project_id=project_id
        )
    }
    candidates: list[dict[str, Any]] = []
    rank = {hit["observation_id"]: i for i, hit in enumerate(hits)}
    for experience_id in ranked_ids:
        record = _experience_record(bodies.get(experience_id, ""))
        if record is None:
            continue
        candidates.append(
            {
                "id": experience_id,
                "record": record,
                "descriptor": _descriptor(record),
                "level": _record_level(record),
                "condition": round(_condition_score(query, record), 4),
                "lexical_rank": rank[experience_id],
            }
        )
    # Inside this query's window, same-condition records go first; lexical rank
    # breaks ties. The condition score is only comparable within one query.
    candidates.sort(key=lambda c: (-c["condition"], c["lexical_rank"]))
    for position, candidate in enumerate(candidates):
        candidate["query_rank"] = position
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
