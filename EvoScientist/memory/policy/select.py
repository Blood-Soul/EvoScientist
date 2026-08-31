"""Two-tier selection: retrieve candidates, rerank to a small working set.

A target task arrives with no annotation. The retriever indexes show only
compact locators. We cannot read 98 full experience records (500K characters)
in search of the relevant handful; that is what the working set delivers.

Step 1 (retrieve): TF-IDF against the summary/domain/scope via
`search_experience_records`, the same experience-only retrieval core that backs
the `search_experience` tool; returns `E-*` IDs plus match snippets. It replaced
a call into the old merged retriever that passed `scope=project` and
`memory_type=semantic` to keep observations out -- filter values that had to
match the experience store's hardcoded pair exactly, so the reuse layer's
candidate supply depended on an incidental coupling between two stores.

Step 2 (rerank): a lightweight aux-model prompt reading compact *descriptors*
(id, level, confidence, core statement, domain, scope -- 200-300 chars each)
along with the target task, selecting 3-5 records and stating the reason. If
the rerank parse fails, the fallback is the top scoring candidates from step 1.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...utils import format_message_content
from ..experiences.retrieval import search_experience_records
from ..experiences.store import list_experience_documents
from ..types import ExperienceLevel, ObservationSearchMode
from .prompts import load_rerank_prompt
from .trace import emit_trace, emit_trace_async

DEFAULT_RETRIEVE_LIMIT = 8
DEFAULT_MAX_SELECTED = 4


def _experience_record(text: str) -> dict[str, Any] | None:
    """Parse one stored experience document body."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def transferable_core(experience: Mapping[str, Any]) -> str:
    """Return the paper-agnostic core, falling back to the prose statement.

    `transferable_core` is written by the current extraction prompts. Records
    extracted before that field existed fall back to the head of `statement` --
    a truncated paragraph opening, which is a weak descriptor but keeps every
    pre-existing record usable without re-extraction.
    """
    core = experience.get("transferable_core")
    if isinstance(core, str) and core.strip():
        return " ".join(core.split())
    statement = experience.get("statement")
    if isinstance(statement, str) and statement.strip():
        return " ".join(statement.split())[:200]
    return ""


def _descriptor(record: Mapping[str, Any]) -> str:
    """Compact one-line candidate for reranking (~200-300 chars).

    The full 2.5K-char statement is withheld until synthesis, so eight
    candidates fit in one small rerank call.
    """
    experience = record.get("experience", {})
    if not isinstance(experience, Mapping):
        experience = {}
    confidence = experience.get("confidence")
    confidence_text = (
        f"{float(confidence):.2f}" if isinstance(confidence, int | float) else "n/a"
    )
    paper = record.get("paper", {})
    title = paper.get("title", "") if isinstance(paper, Mapping) else ""
    return " | ".join(
        part
        for part in (
            f"{record.get('id', '')} [{str(record.get('experience_level', '')).upper()}"
            f" conf={confidence_text}]",
            str(experience.get("domain") or ""),
            str(experience.get("task") or ""),
            transferable_core(experience),
            f"scope: {' '.join(str(experience.get('scope') or '').split())[:160]}",
            f"source: {' '.join(str(title).split())[:80]}",
        )
        if part and not part.endswith((": ", "scope: ", "source: "))
    )


def gather_candidates(
    *,
    memory_dir: str | Path,
    project_id: str,
    query: str,
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
    """
    hits = search_experience_records(
        memory_dir=memory_dir,
        project_id=project_id,
        topic=query,
        method=method,
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
        candidate_ids=[item["id"] for item in candidates],
        descriptors={item["id"]: item["descriptor"] for item in candidates},
    )
    return candidates


async def rerank_candidates(
    *,
    candidates: list[dict[str, Any]],
    task: str,
    max_selected: int,
    model: Any,
    prompt: str,
    memory_dir: str | Path | None = None,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Select the small working set for synthesis.

    Returns ``{"selected": [{"id", "record", "descriptor"}, ...], "reason": ""}``
    on success. The fallback when the model's JSON is unparseable is the first
    `max_selected` candidates from the scored list -- a degraded but safe path
    that keeps the tool from outright failing on one parse error.
    """
    descriptors_text = "\n".join(
        f"{i + 1}. {item['descriptor']}" for i, item in enumerate(candidates)
    )
    filled = prompt.format(
        task=task.strip(),
        descriptors=descriptors_text,
        max_selected=max_selected,
    )
    response = await model.ainvoke(
        [SystemMessage(content=filled), HumanMessage(content="Rerank now.")]
    )
    output = format_message_content(response).strip()

    async def _traced(result: dict[str, Any]) -> dict[str, Any]:
        if memory_dir is not None:
            await emit_trace_async(
                memory_dir,
                "rerank",
                call_id=call_id,
                task=task.strip(),
                raw_output=output,
                selected_ids=[item["id"] for item in result["selected"]],
                reason=result["reason"],
            )
        return result

    fenced = output.removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        payload = json.loads(fenced)
    except json.JSONDecodeError:
        return await _traced(
            {
                "selected": candidates[:max_selected],
                "reason": "rerank parse failed; using top scored candidates",
            }
        )
    if not isinstance(payload, dict):
        return await _traced(
            {
                "selected": candidates[:max_selected],
                "reason": "rerank returned non-object; using top scored candidates",
            }
        )
    selected_ids = payload.get("selected", [])
    if not isinstance(selected_ids, list):
        selected_ids = []
    lookup = {item["id"]: item for item in candidates}
    selected = [lookup[eid] for eid in selected_ids if eid in lookup][:max_selected]
    return await _traced(
        {
            "selected": selected or candidates[:max_selected],
            "reason": str(payload.get("reason") or ""),
        }
    )


__all__ = [
    "DEFAULT_MAX_SELECTED",
    "DEFAULT_RETRIEVE_LIMIT",
    "gather_candidates",
    "load_rerank_prompt",
    "rerank_candidates",
    "transferable_core",
]
