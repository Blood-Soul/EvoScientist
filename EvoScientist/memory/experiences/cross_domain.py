"""Cross-field experience search: find a practice another field solved a structurally similar need with.

The ordinary `search_experience` answers "what has been found about this subject", ranked lexically over the whole
library and filtered by the caller's own discipline/domain. That ranking is a similarity ranking, and similarity is the
wrong objective here: the useful record for a design point is usually the one from a *different* field, which shares
neither vocabulary nor topic with the question. CPR-eval (Jung et al. 2026) measures how far similarity gets on exactly
this task: an untrained dense retriever reaches R@10 0.13, BM25 with generated hypothetical answers 0.41, and an LLM
that reads the candidates and judges transferability 0.76. The gap is the reason this module exists: retrieval proposes,
an LLM decides.

Three steps, each a separate failure mode:

1. **Abstract** -- the design point is rewritten as a field-neutral "How to ..." need, plus a few short descriptions of
   what a practice from another field would look like. The passages exist to put the *other* field's vocabulary into
   the lexical query; without them BM25 only sees the caller's own terms.
2. **Recall** -- BM25 over a per-record view (problem, method, action, mechanism, effect), restricted to records whose
   `discipline`/`domain` differ from the caller's when the caller says what that is. Retrieval is deliberately cheap and
   deliberately lexical: the library is small, and the LLM gate below is what decides.
3. **Gate** -- one LLM call per batch of candidates, judging object mapping and transfer path (`prompt/cross_domain_gate.md`).
   Fail-closed: an unparsable reply keeps none of that batch, because a wrong record presented as transferable is worse
   than a missing one.

The gate's output is a *hypothesis*: a mapping written from text, with no evidence that anybody ever applied it. The
tool that exposes this module says so, and the caller is expected to check the mapping against its own problem before
adopting anything.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .extraction import _prompt_dir_candidates
from .store import list_experience_documents

PROMPTS = {"abstract": "cross_domain_abstract.md", "gate": "cross_domain_gate.md"}
CALL_TIMEOUT_S = 240
GATE_BATCH = 10
DEFAULT_CANDIDATES = 24
DEFAULT_LIMIT = 4
PASSAGES = 4
FIELD_CHARS = 320
MAX_CONCURRENT_CALLS = 4


def load_cross_domain_prompts() -> dict[str, str]:
    """The two prompts, from the same directories the extraction prompts come from."""
    searched: list[str] = []
    for directory in _prompt_dir_candidates():
        found = {key: directory / name for key, name in PROMPTS.items()}
        searched.append(str(directory))
        if all(path.is_file() for path in found.values()):
            return {key: path.read_text(encoding="utf-8") for key, path in found.items()}
    raise FileNotFoundError(
        f"Cross-domain prompts not found ({', '.join(PROMPTS.values())}); searched: {', '.join(searched)}"
    )


def _text(value: Any, limit: int = FIELD_CHARS) -> str:
    if isinstance(value, Mapping):
        value = value.get("text") or value.get("benefit") or ""
    if isinstance(value, (list, tuple)):
        value = "; ".join(_text(item, limit) for item in value if item)
    return " ".join(str(value or "").split())[:limit]


def record_view(record: Mapping[str, Any]) -> str:
    """What the record says, in the record's own words, as ranking text.

    Both shapes are read: the current one (`problem`, `method`, `mechanism` as an object) and the older one
    (`statement`, `applicable_when`, `mechanism` as a string). Field names are left out on purpose -- tokenizing the
    stored JSON would put "mechanism" and "source_setting" into every document and flatten the ranking.
    """
    parts = [
        _text(record.get("domain")),
        _text(record.get("task")),
        _text(record.get("problem") or record.get("trigger_context") or record.get("statement")),
        _text(record.get("method")),
        _text(record.get("action")),
        _text(record.get("mechanism")),
        _text(record.get("effect")),
        _text(record.get("applicable_when")),
    ]
    return " ".join(part for part in parts if part)


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def bm25(documents: Sequence[str], query: str, *, k1: float = 1.5, b: float = 0.75) -> list[float]:
    """BM25 (Lucene idf) in pure Python: the library is small, and a dependency-free scorer keeps retrieval explainable."""
    docs = [Counter(_tokens(document)) for document in documents]
    lengths = [sum(counts.values()) for counts in docs]
    average = (sum(lengths) / len(lengths)) if lengths else 1.0
    frequency = Counter(token for counts in docs for token in counts)
    total = len(docs)
    scores = [0.0] * total
    for token, weight in Counter(_tokens(query)).items():
        seen = frequency.get(token, 0)
        if not seen:
            continue
        idf = math.log(1 + (total - seen + 0.5) / (seen + 0.5))
        for index, counts in enumerate(docs):
            tf = counts.get(token, 0)
            if not tf:
                continue
            norm = k1 * (1 - b + b * (lengths[index] / (average or 1.0)))
            scores[index] += weight * idf * tf * (k1 + 1) / (tf + norm)
    return scores


def _json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    while start != -1:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except ValueError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


async def _ask(model: Any, system: str, human: str, semaphore: asyncio.Semaphore) -> dict[str, Any] | None:
    async with semaphore:
        try:
            reply = await asyncio.wait_for(
                model.ainvoke([SystemMessage(content=system), HumanMessage(content=human)]),
                timeout=CALL_TIMEOUT_S,
            )
        except Exception:
            return None
    return _json_object(str(getattr(reply, "content", "") or ""))


def _candidate_line(index: int, document: Any) -> str:
    record = _record_payload(document)
    labels = "/".join(
        label for label in (_text(record.get("discipline"), 40), _text(record.get("domain"), 40)) if label
    ) or "unknown"
    fields = [
        f"[{index}] field: {labels} | task: {_text(record.get('task'), 100)}",
        f"problem: {_text(record.get('problem') or record.get('trigger_context') or record.get('statement'))}",
        f"done: {_text(record.get('action'))}",
        f"mechanism: {_text(record.get('mechanism'))}",
        f"effect: {_text(record.get('effect'))}",
    ]
    return "\n".join(field for field in fields if not field.endswith(": "))


def _record_payload(document: Any) -> dict[str, Any]:
    try:
        record = json.loads(document.text) if isinstance(getattr(document, "text", None), str) else {}
    except ValueError:
        record = {}
    if isinstance(record.get("experience"), Mapping):
        record = record["experience"]
    return record if isinstance(record, dict) else {}


async def _abstract(model: Any, prompts: Mapping[str, str], design_point: str, semaphore: asyncio.Semaphore) -> dict[str, Any]:
    parsed = await _ask(model, prompts["abstract"], design_point, semaphore)
    goal = _text((parsed or {}).get("goal"), 400)
    passages = [text for text in (parsed or {}).get("passages") or [] if isinstance(text, str) and text.strip()]
    if not goal:
        # The design point itself is still a usable query; the passages are the part that cannot be improvised.
        return {"goal": " ".join(design_point.split())[:400], "passages": [], "abstracted": False}
    return {"goal": goal, "passages": passages[:PASSAGES], "abstracted": True}


async def search_cross_domain(
    *,
    memory_dir: str | Path,
    project_id: str,
    design_point: str,
    model: Any,
    exclude_domain: str = "",
    exclude_discipline: str = "",
    context: str = "",
    limit: int = DEFAULT_LIMIT,
    candidates: int = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    """Rank the library for a cross-field transfer and keep what the gate admits.

    `exclude_domain` / `exclude_discipline` name the caller's own field; records carrying either label are left out,
    because the point of this search is the records from fields the caller would not read. Unknown labels are kept --
    excluding `unknown` would silently narrow the library to whichever records happen to carry a label.

    `context` is what the work has to work with right now (data, compute, metric, constraints); the gate reads it, so
    a transfer whose prerequisites the work does not have is rejected rather than handed over.
    """
    prompts = load_cross_domain_prompts()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
    abstracted = await _abstract(model, prompts, design_point, semaphore)

    documents = list_experience_documents(memory_dir=memory_dir, project_id=project_id)
    wanted_domain = exclude_domain.strip().casefold()
    wanted_discipline = exclude_discipline.strip().casefold()
    eligible = [
        document
        for document in documents
        if not (wanted_domain and (document.domain or "").casefold() == wanted_domain)
        and not (wanted_discipline and (document.discipline or "").casefold() == wanted_discipline)
    ]
    views = [record_view(_record_payload(document)) for document in eligible]
    query = " ".join([abstracted["goal"]] + abstracted["passages"])
    scores = bm25(views, query) if views else []
    scored = sorted(
        ((score, document) for score, document, view in zip(scores, eligible, views, strict=False) if view),
        key=lambda item: -item[0],
    )
    ranked = [document for _, document in scored[:candidates]]
    if not ranked:
        return {"goal": abstracted["goal"], "passages": abstracted["passages"], "abstracted": abstracted["abstracted"],
                "considered": 0, "judged": 0, "kept": [], "dropped": {}, "gate_failed": 0}

    batches = [ranked[start:start + GATE_BATCH] for start in range(0, len(ranked), GATE_BATCH)]
    numbering: dict[str, int] = {}
    for index, document in enumerate(ranked, 1):
        numbering[document.observation_id] = index
    calls = []
    for batch in batches:
        listing = "\n\n".join(_candidate_line(numbering[document.observation_id], document) for document in batch)
        calls.append(_ask_gate(model, prompts["gate"], design_point, abstracted["goal"], context, listing, semaphore, batch))
    result = await _collect(calls, ranked, numbering, abstracted)
    result["kept"] = result["kept"][:limit]   # the gate may admit more than the caller asked to read
    return result


async def _ask_gate(model: Any, template: str, design_point: str, goal: str, context: str, listing: str,
                    semaphore: asyncio.Semaphore, batch: Sequence[Any]) -> dict[str, Any] | None:
    system = template.split("Design point:", 1)[0].replace("{{", "{").replace("}}", "}")
    situation = " ".join(context.split()) or "not stated"
    human = (f"Design point:\n{design_point}\n\nRestated need:\n{goal}\n\nCurrent situation:\n{situation}"
             f"\n\nCandidates:\n{listing}")
    return {"ids": [document.observation_id for document in batch],
            "parsed": await _ask(model, system, human, semaphore)}


async def _collect(calls: Sequence[Any], ranked: Sequence[Any], numbering: Mapping[str, int],
                   abstracted: Mapping[str, Any]) -> dict[str, Any]:
    replies = await asyncio.gather(*calls)
    by_id = {document.observation_id: document for document in ranked}
    kept: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    gate_failed = 0
    judged = 0
    for reply in replies:
        parsed = (reply or {}).get("parsed") or {}
        ids = (reply or {}).get("ids") or []
        judged += len(ids)
        judgements = parsed.get("judgements") if isinstance(parsed, Mapping) else None
        if not isinstance(judgements, list):
            gate_failed += 1   # fail closed: an unparsable batch keeps none of its candidates
            dropped["unparsed"] += len(ids)
            continue
        by_index = {
            judgement["i"]: judgement
            for judgement in judgements
            if isinstance(judgement, Mapping) and isinstance(judgement.get("i"), int)
        }
        for record_id in ids:
            judgement = by_index.get(numbering.get(record_id, -1))
            if judgement is None:
                dropped["unjudged"] += 1
                continue
            if not judgement.get("transferable"):
                dropped[_text(judgement.get("drop_reason") or "rejected", 40)] += 1
                continue
            mappings, relations, path = _mapping(judgement)
            # The prompt asks for all three; this is the same rule in code, because a judgement that says
            # "transferable" and then cannot say why is exactly the case the caller must not see.
            if len(mappings) < 2:
                dropped["no_mapping"] += 1
                continue
            if not relations:
                dropped["no_relation"] += 1
                continue
            if not path:
                dropped["no_path"] += 1
                continue
            document = by_id[record_id]
            record = _record_payload(document)
            kept.append({
                "id": record_id,
                "discipline": document.discipline,
                "domain": document.domain,
                "task": document.task,
                "summary": _text(record.get("problem") or record.get("trigger_context") or record.get("statement"), 240),
                "mappings": mappings,
                "relations": relations,
                "path": path,
            })
    return {"goal": abstracted["goal"], "passages": abstracted["passages"], "abstracted": abstracted["abstracted"],
            "considered": len(ranked), "judged": judged, "kept": kept, "dropped": dict(dropped),
            "gate_failed": gate_failed}


def _mapping(judgement: Mapping[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    """The mapping a judgement wrote, in the shape the renderer prints: mappings, relations, path.

    Older or sloppier replies carry a single `mapping` string instead of the two lists; it is read as one mapping so
    the caller still sees what was written, but it cannot reach the two-mapping minimum on its own.
    """
    mappings = [
        {"source": _text(item.get("source"), 160), "target": _text(item.get("target"), 160),
         "rationale": _text(item.get("rationale"), 240)}
        for item in judgement.get("object_mappings") or []
        if isinstance(item, Mapping) and (item.get("source") or item.get("target"))
    ]
    if not mappings and _text(judgement.get("mapping"), 400):
        mappings = [{"source": "", "target": "", "rationale": _text(judgement.get("mapping"), 400)}]
    relations = [
        {"relation": _text(item.get("relation"), 60), "problem": _text(item.get("problem"), 200),
         "solution": _text(item.get("solution"), 200)}
        for item in judgement.get("shared_relations") or []
        if isinstance(item, Mapping) and (item.get("problem") or item.get("solution"))
    ]
    return mappings, relations, _text(judgement.get("path"), 400)


__all__ = ["bm25", "load_cross_domain_prompts", "record_view", "search_cross_domain"]
