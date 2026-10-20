"""Reviewable, opt-in relations over the existing paper-experience store.

The sidecar contains candidate judgments, not verified scientific claims. It
never modifies experience records. Model decisions are validated against the
supplied records; a matching quotation is not verification against a paper.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...utils import format_message_content
from .candidates import nominate_candidates
from .field_candidates import nominate_field_candidates
from .store import EXPERIENCE_DIR, list_experience_documents

GRAPH_VERSION = 2
GRAPH_FILENAME = "experience_graph.json"
_FIELDS = (
    "discipline",
    "domain",
    "task",
    "trigger_context",
    "statement",
    "action",
    "effect",
    "scope",
    "applicable_when",
    "not_applicable_when",
    "transferable_core",
    "practice_trace",
    "evidence",
    "evidence_scope",
    "rationale",
    "rationale_depth",
    "claim_type",
    "bindings",
)
_RELATIONS = {
    "supports",
    "qualifies",
    "contradicts",
    "alternative_to",
    "complements",
    "overlaps_with",
    "specializes",
    "depends_on",
    "fixed_by",
}
_SYMMETRIC = {"contradicts", "alternative_to", "complements", "overlaps_with"}


def experience_graph_path(memory_dir: str | Path, project_id: str) -> Path:
    """Return the standard optional sidecar location for a project."""
    return Path(memory_dir).expanduser() / EXPERIENCE_DIR / project_id / GRAPH_FILENAME


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_documents(memory_dir: str | Path, project_id: str, papers: set[str] | None):
    records = {}
    for document in list_experience_documents(
        memory_dir=memory_dir, project_id=project_id
    ):
        payload = json.loads(document.text)
        paper, item, record_id = payload["paper"], payload["experience"], payload["id"]
        if papers and paper.get("paper_id") not in papers:
            continue
        if record_id in records:
            raise ValueError(f"duplicate public experience ID: {record_id}")
        records[record_id] = {
            "id": record_id,
            "level": payload["experience_level"],
            "paper_id": paper.get("paper_id"),
            "paper_key": paper.get("paper_key"),
            "paper": paper,
            "source_path": document.path,
            "original_id": item.get("id"),
            "content": {key: item[key] for key in _FIELDS if key in item},
            "source_l1_ids": item.get("source_l1_ids", []),
        }
    return records


def _hierarchy(
    records: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    lookup: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records.values():
        if record["level"] == "l1" and record["original_id"]:
            lookup[(record["paper_key"], record["original_id"])].append(record["id"])
    edges, warnings = [], []
    for record in records.values():
        for original_id in record["source_l1_ids"]:
            matches = lookup.get((record["paper_key"], original_id), [])
            if len(matches) != 1:
                warnings.append(f"{record['id']}: unresolved L1 source {original_id}")
                continue
            target = records[matches[0]]
            edges.append(
                {
                    "from": record["id"],
                    "to": target["id"],
                    "relation": "derived_from",
                    "status": "source_record",
                    "claim": "Induced experience lists this L1 source.",
                    "reason": "The source_l1_ids field references this record.",
                    "conditions": [],
                    "limitations": [
                        "The source pointer alone does not verify the claim."
                    ],
                    "unresolved": [],
                    "evidence": [
                        {
                            "id": record["id"],
                            "field": "source_l1_ids",
                            "quote": original_id,
                        }
                    ],
                    "provenance": {
                        "method": "source_pointer",
                        "prompt_hash": None,
                        "classifier_id": None,
                    },
                    "record_versions": {
                        record["id"]: _digest(record),
                        target["id"]: _digest(target),
                    },
                }
            )
    return edges, warnings


def _pair_group(a: dict[str, Any], b: dict[str, Any]) -> str:
    levels = "-".join(sorted((a["level"], b["level"])))
    scope = "same-paper" if a["paper_key"] == b["paper_key"] else "cross-paper"
    return f"{scope}:{levels}"


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for entry in value:
            yield from _strings(entry)
    elif isinstance(value, dict):
        for entry in value.values():
            yield from _strings(entry)


def _validate_decision(
    raw: str, a: dict[str, Any], b: dict[str, Any]
) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("linker output must be an object")
    decision = result.get("decision")
    if decision in {"none", "uncertain"}:
        if (
            set(result) != {"decision", "reason"}
            or not isinstance(result["reason"], str)
            or not result["reason"].strip()
        ):
            raise ValueError("none/uncertain requires a reason")
        return result
    if decision != "related" or set(result) != {"decision", "relations"}:
        raise ValueError("invalid decision or fields")
    relations = result["relations"]
    if not isinstance(relations, list) or not relations:
        raise ValueError("related requires a nonempty relations list")
    records = {a["id"]: a, b["id"]: b}
    seen_relations = set()
    for relation in relations:
        required = {
            "relation",
            "from",
            "to",
            "claim",
            "reason",
            "conditions",
            "limitations",
            "unresolved",
            "evidence",
        }
        if not isinstance(relation, dict) or set(relation) != required:
            raise ValueError("invalid relation fields")
        kind, source, target = relation["relation"], relation["from"], relation["to"]
        if (
            kind not in _RELATIONS
            or source == target
            or {source, target} != set(records)
        ):
            raise ValueError("invalid relation type or endpoints")
        if (
            kind == "fixed_by"
            and records[source]["paper_key"] != records[target]["paper_key"]
        ):
            raise ValueError("fixed_by requires the same source paper")
        key = (
            (kind, *sorted((source, target)))
            if kind in _SYMMETRIC
            else (kind, source, target)
        )
        if key in seen_relations:
            raise ValueError("duplicate relation")
        seen_relations.add(key)
        for field in ("claim", "reason"):
            if not isinstance(relation[field], str) or not relation[field].strip():
                raise ValueError(f"{field} must be nonempty")
        for field in ("conditions", "limitations", "unresolved"):
            if not isinstance(relation[field], list) or any(
                not isinstance(x, str) or not x.strip() for x in relation[field]
            ):
                raise ValueError(f"{field} must be a list of nonempty strings")
        evidence = relation["evidence"]
        if not isinstance(evidence, list) or len(evidence) < 2:
            raise ValueError("evidence must cite both records")
        seen_ids = set()
        for entry in evidence:
            if not isinstance(entry, dict) or set(entry) != {"id", "field", "quote"}:
                raise ValueError("invalid evidence object")
            record = records.get(entry["id"])
            field, quote = entry["field"], entry["quote"]
            if (
                record is None
                or not isinstance(field, str)
                or field not in record["content"]
                or not isinstance(quote, str)
                or not quote.strip()
            ):
                raise ValueError("invalid evidence source, field, or quote")
            if not any(quote in value for value in _strings(record["content"][field])):
                raise ValueError(
                    f"evidence quote not found in {entry['id']} field {field}"
                )
            seen_ids.add(entry["id"])
        if seen_ids != set(records):
            raise ValueError("evidence must cite both records")
    return result


async def build_experience_graph(
    *,
    memory_dir: str | Path,
    project_id: str,
    prompt: str,
    model: Any | None = None,
    papers: set[str] | None = None,
    neighbors: int = 3,
    max_pairs: int | None = None,
    recall_pool: int = 20,
    min_similarity: float = 0.12,
    min_shared_terms: int = 2,
    max_document_frequency: float = 0.6,
    diversity_ratio: float = 0.8,
    candidate_strategy: str = "lexical",
    candidate_routes: set[str] | None = None,
    previous: dict[str, Any] | None = None,
    classifier_id: str = "",
) -> dict[str, Any]:
    """Build a sidecar, preserving judgments only when all their inputs match."""
    if neighbors < 1 or (max_pairs is not None and max_pairs < 1):
        raise ValueError("neighbors and max_pairs must be positive")
    if previous and previous.get("project_id") != project_id:
        raise ValueError("previous graph belongs to another project")
    records = _record_documents(memory_dir, project_id, papers)
    hierarchy, warnings = _hierarchy(records)
    if candidate_strategy not in {"lexical", "field"}:
        raise ValueError("unknown candidate strategy")
    if candidate_strategy == "lexical" and candidate_routes is not None:
        raise ValueError("candidate_routes require the field strategy")
    nominator = (
        nominate_field_candidates
        if candidate_strategy == "field"
        else nominate_candidates
    )
    extra = (
        {"enabled_routes": candidate_routes} if candidate_strategy == "field" else {}
    )
    nominated, candidate_state, candidate_stats = nominator(
        records,
        neighbors=neighbors,
        recall_pool=recall_pool,
        min_similarity=min_similarity,
        min_shared_terms=min_shared_terms,
        max_document_frequency=max_document_frequency,
        diversity_ratio=diversity_ratio,
        previous=(previous or {}).get("candidate_state"),
        **extra,
    )
    ranked_pairs = sorted(
        nominated, key=lambda pair: (-nominated[pair]["similarity"], pair)
    )
    selected_pairs = ranked_pairs if max_pairs is None else ranked_pairs[:max_pairs]
    pairs = {
        pair: _pair_group(records[pair[0]], records[pair[1]]) for pair in selected_pairs
    }
    candidate_stats["selected_pairs"] = len(pairs)
    selected_ids = {key for pair in pairs for key in pair}
    candidate_stats["selected_covered_nodes"] = len(selected_ids)
    candidate_stats["selected_isolated_ids"] = sorted(set(records) - selected_ids)
    prompt_hash = _digest(prompt)
    if not classifier_id:
        if model is None:
            classifier_id = (previous or {}).get("classifier_id", "")
        else:
            classifier_id = _digest(
                {
                    "class": f"{type(model).__module__}.{type(model).__qualname__}",
                    "model": str(
                        getattr(model, "model_name", "") or getattr(model, "model", "")
                    ),
                    "temperature": str(getattr(model, "temperature", "")),
                }
            )
    compatible = bool(
        previous
        and previous.get("version") == GRAPH_VERSION
        and previous.get("project_id") == project_id
    )
    old_rows = previous.get("decisions", []) if compatible else []
    matching_run = (
        compatible
        and previous.get("prompt_hash") == prompt_hash
        and previous.get("classifier_id", "") == classifier_id
    )
    cached = {}
    if matching_run:
        for entry in previous.get("history", []):
            if (
                entry.get("graph_version", GRAPH_VERSION) == GRAPH_VERSION
                and entry.get("prompt_hash") == prompt_hash
                and entry.get("classifier_id", "") == classifier_id
            ):
                row = entry["decision"]
                cached[tuple(row["ids"])] = row
        cached.update({tuple(row["ids"]): row for row in old_rows})
    decisions, edges = [], list(hierarchy)
    reused = set()
    classifier_calls = 0
    for (a_id, b_id), group in pairs.items():
        a, b = records[a_id], records[b_id]
        fingerprint = _digest([a, b])
        old = cached.get((a_id, b_id))
        if old and old.get("fingerprint") == fingerprint and "result" in old:
            row = old
            reused.add((a_id, b_id))
        else:
            row = {"ids": [a_id, b_id], "group": group, "fingerprint": fingerprint}
            if model is not None:
                classifier_calls += 1
                try:
                    response = await model.ainvoke(
                        [
                            SystemMessage(content=prompt),
                            HumanMessage(
                                content=json.dumps([a, b], ensure_ascii=False)
                            ),
                        ]
                    )
                    row["result"] = _validate_decision(
                        format_message_content(response), a, b
                    )
                    row["judged_at"] = datetime.now(UTC).isoformat()
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
        decisions.append(row)
        result = row.get("result", {})
        if result.get("decision") == "related":
            for relation in result["relations"]:
                edges.append(
                    {
                        **relation,
                        "status": "unreviewed",
                        "judged_at": row["judged_at"],
                        "record_versions": {a_id: _digest(a), b_id: _digest(b)},
                        "provenance": {
                            "method": "classifier_inference",
                            "prompt_hash": prompt_hash,
                            "classifier_id": classifier_id,
                            "source_check": "experience_record_only",
                        },
                    }
                )
    history = list(previous.get("history", [])) if previous else []
    if previous and not compatible:
        warnings.append(
            "Previous graph version is incompatible; its judgments are retained "
            "in history and must be reclassified."
        )
        history.extend(
            {
                "graph_version": previous.get("version"),
                "prompt_hash": previous.get("prompt_hash"),
                "classifier_id": previous.get("classifier_id"),
                "decision": row,
            }
            for row in previous.get("decisions", [])
            if "result" in row or "error" in row
        )
    for row in old_rows:
        if tuple(row["ids"]) not in reused and ("result" in row or "error" in row):
            history.append(
                {
                    "prompt_hash": previous.get("prompt_hash"),
                    "classifier_id": previous.get("classifier_id"),
                    "decision": row,
                }
            )
    candidate_stats["judgments_reused"] = len(reused)
    candidate_stats["classifier_calls"] = classifier_calls
    candidate_stats["pending_pairs"] = sum(
        "result" not in row and "error" not in row for row in decisions
    )
    return {
        "version": GRAPH_VERSION,
        "project_id": project_id,
        "paper_filter": sorted(papers) if papers else None,
        "prompt_hash": prompt_hash,
        "classifier_id": classifier_id,
        "records": records,
        "candidate_state": candidate_state,
        "candidate_strategy": candidate_strategy,
        "candidate_stats": candidate_stats,
        "candidate_evidence": [
            {"ids": list(pair), **nominated[pair]} for pair in pairs
        ],
        "edges": edges,
        "decisions": decisions,
        "history": history,
        "warnings": warnings,
    }


def expand_graph(
    graph: dict[str, Any],
    ids: list[str],
    *,
    limit: int = 8,
    relations: set[str] | None = None,
    allowed_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return bounded one-hop neighbors and their edges, not applicability verdicts."""
    if limit <= 0:
        return []
    selected, expanded = set(ids), {}
    for edge in graph["edges"]:
        if relations is not None and edge["relation"] not in relations:
            continue
        if edge["from"] in selected:
            other = edge["to"]
        elif edge["to"] in selected:
            other = edge["from"]
        else:
            continue
        if (
            other in selected
            or other not in graph["records"]
            or (allowed_ids is not None and other not in allowed_ids)
        ):
            continue
        if other not in expanded:
            if len(expanded) >= limit:
                continue
            expanded[other] = {"edges": [], "record": graph["records"][other]}
        expanded[other]["edges"].append(edge)
    return list(expanded.values())


def graph_is_current(graph: dict[str, Any], *, memory_dir: str | Path) -> bool:
    """Reject stale, old-format, or incomplete snapshots before graph search."""
    if graph.get("version") != GRAPH_VERSION:
        return False
    paper_filter = graph.get("paper_filter")
    current = _record_documents(
        memory_dir, graph["project_id"], set(paper_filter) if paper_filter else None
    )
    return current == graph["records"]


async def rebase_experience_graph(*, source_graph, memory_dir, project_id, prompt):
    """Revalidate existing decisions after an identical library is mounted elsewhere.

    Only storage locations/project metadata may differ. Project-scoped public
    IDs are rebound through an exact content mapping; no model is contacted.
    """
    from .routing import build_routing_index

    if source_graph.get("version") != GRAPH_VERSION or source_graph.get(
        "prompt_hash"
    ) != _digest(prompt):
        raise ValueError(
            "Graph version/prompt changed; classify again instead of rebasing"
        )
    current = _record_documents(memory_dir, project_id, None)
    fields = (
        "level",
        "paper_id",
        "paper_key",
        "original_id",
        "content",
        "source_l1_ids",
    )
    lookup = {_digest({f: r.get(f) for f in fields}): key for key, r in current.items()}
    if len(lookup) != len(current) or len(current) != len(source_graph["records"]):
        raise ValueError("Mounted records are missing or ambiguous")
    id_map = {}
    for old_id, old_record in source_graph["records"].items():
        key = lookup.get(_digest({f: old_record.get(f) for f in fields}))
        if key is None:
            raise ValueError(f"Mounted record content changed: {old_id}")
        id_map[old_id] = key
        record = current[key]
        original_paper = dict(old_record["paper"])
        mounted_paper = dict(record["paper"])
        original_paper.pop("project_id", None)
        mounted_paper.pop("project_id", None)
        if original_paper != mounted_paper:
            raise ValueError(f"Mounted paper metadata changed: {key}")

    def remap(value):
        if isinstance(value, dict):
            return {id_map.get(k, k): remap(v) for k, v in value.items()}
        if isinstance(value, list):
            return [remap(v) for v in value]
        return id_map.get(value, value) if isinstance(value, str) else value

    graph = remap(source_graph)
    graph.update(project_id=project_id, records=current)
    for row in graph["decisions"]:
        row["ids"] = sorted(row["ids"])
        a, b = [current[key] for key in row["ids"]]
        row["result"] = _validate_decision(json.dumps(row["result"]), a, b)
        row["fingerprint"] = _digest([a, b])
    for edge in graph["edges"]:
        edge["record_versions"] = {
            key: _digest(current[key]) for key in (edge["from"], edge["to"])
        }
        if edge["relation"] != "derived_from":
            edge["provenance"]["rebased_from_project"] = source_graph["project_id"]
    graph["candidate_state"]["record_versions"] = {
        key: _digest(record) for key, record in current.items()
    }
    graph["candidate_state"]["routing_index"] = build_routing_index(current)
    graph["mount_provenance"] = {
        "source_graph_sha256": _digest(source_graph),
        "public_id_map": id_map,
        "model_requests": 0,
    }
    return graph
