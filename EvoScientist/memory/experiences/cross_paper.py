"""Cross-paper L1 -> L2 induction: group similar L1 records from different papers and induce findings across them.

This module is the only L2 induction pass: it looks across all papers' L1 records:
records that answer the same kind of problem, or compare the same kind of design choice, are grouped, and an LLM
writes findings that hold across the group -- the step from specific to abstract. Design notes:
`docs/cross_paper_l2.zh-CN.md`.

Flow (triggered after paper extraction, every `CROSS_PAPER_EVERY` new L1 records):

* **rebuild** (first trigger, then every `CROSS_PAPER_REBUILD_EVERY`-th): lexical TF-IDF over a per-record view
  (problem, method aliases, task, requires, action) nominates cross-paper neighbour pairs; Kruskal with a size cap
  turns them into candidate groups; the LLM splits each candidate group into real groups (or leaves records out);
  each group with >= 2 papers gets findings from the induction prompt. New findings are aligned with the previous
  ones by support overlap; unmatched old findings are retired, not deleted.
* **incremental** (the other triggers): each new record is offered to its nearest existing groups and the LLM
  assigns it or not. A group that gained members revises each of its findings with one operation (KEEP / REVISE /
  ADD_CONFLICT / RETIRE) instead of a rewrite. Unassigned records go to a pool, which is grouped like a rebuild.

Grounding is enforced in code: every `from` / `support` id must be a group member, a finding must rest on >= 2
papers, a stated limit must summarize >= 2 members' own `limits_stated`, and numbers in the claim and options must
appear in the cited records. The findings live in `experiences/projects/<P>/_cross_paper/l2.json`, which the
existing listing reads like one more paper directory, so retrieval, rendering and plan review need no new path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...utils import format_message_content
from .._atomic import atomic_write_json, read_json
from .extraction import _prompt_dir_candidates, _ungrounded_numbers
from .store import (
    EXPERIENCE_DIR,
    _cache_lock,
    _experience_id,
    _level_parse_cache,
    list_experience_documents,
    refresh_experience_catalog,
)

logger = logging.getLogger(__name__)

CROSS_DIR = "_cross_paper"
GROUP_PROMPT = "l2_cross_paper_group.md"
INDUCE_PROMPT = "l2_cross_paper.md"
ASSIGN_PROMPT = "l2_cross_paper_assign.md"
REVISE_PROMPT = "l2_cross_paper_revise.md"


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, "").strip() or default))
    except ValueError:
        return default


# Tentative values; tune from experiments (see the design doc).
def every() -> int:
    return _env_int("EVOSCIENTIST_CROSS_PAPER_EVERY", 15)


def rebuild_every() -> int:
    return _env_int("EVOSCIENTIST_CROSS_PAPER_REBUILD_EVERY", 3)


def concurrency() -> int:
    return _env_int("EVOSCIENTIST_CROSS_PAPER_CONCURRENCY", 6)


NEIGHBOURS = 6
NOMINATE_MIN_COS = 0.08
GROUP_CAP = 8
ASSIGN_TOP = 3
MIN_PAPERS = 2
REVISE_MIN_KEEP = 0.6
CALL_TIMEOUT_S = 900
MAX_FINDINGS_PER_GROUP = 3

_STOP = frozenset(
    """the and for with from that this these those what when where which while into onto over under than then there
    their them they its are was were been being have has had does did not but any all each per via how why who can
    could should would may might must will shall also only such same other about after before between both within
    without whether more most less least use used using uses one two model models data dataset datasets task tasks
    method methods approach approaches result results performance training train trained trains improve improves
    improved improvement better worse higher lower score scores accuracy based new paper work""".split()
)


# ----------------------------------------------------------------------------- records


def _cross_root(memory_dir: str | Path, project_id: str) -> Path:
    return Path(memory_dir).expanduser() / EXPERIENCE_DIR / project_id / CROSS_DIR


def load_l1_records(memory_dir: str | Path, project_id: str) -> list[dict[str, Any]]:
    """Current-shape L1 records of the project, with their stored ids and paper ids."""
    out: list[dict[str, Any]] = []
    for doc in list_experience_documents(memory_dir=memory_dir, project_id=project_id):
        if doc.experience_level != "l1" or doc.paper_key == CROSS_DIR:
            continue
        try:
            body = json.loads(doc.text)
        except ValueError:
            continue
        item = body.get("experience")
        if not isinstance(item, Mapping) or "problem" not in item or "effect" not in item:
            continue
        paper = body.get("paper") or {}
        out.append({"id": doc.observation_id, "paper": str(paper.get("paper_id") or doc.paper_key), "item": dict(item)})
    return out


def _view(item: Mapping[str, Any]) -> str:
    methods = " ".join(str(m) for m in item.get("method") or [])
    requires = " ".join(str(r.get("text") or "") for r in item.get("requires") or [] if isinstance(r, Mapping))
    problem = str(item.get("problem") or "")
    return " ".join([problem, problem, methods, methods, str(item.get("task") or ""), requires, str(item.get("action") or "")])


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9\-]+", text.lower()) if len(w) > 2 and w not in _STOP]


def tfidf_vectors(records: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    toks = [_tokens(_view(r["item"])) for r in records]
    n = len(records)
    df = Counter(w for t in toks for w in set(t))
    vecs = []
    for t in toks:
        tf = Counter(t)
        v = {w: (1 + math.log(c)) * math.log((n + 1) / (df[w] + 1)) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / norm for w, x in v.items()})
    return vecs


def _cos(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def candidate_groups(records: Sequence[Mapping[str, Any]], *, cap: int = GROUP_CAP) -> list[list[int]]:
    """Kruskal over cross-paper top-k neighbour edges with a component size cap; groups need >= 2 papers."""
    vecs = tfidf_vectors(records)
    edges: dict[tuple[int, int], float] = {}
    for i in range(len(records)):
        scored = sorted(
            ((_cos(vecs[i], vecs[j]), j) for j in range(len(records)) if records[j]["paper"] != records[i]["paper"]),
            reverse=True,
        )[:NEIGHBOURS]
        for score, j in scored:
            if score >= NOMINATE_MIN_COS:
                key = (min(i, j), max(i, j))
                edges[key] = max(edges.get(key, 0.0), score)
    parent = list(range(len(records)))
    size = [1] * len(records)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), _score in sorted(edges.items(), key=lambda kv: -kv[1]):
        a, b = find(i), find(j)
        if a != b and size[a] + size[b] <= cap:
            parent[b] = a
            size[a] += size[b]
    comps: dict[int, list[int]] = {}
    for i in range(len(records)):
        comps.setdefault(find(i), []).append(i)
    return [c for c in comps.values() if len(c) >= 2 and len({records[i]["paper"] for i in c}) >= MIN_PAPERS]


# ----------------------------------------------------------------------------- LLM plumbing


def _load_prompt(name: str) -> str:
    for directory in _prompt_dir_candidates():
        path = directory / name
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"cross-paper prompt {name} not found")


def _json_object(text: str) -> dict[str, Any]:
    cand = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cand, flags=re.S | re.I)
    if fenced:
        cand = fenced.group(1)
    start = cand.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    payload, _ = json.JSONDecoder().raw_decode(cand[start:])
    if not isinstance(payload, dict):
        raise ValueError("reply is not a JSON object")
    return payload


async def _ask(model: Any, prompt: str, human: str, sem: asyncio.Semaphore) -> dict[str, Any]:
    note = ""
    last: Exception | None = None
    for _ in range(2):
        async with sem:
            # A stuck request must not hold the trigger (and its lock) forever.
            response = await asyncio.wait_for(
                model.ainvoke([SystemMessage(content=prompt), HumanMessage(content=human + note)]), CALL_TIMEOUT_S
            )
        try:
            return _json_object(format_message_content(response))
        except ValueError as exc:
            last = exc
            note = f"\n\n[Correction] Your previous reply could not be parsed ({exc}). Return only the JSON object."
    raise ValueError(f"unreadable reply: {last}")


def _brief(local: str, rec: Mapping[str, Any]) -> dict[str, Any]:
    item = rec["item"]
    return {"id": local, "paper": rec["paper"], "task": item.get("task"), "problem": item.get("problem"),
            "method": item.get("method"), "action": item.get("action")}


def _full(local: str, rec: Mapping[str, Any]) -> dict[str, Any]:
    item = rec["item"]
    keep = ("domain", "task", "problem", "method", "requires", "source_setting", "limits_stated", "action", "effect",
            "comparison")
    return {"id": local, "paper": rec["paper"], **{k: item.get(k) for k in keep}}


# ----------------------------------------------------------------------------- validation


def _ids(value: Any, allowed: Mapping[str, str]) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(allowed[str(v)] for v in value if str(v) in allowed))


def validate_finding(
    raw: Mapping[str, Any], local_to_id: Mapping[str, str], members: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    """Map local ids to stored ids and enforce the cross-paper rules; None when the finding does not survive."""
    for key in ("domain", "task", "problem", "action"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            return None
    method = raw.get("method")
    if not isinstance(method, list) or not [m for m in method if isinstance(m, str) and m.strip()]:
        return None
    support = _ids(raw.get("support"), local_to_id)
    if len({members[s]["paper"] for s in support}) < MIN_PAPERS:
        return None

    def entries(name: str, *, min_from: int = 1, need_limits: bool = False) -> list[dict[str, Any]]:
        out = []
        for entry in raw.get(name) or []:
            if not isinstance(entry, Mapping) or not str(entry.get("text") or entry.get("option") or "").strip():
                continue
            src = _ids(entry.get("from"), local_to_id)
            if need_limits:
                src = [s for s in src if members[s]["item"].get("limits_stated")]
            if len(src) < min_from:
                continue
            out.append({**{k: v for k, v in entry.items() if k != "from"}, "from": src})
        return out

    record = {
        "domain": " ".join(raw["domain"].split()),
        "task": " ".join(raw["task"].split()),
        "problem": " ".join(raw["problem"].split()),
        "method": [" ".join(m.split()) for m in method if isinstance(m, str) and m.strip()],
        "action": " ".join(raw["action"].split()),
        "compared_options": [o for o in entries("compared_options") if str(o.get("result") or "").strip()],
        "mechanism": " ".join(str(raw.get("mechanism") or "").split()),
        "requires": [
            {**r, "basis": r.get("basis") if r.get("basis") in {"stated", "inferred"} else "inferred"}
            for r in entries("requires")
        ],
        "limits_stated": entries("limits_stated", min_from=2, need_limits=True),
        "conflicts": entries("conflicts", min_from=2),
        "origin": "cross_paper",
        "locator": "",
        "source_l1_ids": support,
    }
    source_text = json.dumps([members[s]["item"] for s in support], ensure_ascii=False)
    bad, total = _ungrounded_numbers([record["compared_options"], record["action"]], source_text)
    if total >= 3 and bad / total > 0.5:
        return None
    record["source_papers"] = sorted({members[s]["paper"] for s in support})
    record["layer"] = "l2"
    return record


def _text_size(record: Mapping[str, Any]) -> int:
    return len(json.dumps({k: record.get(k) for k in ("action", "mechanism", "compared_options", "requires",
                                                    "limits_stated")}, ensure_ascii=False))


# ----------------------------------------------------------------------------- state


def _empty_state() -> dict[str, Any]:
    return {"version": 1, "seen_l1": [], "triggers": 0, "groups": {}, "findings": {}, "retired": [], "pool": [],
            "history": []}


def load_state(memory_dir: str | Path, project_id: str) -> dict[str, Any]:
    state = read_json(_cross_root(memory_dir, project_id) / "state.json")
    return state if isinstance(state, dict) and state.get("version") == 1 else _empty_state()


def _finding_id(project_id: str, record: Mapping[str, Any]) -> str:
    return _experience_id(project_id=project_id, paper_key=CROSS_DIR, level="l2", index=0, item=record)


def save(memory_dir: str | Path, project_id: str, state: Mapping[str, Any]) -> None:
    root = _cross_root(memory_dir, project_id)
    root.mkdir(parents=True, exist_ok=True)
    findings = [f["record"] for f in state["findings"].values()]
    atomic_write_json(root / "l2.json", {"paper_id": CROSS_DIR, "domain_arxiv": None, "experiences": findings})
    with _cache_lock:
        _level_parse_cache.pop(str(root / "l2.json"), None)
    atomic_write_json(root / "state.json", state)
    papers = sorted({p for f in state["findings"].values() for p in f["record"].get("source_papers") or []})
    atomic_write_json(root / "metadata.json", {
        "project_id": project_id, "paper_id": CROSS_DIR, "canonical_paper_id": CROSS_DIR,
        "title": f"Cross-paper findings ({len(findings)} from {len(papers)} papers)", "url": "",
        "domain_arxiv": None, "extraction_schema": "v2", "origin": "cross_paper",
        "supporting_papers": papers, "extracted_at": datetime.now(UTC).isoformat(),
    })
    refresh_experience_catalog(memory_dir=memory_dir, project_id=project_id)


# ----------------------------------------------------------------------------- steps


async def _group(model, records: Sequence[dict[str, Any]], comps: Sequence[list[int]], sem,
                 failed: list[str] | None = None) -> list[dict[str, Any]]:
    """LLM splits each candidate component into real groups; returns [{members: [stored ids], context}].

    Records of a component whose call failed are appended to `failed`, so the caller can retry them later."""
    prompt = await asyncio.to_thread(_load_prompt, GROUP_PROMPT)

    async def one(comp: list[int]) -> list[dict[str, Any]]:
        local = {f"r{k + 1}": records[i] for k, i in enumerate(comp)}
        try:
            reply = await _ask(model, prompt, json.dumps([_brief(k, r) for k, r in local.items()], ensure_ascii=False), sem)
        except Exception as exc:  # one bad component must not cost the others
            logger.warning("cross-paper grouping failed: %s: %s", type(exc).__name__, exc)
            if failed is not None:
                failed.extend(r["id"] for r in local.values())
            return []
        out, used = [], set()
        for group in reply.get("groups") or []:
            if not isinstance(group, Mapping):
                continue
            mem = [local[m]["id"] for m in dict.fromkeys(str(x) for x in group.get("members") or [])
                   if m in local and m not in used]
            papers = {local[m]["paper"] for m in group.get("members") or [] if str(m) in local}
            if len(mem) >= 2 and len(papers) >= MIN_PAPERS:
                used.update(str(m) for m in group.get("members") or [])
                out.append({"members": mem, "context": " ".join(str(group.get("context") or "").split())})
        return out

    results = await asyncio.gather(*(one(c) for c in comps))
    return [g for gs in results for g in gs]


async def _induce(model, group: Mapping[str, Any], by_id: Mapping[str, dict[str, Any]], sem) -> list[dict[str, Any]]:
    prompt = await asyncio.to_thread(_load_prompt, INDUCE_PROMPT)
    local_to_id = {f"r{k + 1}": sid for k, sid in enumerate(group["members"])}
    members = {sid: by_id[sid] for sid in group["members"]}
    human = (f"[context] {group.get('context') or ''}\n\n"
             + json.dumps([_full(loc, by_id[sid]) for loc, sid in local_to_id.items()], ensure_ascii=False))
    try:
        reply = await _ask(model, prompt, human, sem)
    except Exception as exc:
        logger.warning("cross-paper induction failed: %s: %s", type(exc).__name__, exc)
        return []
    out = []
    for raw in (reply.get("experiences") or [])[:MAX_FINDINGS_PER_GROUP]:
        if isinstance(raw, Mapping) and (rec := validate_finding(raw, local_to_id, members)) is not None:
            rec["context"] = group.get("context") or ""
            out.append(rec)
    return out


async def _assign(model, state, new_ids: Sequence[str], by_id, vecs: Mapping[str, dict], sem) -> dict[str, str | None]:
    groups = state["groups"]
    if not groups or not new_ids:
        return dict.fromkeys(new_ids)
    offered: dict[str, list[str]] = {}
    for nid in new_ids:
        scored = []
        for gid, g in groups.items():
            sims = [_cos(vecs[nid], vecs[m]) for m in g["members"] if m in vecs]
            if sims:
                scored.append((max(sims) + sum(sims) / len(sims), gid))
        offered[nid] = [gid for _s, gid in sorted(scored, reverse=True)[:ASSIGN_TOP]]
    gids = sorted({g for gs in offered.values() for g in gs})
    if not gids:
        return dict.fromkeys(new_ids)
    prompt = await asyncio.to_thread(_load_prompt, ASSIGN_PROMPT)
    local_new = {f"n{k + 1}": nid for k, nid in enumerate(new_ids)}
    payload = {
        "groups": [{"id": gid, "context": groups[gid].get("context"),
                    "examples": [by_id[m]["item"].get("problem") for m in groups[gid]["members"][:3] if m in by_id]}
                   for gid in gids],
        "new": [_brief(loc, by_id[nid]) for loc, nid in local_new.items()],
    }
    try:
        reply = await _ask(model, prompt, json.dumps(payload, ensure_ascii=False), sem)
    except Exception as exc:
        logger.warning("cross-paper assignment failed: %s: %s", type(exc).__name__, exc)
        return dict.fromkeys(new_ids)
    out: dict[str, str | None] = {}
    raw = reply.get("assign") if isinstance(reply.get("assign"), Mapping) else {}
    for loc, nid in local_new.items():
        gid = raw.get(loc)
        # only a group it was offered, and only if the group would still span its papers
        out[nid] = gid if isinstance(gid, str) and gid in offered[nid] else None
    return out


async def _revise(model, fid: str, finding: dict[str, Any], group, new_members, by_id, sem) -> dict[str, Any]:
    """One operation on one finding. Returns a history entry; mutates `finding` in place on REVISE/ADD_CONFLICT."""
    prompt = await asyncio.to_thread(_load_prompt, REVISE_PROMPT)
    old_ids = [m for m in group["members"] if m not in new_members]
    local_to_id = {f"r{k + 1}": sid for k, sid in enumerate(old_ids + list(new_members))}
    id_to_local = {v: k for k, v in local_to_id.items()}
    rec = finding["record"]
    shown = {**{k: rec.get(k) for k in ("problem", "method", "action", "compared_options", "mechanism", "requires",
                                        "limits_stated", "conflicts")},
             "support": [id_to_local[s] for s in rec.get("source_l1_ids") or [] if s in id_to_local]}
    for key in ("compared_options", "requires", "limits_stated", "conflicts"):
        shown[key] = [{**e, "from": [id_to_local[s] for s in e.get("from") or [] if s in id_to_local]}
                      for e in shown.get(key) or []]
    human = json.dumps({
        "finding": shown,
        "records": [_full(id_to_local[s], by_id[s]) for s in old_ids if s in by_id],
        "new_records": [_full(id_to_local[s], by_id[s]) for s in new_members if s in by_id],
    }, ensure_ascii=False)
    entry: dict[str, Any] = {"ts": time.time(), "finding": fid, "op": "ERROR"}
    try:
        reply = await _ask(model, prompt, human, sem)
    except Exception as exc:
        entry["error"] = str(exc)
        return entry
    op = str(reply.get("op") or "KEEP").upper()
    entry.update(op=op, reason=" ".join(str(reply.get("reason") or "").split())[:300])
    members = {s: by_id[s] for s in local_to_id.values() if s in by_id}
    add = [s for s in _ids(reply.get("add_support"), local_to_id) if s in new_members]
    if op == "RETIRE":
        return entry
    if op == "ADD_CONFLICT":
        conflict = reply.get("conflict")
        if isinstance(conflict, Mapping):
            src = _ids(conflict.get("from"), local_to_id)
            if len(src) >= 2 and str(conflict.get("text") or "").strip():
                rec.setdefault("conflicts", []).append({"text": " ".join(str(conflict["text"]).split()), "from": src})
                add = list(dict.fromkeys(add + [s for s in src if s in new_members]))
            else:
                entry["op"] = "KEEP"
    if op == "REVISE":
        changes = reply.get("changes") if isinstance(reply.get("changes"), Mapping) else {}
        allowed = ("domain", "task", "problem", "method", "action", "compared_options", "mechanism", "requires",
                   "limits_stated")
        merged_raw = {**{k: rec.get(k) for k in allowed}, **{k: v for k, v in changes.items() if k in allowed}}
        # back to local ids for validation
        for key in ("compared_options", "requires", "limits_stated"):
            if key not in changes:
                merged_raw[key] = [{**e, "from": [id_to_local[s] for s in e.get("from") or [] if s in id_to_local]}
                                   for e in rec.get(key) or []]
        merged_raw["support"] = [id_to_local[s] for s in (rec.get("source_l1_ids") or []) + add if s in id_to_local]
        revised = validate_finding(merged_raw, local_to_id, members)
        if revised is None or _text_size(revised) < REVISE_MIN_KEEP * _text_size(rec):
            entry.update(op="KEEP", rejected="revision failed validation or shrank the finding below the floor")
        else:
            finding.setdefault("previous", []).append(rec.copy())
            revised.update(context=rec.get("context") or "", conflicts=rec.get("conflicts") or [])
            finding["record"] = rec = revised
            entry["changed"] = sorted(k for k in changes if k in allowed)
    if add:
        rec["source_l1_ids"] = list(dict.fromkeys((rec.get("source_l1_ids") or []) + add))
        rec["source_papers"] = sorted({by_id[s]["paper"] for s in rec["source_l1_ids"] if s in by_id})
    if entry["op"] in ("REVISE", "ADD_CONFLICT") or add:
        finding["version"] = int(finding.get("version") or 1) + (1 if entry["op"] in ("REVISE", "ADD_CONFLICT") else 0)
    return entry


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


# ----------------------------------------------------------------------------- orchestration


async def run_cross_paper_induction(
    *, memory_dir: str | Path, project_id: str, model: Any, force: str | None = None
) -> dict[str, Any]:
    """Run one trigger if enough new L1 records arrived (or `force` is 'rebuild'/'incremental'). Never raises."""
    started = time.monotonic()
    try:
        records = await asyncio.to_thread(load_l1_records, memory_dir, project_id)
        state = await asyncio.to_thread(load_state, memory_dir, project_id)
        seen = set(state["seen_l1"])
        by_id = {r["id"]: r for r in records}
        new_ids = [r["id"] for r in records if r["id"] not in seen]
        if not force and len(new_ids) < every():
            return {"status": "waiting", "new": len(new_ids), "every": every()}
        mode = force or ("rebuild" if not state["groups"] or (state["triggers"] + 1) % rebuild_every() == 0
                         else "incremental")
        sem = asyncio.Semaphore(concurrency())
        summary: dict[str, Any] = {"mode": mode, "l1": len(records), "new": len(new_ids)}
        logger.info("cross-paper %s: %d L1, %d new", mode, len(records), len(new_ids))
        if mode == "rebuild":
            summary.update(await _rebuild(model, project_id, state, records, by_id, sem))
        else:
            summary.update(await _incremental(model, project_id, state, records, by_id, new_ids, sem))
        state["seen_l1"] = sorted(seen | set(new_ids))
        state["triggers"] += 1
        state["history"].append({"ts": time.time(), "trigger": state["triggers"], **summary})
        await asyncio.to_thread(save, memory_dir, project_id, state)
        summary.update(status="ok", findings=len(state["findings"]), groups=len(state["groups"]),
                       seconds=round(time.monotonic() - started, 1))
    except Exception as exc:  # an aid; never take extraction or the agent down
        logger.warning("cross-paper induction failed", exc_info=True)
        summary = {"status": "error", "error": str(exc), "seconds": round(time.monotonic() - started, 1)}
    try:
        from ..policy.trace import emit_trace

        emit_trace(memory_dir, "cross_paper_induce", project_id=project_id, **summary)
    except Exception:
        pass
    return summary


async def _rebuild(model, project_id, state, records, by_id, sem) -> dict[str, Any]:
    comps = await asyncio.to_thread(candidate_groups, records)
    failed: list[str] = []
    groups = await _group(model, records, comps, sem, failed)
    logger.info("cross-paper rebuild: %d candidate groups -> %d groups", len(comps), len(groups))
    induced = await asyncio.gather(*(_induce(model, g, by_id, sem) for g in groups))
    old = state["findings"]
    new_groups: dict[str, Any] = {}
    new_findings: dict[str, Any] = {}
    matched_old: set[str] = set()
    for k, (group, findings) in enumerate(zip(groups, induced, strict=True)):
        gid = f"g{state['triggers'] + 1}-{k + 1}"
        new_groups[gid] = {"members": group["members"], "context": group["context"], "findings": []}
        for rec in findings:
            best, score = None, 0.0
            for oid, of in old.items():
                s = _jaccard(rec["source_l1_ids"], of["record"].get("source_l1_ids") or [])
                if s > score and oid not in matched_old:
                    best, score = oid, s
            version, lineage = 1, None
            if best and score >= 0.5:
                matched_old.add(best)
                version, lineage = int(old[best].get("version") or 1) + 1, best
            rec["group_id"] = gid
            fid = _finding_id(project_id, rec)
            new_findings[fid] = {"record": rec, "version": version, "lineage": lineage, "group": gid}
            new_groups[gid]["findings"].append(fid)
    retired = [oid for oid in old if oid not in matched_old]
    state["retired"].extend({"id": oid, "ts": time.time(), "why": "not reproduced by rebuild", "record": old[oid]["record"]}
                            for oid in retired)
    # A group whose induction failed keeps its members and is induced again on the next trigger; records whose
    # grouping call failed go to the pool. Neither is lost until the next rebuild.
    state["groups"], state["findings"], state["pool"] = new_groups, new_findings, list(dict.fromkeys(failed))
    return {"candidate_groups": len(comps), "groups_made": len(groups), "induced": len(new_findings),
            "carried_over": len(matched_old), "retired": len(retired)}


async def _incremental(model, project_id, state, records, by_id, new_ids, sem) -> dict[str, Any]:
    vecs = dict(zip([r["id"] for r in records], tfidf_vectors(records), strict=True))
    assignment = await _assign(model, state, new_ids, by_id, vecs, sem)
    gained: dict[str, list[str]] = {}
    for nid, gid in assignment.items():
        if gid:
            gained.setdefault(gid, []).append(nid)
    for gid, nids in gained.items():
        state["groups"][gid]["members"] = list(dict.fromkeys(state["groups"][gid]["members"] + nids))
    empty = [gid for gid, g in state["groups"].items() if not [f for f in g.get("findings") or [] if f in state["findings"]]]
    reinduced = await asyncio.gather(*(_induce(model, state["groups"][gid], by_id, sem) for gid in empty))
    for gid, findings in zip(empty, reinduced, strict=True):
        for rec in findings:
            rec["group_id"] = gid
            fid = _finding_id(project_id, rec)
            state["findings"][fid] = {"record": rec, "version": 1, "lineage": None, "group": gid}
            state["groups"][gid].setdefault("findings", []).append(fid)
    jobs = [(fid, gid, nids) for gid, nids in gained.items() if gid not in empty
            for fid in state["groups"][gid].get("findings") or [] if fid in state["findings"]]
    entries = await asyncio.gather(*(
        _revise(model, fid, state["findings"][fid], state["groups"][gid], nids, by_id, sem) for fid, gid, nids in jobs
    ))
    ops = Counter(e["op"] for e in entries)
    for e in entries:
        state["history"].append(e)
        if e["op"] == "RETIRE":
            f = state["findings"].pop(e["finding"])
            state["retired"].append({"id": e["finding"], "ts": time.time(), "why": e.get("reason"), "record": f["record"]})
            for g in state["groups"].values():
                g["findings"] = [x for x in g.get("findings") or [] if x != e["finding"]]
    # revised findings get a new content id: re-key them
    for fid in list(state["findings"]):
        f = state["findings"][fid]
        nid = _finding_id(project_id, f["record"])
        if nid != fid:
            state["findings"][nid] = {**f, "lineage": fid}
            del state["findings"][fid]
            for g in state["groups"].values():
                g["findings"] = [nid if x == fid else x for x in g.get("findings") or []]
    # unassigned records: pool, grouped like a rebuild once they span papers
    pool_ids = list(dict.fromkeys(state["pool"] + [n for n, g in assignment.items() if not g]))
    pool_ids = [p for p in pool_ids if p in by_id]
    pool_records = [by_id[p] for p in pool_ids]
    made = 0
    if len({r["paper"] for r in pool_records}) >= MIN_PAPERS:
        comps = await asyncio.to_thread(candidate_groups, pool_records)
        groups = await _group(model, pool_records, comps, sem)  # failures stay in the pool below
        induced = await asyncio.gather(*(_induce(model, g, by_id, sem) for g in groups))
        grouped: set[str] = set()
        for k, (group, findings) in enumerate(zip(groups, induced, strict=True)):
            if not findings:
                continue
            gid = f"g{state['triggers'] + 1}-p{k + 1}"
            state["groups"][gid] = {"members": group["members"], "context": group["context"], "findings": []}
            for rec in findings:
                rec["group_id"] = gid
                fid = _finding_id(project_id, rec)
                state["findings"][fid] = {"record": rec, "version": 1, "lineage": None, "group": gid}
                state["groups"][gid]["findings"].append(fid)
                made += 1
            grouped.update(group["members"])
        pool_ids = [p for p in pool_ids if p not in grouped]
    state["pool"] = pool_ids
    return {"assigned": sum(1 for g in assignment.values() if g), "unassigned": sum(1 for g in assignment.values() if not g),
            "ops": dict(ops), "reinduced": sum(len(f) for f in reinduced), "from_pool": made, "pool": len(pool_ids)}


# ----------------------------------------------------------------------------- trigger


async def maybe_run_cross_paper(*, memory_dir: str | Path, project_id: str, model: Any) -> dict[str, Any] | None:
    """Called after extraction stored papers. Runs one trigger when the switch is on and nobody else is running one."""
    try:
        from ...config.settings import get_effective_config

        if not getattr(get_effective_config(), "memory_cross_paper_enabled", False):
            return None
    except Exception:
        return None
    from filelock import AsyncFileLock, Timeout

    if model is None:
        from ...EvoScientist import _ensure_auxiliary_chat_model
        from ...middleware.utils import quiet_model

        model = quiet_model(_ensure_auxiliary_chat_model())
    root = _cross_root(memory_dir, project_id)
    await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
    lock = AsyncFileLock(str(root / ".lock"))
    try:
        await lock.acquire(timeout=0)
    except Timeout:
        return {"status": "busy"}
    try:
        return await run_cross_paper_induction(memory_dir=memory_dir, project_id=project_id, model=model)
    finally:
        await lock.release()


_background: set[asyncio.Task[Any]] = set()


def schedule_cross_paper(*, memory_dir: str | Path, project_id: str, model: Any = None) -> None:
    """Start a trigger check in the background so extraction never waits on induction. Safe to call often:
    below the trigger it returns at once, and the file lock lets only one induction run per project."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(maybe_run_cross_paper(memory_dir=memory_dir, project_id=project_id, model=model))
    _background.add(task)
    task.add_done_callback(_background.discard)


__all__ = ["candidate_groups", "load_l1_records", "load_state", "maybe_run_cross_paper", "run_cross_paper_induction",
           "schedule_cross_paper", "tfidf_vectors", "validate_finding"]
