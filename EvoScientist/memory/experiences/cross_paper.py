"""L1 -> L2 induction over clusters of L1 records. The only source of L2: a paper yields L1 records alone.

Every L1 record of the project belongs to one cluster. A cluster is kept as a centroid over a lexical view of its
members (problem, method aliases, task, requires, action, mechanism); where the records came from does not matter.

* **assign** -- a record not yet seen joins the cluster whose centroid is nearest, when the cosine reaches the join
  threshold; otherwise it starts a cluster of its own. A centre is kept even for one record, so later related records
  have somewhere to gather.
* **split** -- a cluster larger than the size cap, or one whose members sit on average further from its centroid than
  the cohesion floor allows, is bisected (2-means seeded by its two most distant members) until every part fits.
* **induce** -- a cluster gets findings once it has `min_size()` members (at most `MAX_FINDINGS`). A smaller cluster has
  a centre but no findings yet.
* **update** -- new members of a cluster that already has findings wait. The cluster is updated once the new members
  since its last induction reach `update_share()` of its size then (at least `UPDATE_MIN_NEW`), or when it was split.
  An update shows the model the current findings with all members and asks it to revise them, not to start over:
  claims the records still support stay as written, so a finding keeps its wording and its id across updates.

Grounding is enforced in code: every `support` / `from` id must be a member, a finding rests on at least `min_size()`
members, a stated limit summarizes at least two members' own `limits_stated`, and most numbers in the claim and options
must appear in the cited records. A new finding takes over the version of the old finding of the cluster it overlaps
most (Jaccard over support >= 0.5); an old finding nothing takes over is retired, not deleted. The findings live in
`experiences/projects/<P>/_cross_paper/l2.json`, which the existing listing reads like one more paper directory, so
retrieval, rendering and plan review need no new path.
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
INDUCE_PROMPT = "l2_from_l1_cluster.md"
STATE_VERSION = 2


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, "").strip() or default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


# Measured on the judgelab library (2,697 L1 records, 587 papers, the TF-IDF view below): join 0.30 with min size 3
# gives 168 clusters that get findings, covering 24% of the records; the rest wait for more members.
def join_min() -> float:
    return _env_float("EVOSCIENTIST_L2_CLUSTER_JOIN", 0.30)


def min_size() -> int:
    return _env_int("EVOSCIENTIST_L2_CLUSTER_MIN_SIZE", 3)


def max_size() -> int:
    return _env_int("EVOSCIENTIST_L2_CLUSTER_MAX_SIZE", 12)


def min_cohesion() -> float:
    return _env_float("EVOSCIENTIST_L2_CLUSTER_COHESION", 0.40)


def update_share() -> float:
    """New members, as a share of the size at the last induction, that make a cluster with findings due again."""
    return _env_float("EVOSCIENTIST_L2_UPDATE_SHARE", 0.5)


UPDATE_MIN_NEW = 2


def every() -> int:
    """New L1 records needed before a trigger runs."""
    return _env_int("EVOSCIENTIST_CROSS_PAPER_EVERY", 15)


def concurrency() -> int:
    return _env_int("EVOSCIENTIST_CROSS_PAPER_CONCURRENCY", 6)


MAX_FINDINGS = 3
CALL_TIMEOUT_S = 900
SPLIT_ROUNDS = 10

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


def _mechanism_text(item: Mapping[str, Any]) -> str:
    value = item.get("mechanism")
    return str(value.get("text") or "") if isinstance(value, Mapping) else str(value or "")


def _view(item: Mapping[str, Any]) -> str:
    methods = " ".join(str(m) for m in item.get("method") or [])
    requires = " ".join(str(r.get("text") or "") for r in item.get("requires") or [] if isinstance(r, Mapping))
    problem = str(item.get("problem") or "")
    return " ".join([problem, problem, methods, methods, str(item.get("task") or ""), requires,
                     str(item.get("action") or ""), _mechanism_text(item)])


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9\-]+", text.lower()) if len(w) > 2 and w not in _STOP]


Vec = dict[str, float]


def _unit(v: Mapping[str, float]) -> Vec:
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / norm for w, x in v.items()}


def tfidf_vectors(records: Sequence[Mapping[str, Any]]) -> list[Vec]:
    toks = [_tokens(_view(r["item"])) for r in records]
    n = len(records)
    df = Counter(w for t in toks for w in set(t))
    return [_unit({w: (1 + math.log(c)) * math.log((n + 1) / (df[w] + 1)) for w, c in Counter(t).items()})
            for t in toks]


def _cos(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def _centroid(members: Sequence[str], vecs: Mapping[str, Vec]) -> Vec:
    total: Counter[str] = Counter()
    for m in members:
        total.update(vecs[m])
    return _unit(total)


def _cohesion(members: Sequence[str], vecs: Mapping[str, Vec]) -> float:
    centre = _centroid(members, vecs)
    return sum(_cos(vecs[m], centre) for m in members) / len(members)


# ----------------------------------------------------------------------------- clustering


def _bisect(members: list[str], vecs: Mapping[str, Vec]) -> tuple[list[str], list[str]]:
    """2-means seeded by the member furthest from the centroid and the member furthest from that one."""
    centre = _centroid(members, vecs)
    a = min(members, key=lambda m: _cos(vecs[m], centre))
    b = min(members, key=lambda m: _cos(vecs[m], vecs[a]))
    ca, cb = vecs[a], vecs[b]
    left, right = members, []
    for _ in range(SPLIT_ROUNDS):
        left = [m for m in members if _cos(vecs[m], ca) >= _cos(vecs[m], cb)]
        right = [m for m in members if _cos(vecs[m], ca) < _cos(vecs[m], cb)]
        if not left or not right:
            break
        ca, cb = _centroid(left, vecs), _centroid(right, vecs)
    return left, right


def _needs_split(members: Sequence[str], vecs: Mapping[str, Vec]) -> bool:
    return len(members) > max_size() or (len(members) >= min_size() and _cohesion(members, vecs) < min_cohesion())


def split(members: list[str], vecs: Mapping[str, Vec]) -> list[list[str]]:
    """The parts a cluster breaks into, each within the size cap and cohesive enough (or no longer divisible)."""
    done, todo = [], [members]
    while todo:
        part = todo.pop()
        if _needs_split(part, vecs):
            left, right = _bisect(part, vecs)
            if left and right:
                todo += [left, right]
                continue
        done.append(part)
    return done


def _new_cluster_id(state: dict[str, Any]) -> str:
    state["next_cluster"] = int(state.get("next_cluster") or 0) + 1
    return f"c{state['next_cluster']}"


def assign(state: dict[str, Any], new_ids: Sequence[str], vecs: Mapping[str, Vec]) -> set[str]:
    """Put each new record into its nearest cluster or a new one, then split what grew too big or too loose.

    Centroids are recomputed from the current vectors, so a cluster's centre follows the library's vocabulary.
    Returns the ids of the clusters whose membership changed (including the parts of a split cluster).
    """
    clusters: dict[str, dict[str, Any]] = state["clusters"]
    centres = {cid: _centroid(c["members"], vecs) for cid, c in clusters.items() if c["members"]}
    changed: set[str] = set()
    for nid in new_ids:
        best, score = None, 0.0
        for cid, centre in centres.items():
            s = _cos(vecs[nid], centre)
            if s > score:
                best, score = cid, s
        if best is not None and score >= join_min():
            clusters[best]["members"].append(nid)
            centres[best] = _centroid(clusters[best]["members"], vecs)
        else:
            best = _new_cluster_id(state)
            clusters[best] = {"members": [nid], "findings": [], "induced_size": 0}
            centres[best] = vecs[nid]
        changed.add(best)
    for cid in sorted(changed):
        parts = split(clusters[cid]["members"], vecs)
        if len(parts) == 1:
            continue
        old_findings = clusters.pop(cid)["findings"]
        changed.discard(cid)
        for k, part in enumerate(sorted(parts, key=len, reverse=True)):
            # The largest part inherits the findings and revises them; the others start without. `split` marks the
            # part due now: its findings were induced from members it no longer has.
            pid = _new_cluster_id(state)
            clusters[pid] = {"members": part, "findings": old_findings if k == 0 else [], "induced_size": 0,
                             "split": True}
            changed.add(pid)
    return changed


def is_due(cluster: Mapping[str, Any], *, has_findings: bool) -> bool:
    """Whether a cluster's findings should be (re)written on this trigger."""
    size = len(cluster["members"])
    if size < min_size():
        return False
    if not has_findings or cluster.get("split"):
        return True
    done = int(cluster.get("induced_size") or 0)
    return size - done >= max(UPDATE_MIN_NEW, math.ceil(update_share() * done))


# ----------------------------------------------------------------------------- LLM plumbing


def _load_prompt(name: str) -> str:
    for directory in _prompt_dir_candidates():
        path = directory / name
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"L2 induction prompt {name} not found")


def _json_object(text: str) -> dict[str, Any]:
    cand = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cand, flags=re.DOTALL | re.IGNORECASE)
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


def _full(local: str, rec: Mapping[str, Any]) -> dict[str, Any]:
    item = rec["item"]
    keep = ("domain", "task", "problem", "method", "requires", "source_setting", "limits_stated", "action", "effect",
            "mechanism", "comparison")
    return {"id": local, "paper": rec["paper"], **{k: item.get(k) for k in keep}}


# ----------------------------------------------------------------------------- validation


def _ids(value: Any, allowed: Mapping[str, str]) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(allowed[str(v)] for v in value if str(v) in allowed))


def validate_finding(
    raw: Mapping[str, Any], local_to_id: Mapping[str, str], members: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    """Map local ids to stored ids and enforce the grounding rules; None when the finding does not survive."""
    for key in ("domain", "task", "problem", "action"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            return None
    method = raw.get("method")
    if not isinstance(method, list) or not [m for m in method if isinstance(m, str) and m.strip()]:
        return None
    support = _ids(raw.get("support"), local_to_id)
    if len(support) < min_size():
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


# ----------------------------------------------------------------------------- state


def _empty_state() -> dict[str, Any]:
    return {"version": STATE_VERSION, "triggers": 0, "next_cluster": 0, "clusters": {}, "findings": {},
            "pending": [], "retired": [], "history": []}


def load_state(memory_dir: str | Path, project_id: str) -> dict[str, Any]:
    """The clustering state. A state from the older grouping scheme is not reused: its records are clustered afresh,
    and its findings are moved to `retired`."""
    state = read_json(_cross_root(memory_dir, project_id) / "state.json")
    if isinstance(state, dict) and state.get("version") == STATE_VERSION:
        return state
    fresh = _empty_state()
    if isinstance(state, dict) and isinstance(state.get("findings"), Mapping):
        fresh["retired"] = list(state.get("retired") or []) + [
            {"id": fid, "ts": time.time(), "why": "replaced by centroid clustering", "record": f.get("record")}
            for fid, f in state["findings"].items() if isinstance(f, Mapping)
        ]
    return fresh


def _finding_id(project_id: str, record: Mapping[str, Any]) -> str:
    """Content id of the claim (domain, task, problem, action), scoped to its cluster: two clusters that word a finding
    the same way still hold two records, and a re-induced finding that keeps its claim keeps its id."""
    return _experience_id(project_id=project_id, paper_key=f"{CROSS_DIR}/{record.get('cluster_id') or ''}",
                          level="l2", index=0, item=record)


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
        "title": f"Findings induced from L1 clusters ({len(findings)} from {len(papers)} papers)", "url": "",
        "domain_arxiv": None, "extraction_schema": "v2", "origin": "cross_paper",
        "supporting_papers": papers, "extracted_at": datetime.now(UTC).isoformat(),
    })
    refresh_experience_catalog(memory_dir=memory_dir, project_id=project_id)


# ----------------------------------------------------------------------------- induction


def _shown_finding(record: Mapping[str, Any], id_to_local: Mapping[str, str]) -> dict[str, Any]:
    """A stored finding with its record ids mapped back to the local ids of this call."""
    def local(ids: Any) -> list[str]:
        return [id_to_local[s] for s in ids or [] if s in id_to_local]

    shown = {k: record.get(k) for k in ("domain", "task", "problem", "method", "action", "mechanism")}
    for key in ("compared_options", "requires", "limits_stated", "conflicts"):
        shown[key] = [{**e, "from": local(e.get("from"))} for e in record.get(key) or [] if isinstance(e, Mapping)]
    shown["support"] = local(record.get("source_l1_ids"))
    return shown


async def induce(model: Any, members: Sequence[str], by_id: Mapping[str, dict[str, Any]], sem: asyncio.Semaphore,
                 current: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]] | None:
    """Findings for one cluster; None when the call failed (the cluster is retried on the next trigger).

    `current` holds the cluster's findings so far. When given, the model revises them against all members rather
    than writing new ones from scratch."""
    prompt = await asyncio.to_thread(_load_prompt, INDUCE_PROMPT)
    local_to_id = {f"r{k + 1}": sid for k, sid in enumerate(members)}
    id_to_local = {v: k for k, v in local_to_id.items()}
    cited = {sid: by_id[sid] for sid in members}
    records = [_full(loc, by_id[sid]) for loc, sid in local_to_id.items()]
    if current:
        human = json.dumps({"current_findings": [_shown_finding(f, id_to_local) for f in current],
                            "records": records}, ensure_ascii=False)
    else:
        human = json.dumps({"records": records}, ensure_ascii=False)
    try:
        reply = await _ask(model, prompt, human, sem)
    except Exception as exc:  # one bad cluster must not cost the others
        logger.warning("L2 induction failed: %s: %s", type(exc).__name__, exc)
        return None
    out = []
    for raw in (reply.get("experiences") or [])[:MAX_FINDINGS]:
        if isinstance(raw, Mapping) and (rec := validate_finding(raw, local_to_id, cited)) is not None:
            out.append(rec)
    return out


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def _retire(state: dict[str, Any], fid: str, why: str) -> None:
    state["retired"].append({"id": fid, "ts": time.time(), "why": why, "record": state["findings"].pop(fid)["record"]})


def _replace_findings(state: dict[str, Any], project_id: str, cid: str, records: list[dict[str, Any]]) -> dict[str, int]:
    """Swap a cluster's findings for `records`, carrying versions over by support overlap; retire the rest."""
    cluster = state["clusters"][cid]
    old = [fid for fid in cluster.get("findings") or [] if fid in state["findings"]]
    taken: set[str] = set()
    new_ids: list[str] = []
    for rec in records:
        best, score = None, 0.0
        for oid in old:
            s = _jaccard(rec["source_l1_ids"], state["findings"][oid]["record"].get("source_l1_ids") or [])
            if s > score and oid not in taken:
                best, score = oid, s
        version, lineage = 1, None
        if best is not None and score >= 0.5:
            taken.add(best)
            version, lineage = int(state["findings"][best].get("version") or 1) + 1, best
        rec["cluster_id"] = cid
        fid = _finding_id(project_id, rec)
        new_ids.append(fid)
        existing = state["findings"].get(fid)
        if existing is not None and existing["record"] == rec:   # nothing changed: keep it as it is
            continue
        if lineage == fid:   # same claim, new support or options: the record is updated in place under its id
            lineage = existing.get("lineage")
        state["findings"][fid] = {"record": rec, "version": version, "lineage": lineage, "cluster": cid}
    retired = 0
    for oid in old:
        if oid in new_ids:
            continue
        if oid in taken:
            state["findings"].pop(oid)   # superseded by its successor, which records it as `lineage`
        else:
            _retire(state, oid, "its cluster changed and no new finding took it over")
            retired += 1
    cluster["findings"] = new_ids
    return {"made": len(records), "carried": len(taken), "retired": retired}


def _retire_orphans(state: dict[str, Any]) -> int:
    """Findings whose cluster no longer holds them (a split part that did not inherit them) are retired."""
    live = {fid for c in state["clusters"].values() for fid in c.get("findings") or []}
    orphans = [fid for fid in state["findings"] if fid not in live]
    for fid in orphans:
        _retire(state, fid, "its cluster was split")
    return len(orphans)


# ----------------------------------------------------------------------------- orchestration


async def run_cross_paper_induction(
    *, memory_dir: str | Path, project_id: str, model: Any, force: bool = False
) -> dict[str, Any]:
    """Run one trigger when enough new L1 records arrived, a failed cluster is pending, or `force`. Never raises."""
    started = time.monotonic()
    try:
        records = await asyncio.to_thread(load_l1_records, memory_dir, project_id)
        state = await asyncio.to_thread(load_state, memory_dir, project_id)
        by_id = {r["id"]: r for r in records}
        # records that left the library (re-extracted or dropped by the gate) leave their clusters
        for c in state["clusters"].values():
            c["members"] = [m for m in c["members"] if m in by_id]
        state["clusters"] = {cid: c for cid, c in state["clusters"].items() if c["members"]}
        seen = {m for c in state["clusters"].values() for m in c["members"]}
        new_ids = [r["id"] for r in records if r["id"] not in seen]
        pending = [cid for cid in state.get("pending") or [] if cid in state["clusters"]]
        if not force and len(new_ids) < every() and not pending:
            return {"status": "waiting", "new": len(new_ids), "every": every()}
        vecs = dict(zip([r["id"] for r in records], tfidf_vectors(records), strict=True))
        changed = (assign(state, new_ids, vecs) | set(pending)) & set(state["clusters"])
        orphans = _retire_orphans(state)
        def findings_of(cid: str) -> list[dict[str, Any]]:
            return [state["findings"][f]["record"] for f in state["clusters"][cid].get("findings") or []
                    if f in state["findings"]]

        due = sorted(cid for cid in changed if is_due(state["clusters"][cid], has_findings=bool(findings_of(cid))))
        waiting = sum(1 for cid in changed if cid not in due and findings_of(cid))
        logger.info("L2 induction: %d L1, %d new, %d clusters due", len(records), len(new_ids), len(due))
        sem = asyncio.Semaphore(concurrency())
        results = await asyncio.gather(*(
            induce(model, state["clusters"][cid]["members"], by_id, sem, current=findings_of(cid)) for cid in due))
        totals: Counter[str] = Counter()
        failed: list[str] = []
        for cid, found in zip(due, results, strict=True):
            if found is None:
                failed.append(cid)
                continue
            totals.update(_replace_findings(state, project_id, cid, found))
            cluster = state["clusters"][cid]
            cluster["induced_size"] = len(cluster["members"])
            cluster.pop("split", None)
        # A cluster that changed but is not due stays as it is; it is checked again when it next gains members.
        # A failed call is retried on the next trigger, even below `every`.
        state["pending"] = failed
        state["triggers"] += 1
        sizes = [len(c["members"]) for c in state["clusters"].values()]
        summary: dict[str, Any] = {
            "l1": len(records), "new": len(new_ids), "clusters": len(sizes),
            "clusters_eligible": sum(1 for s in sizes if s >= min_size()),
            "induced": len(due) - len(failed), "failed": len(failed), "waiting_for_more": waiting,
            "findings_made": totals["made"], "carried_over": totals["carried"], "retired": totals["retired"] + orphans,
        }
        state["history"].append({"ts": time.time(), "trigger": state["triggers"], **summary})
        await asyncio.to_thread(save, memory_dir, project_id, state)
        summary.update(status="ok", findings=len(state["findings"]), seconds=round(time.monotonic() - started, 1))
    except Exception as exc:  # an aid; never take extraction or the agent down
        logger.warning("L2 induction failed", exc_info=True)
        summary = {"status": "error", "error": str(exc), "seconds": round(time.monotonic() - started, 1)}
    try:
        from ..policy.trace import emit_trace

        emit_trace(memory_dir, "cross_paper_induce", project_id=project_id, **summary)
    except Exception:
        pass
    return summary


# ----------------------------------------------------------------------------- trigger


async def maybe_run_cross_paper(*, memory_dir: str | Path, project_id: str, model: Any) -> dict[str, Any] | None:
    """Called after extraction stored papers. Runs one trigger when the switch is on and nobody else is running one."""
    try:
        from ...config.settings import get_effective_config

        if not getattr(get_effective_config(), "memory_cross_paper_enabled", True):
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


__all__ = ["assign", "induce", "load_l1_records", "load_state", "maybe_run_cross_paper", "run_cross_paper_induction",
           "schedule_cross_paper", "split", "tfidf_vectors", "validate_finding"]
