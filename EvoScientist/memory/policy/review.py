"""Plan review: read a finished plan next to the task and the experience library, and say where the library bears on it.

This replaces step-by-step coaching for the plan-writer. The coach used to watch the agent's last few actions and push
guidance into the next model call. Here the agent finishes a plan first; a reviewer then reads three things -- the task,
the whole plan, and the experience records found for the plan's design choices -- and returns suggestions, each tied to
the record ids it rests on. The agent decides what to do with them.

Two rules are enforced in code, not left to the prompt:
* a suggestion without at least one valid record id is dropped (the ids must be among the records the reviewer was shown);
* a review of an empty library, or of an empty plan, costs no model call.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...utils import format_message_content
from .prompts import load_policy_prompt
from .select import fetch_candidates, gather_candidates
from .trace import emit_trace_async, response_usage

logger = logging.getLogger(__name__)

REVIEW_PROMPT_FILENAME = "policy_review.md"
KINDS = ("change", "add", "check", "confirm")
DEFAULT_MAX_SUGGESTIONS = 6
# How many records the reviewer sees, and how many query facets are drawn from the plan.
DEFAULT_RECORD_LIMIT = 10
MAX_PLAN_CHARS = 60000
MAX_RECORD_CHARS = 9000
# The cross-field leg of the review: a few other-field candidates, at most two hypotheses.
CROSS_DOMAIN_HEADING = "## Cross-field transfer hypotheses (hypotheses, not evidence)"
CROSS_DOMAIN_CANDIDATES = 12
CROSS_DOMAIN_LIMIT = 2
_ID = re.compile(r"\bE-[0-9a-f]{6,}\b")
_HEADING = re.compile(r"^(#{1,3})\s*(.+?)\s*$", re.M)


def plan_queries(plan: str, limit: int = 6) -> list[str]:
    """Short search strings drawn from the plan's own text: its hypothesis line and its section headings.

    Retrieval is lexical, so the queries are made of the plan's own words. No model decides them.
    """
    text = plan[:MAX_PLAN_CHARS]
    out: list[str] = []
    m = re.search(r"##\s*Direction and hypothesis\s*\n(.+?)(?=\n##\s|\Z)", text, re.S | re.I)
    if m:
        out.append(" ".join(m.group(1).split())[:400])
    for h in _HEADING.finditer(text):
        title = h.group(2).strip()
        if len(title) > 12 and not re.match(r"(plan|evidence|risks|budget|direction)", title, re.I):
            out.append(title[:160])
    # a few of the longest bullet lines under "What to build": they name the actual design choices
    m = re.search(r"##\s*What to build\s*\n(.+?)(?=\n##\s|\Z)", text, re.S | re.I)
    if m:
        lines = [ln.strip("-*0123456789. \t") for ln in m.group(1).splitlines() if len(ln.strip()) > 60]
        lines.sort(key=len, reverse=True)
        out.extend(" ".join(ln.split())[:300] for ln in lines[:limit])
    seen, uniq = set(), []
    for q in out:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(q)
    return uniq[:limit]


def merge_round_robin(per_query: Sequence[Sequence[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Merge each query's ranked candidates by taking rank 1 of every query, then rank 2, and so on.

    Scores from different queries are not comparable (a three-word heading and a sixty-word hypothesis line give
    condition scores on different scales), so they are never sorted against each other. Every query that found
    something gets its best record in before any query gets its second. Duplicates keep their first position.
    """
    merged: dict[str, dict[str, Any]] = {}
    depth = max((len(q) for q in per_query), default=0)
    for rank in range(depth):
        for candidates in per_query:
            if rank < len(candidates):
                merged.setdefault(candidates[rank]["id"], candidates[rank])
    return merged


def _rendered(c: dict[str, Any]) -> str:
    from ..experiences.render import render_record

    return render_record(c["record"])


def record_sizes(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """For the trace: how big each shown record is and whether the reviewer got it cut short."""
    out = []
    for c in candidates:
        n = len(_rendered(c))
        out.append({"id": c["id"], "chars": n, "truncated": n > MAX_RECORD_CHARS})
    return out


def _render_records(candidates: Sequence[dict[str, Any]]) -> str:
    blocks = []
    for c in candidates:
        text = _rendered(c)
        if len(text) > MAX_RECORD_CHARS:
            text = text[:MAX_RECORD_CHARS] + "\n... [record truncated]"
        blocks.append(text)
    return "\n\n".join(blocks)


def parse_review(raw: str, allowed_ids: set[str], max_suggestions: int) -> dict[str, Any]:
    """Parse the reviewer's JSON and drop every suggestion that does not rest on a record it was shown."""
    cand = raw.strip()
    if cand.startswith("```"):
        cand = cand.split("\n", 1)[-1] if "\n" in cand else cand
        cand = cand.removesuffix("```").strip().removeprefix("json").strip()
    try:
        payload = json.loads(cand)
    except ValueError:
        start = cand.find("{")
        if start < 0:
            raise ValueError("review did not return a JSON object")
        payload, _ = json.JSONDecoder().raw_decode(cand[start:])
    if not isinstance(payload, dict):
        raise ValueError("review output must be a JSON object")
    kept, dropped = [], 0
    for item in payload.get("suggestions") or []:
        if not isinstance(item, dict):
            dropped += 1
            continue
        srcs = [s for s in (item.get("sources") or []) if isinstance(s, str) and s in allowed_ids]
        about, text, because = (" ".join(str(item.get(k) or "").split()) for k in ("about", "suggestion", "because"))
        if not srcs or not about or not text or not because:
            dropped += 1
            continue
        kind = str(item.get("kind") or "check").strip().lower()
        kind = kind if kind in KINDS else "check"
        why = " ".join(str(item.get("why_applicable") or "").split())
        # A suggestion must say which condition in the record holds here.
        # "confirm" may omit it: it only says the plan already does what the
        # record supports.
        if kind != "confirm" and not why:
            dropped += 1
            continue
        kept.append({"about": about, "suggestion": text, "because": because,
                     "sources": list(dict.fromkeys(srcs)), "kind": kind,
                     "why_applicable": why})
    return {"suggestions": kept[:max_suggestions], "dropped_without_source": dropped,
            "no_suggestion_reason": " ".join(str(payload.get("no_suggestion_reason") or "").split())[:300]}


def render_review(review: dict[str, Any], titles: dict[str, str] | None = None) -> str:
    """The review as text for the plan-writer: numbered suggestions, each with the records it rests on."""
    titles = titles or {}
    sugg = review.get("suggestions") or []
    if not sugg:
        why = review.get("no_suggestion_reason") or "no stored experience bears on this plan"
        return f"Plan review: no suggestions ({why})."
    lines = ["Plan review against stored experience. Each item rests on the records named and says why it would apply "
             "here. The reviewer saw only your task description, the plan and these records, "
             "not the tree, your measurements or the budget. You decide: check `Why applicable here` against what you know, "
             "take the item and revise the plan file, or refuse it. Check a record with `read_memory` before relying on it."]
    for i, s in enumerate(sugg, 1):
        refs = ", ".join(f"{e}" + (f" ({titles[e]})" if e in titles else "") for e in s["sources"])
        lines.append(
            f"\n{i}. [{s['kind']}] About: {s['about']}"
            f"\n   Suggestion: {s['suggestion']}"
            f"\n   Because: {s['because']}"
            f"\n   Why applicable here: {s.get('why_applicable') or '(not stated)'}"
            f"\n   Sources: {refs}"
        )
    return "\n".join(lines)


def _section_text(plan: str, name: str) -> str:
    match = re.search(rf"##\s*{re.escape(name)}\s*\n(.+?)(?=\n##\s|\Z)", plan, re.S | re.I)
    return match.group(1).strip() if match else ""


def plan_design_point(plan: str, task: str) -> str:
    """The plan's design point, without asking the caller to decide which field it belongs to."""
    parts = [
        _section_text(plan, "Direction and hypothesis"),
        _section_text(plan, "What to build"),
        _section_text(plan, "Open choices"),
    ]
    body = "\n\n".join(part for part in parts if part) or plan[:3000]
    return ((" ".join(task.split())[:1200] + "\n\n") if task.strip() else "") + body[:4000]


def same_field_domains(shown: Sequence[dict[str, Any]], *, min_count: int = 2, max_labels: int = 3) -> list[str]:
    """Domains the plan itself is working in, from the records the normal review was shown.

    Used only to keep the cross-field leg away from the caller's own field when that is knowable;
    a library with too few labelled records yields no exclusion.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for candidate in shown:
        record = candidate.get("record") if isinstance(candidate, dict) else None
        experience = record.get("experience") if isinstance(record, dict) else None
        domain = str((experience or {}).get("domain") or "").strip() if isinstance(experience, dict) else ""
        if domain:
            counts[domain] += 1
    return [domain for domain, count in counts.most_common(max_labels) if count >= min_count]


async def cross_domain_section(
    *,
    memory_dir: str | Path,
    project_id: str,
    plan: str,
    task: str,
    model: Any,
    same_domains: Sequence[str] = (),
    limit: int = CROSS_DOMAIN_LIMIT,
    candidates: int = CROSS_DOMAIN_CANDIDATES,
) -> dict[str, Any]:
    """The review's second leg: other-field practices that pass the transfer check.

    It is a section of the same review, not a tool the agent has to remember: the main agent sees it wherever it
    reviews a plan. The result is a hypothesis, rendered with its object mappings and transfer path; it is never
    presented as a stored finding.
    """
    try:
        from ..experiences.cross_domain import search_cross_domain
        from ..experiences.render import render_cross_domain

        result = await search_cross_domain(
            memory_dir=memory_dir,
            project_id=project_id,
            design_point=plan_design_point(plan, task),
            model=model,
            exclude_domain=(same_domains[0] if same_domains else ""),
            context=" ".join(task.split())[:1500],
            limit=limit,
            candidates=candidates,
        )
        if same_domains:
            own = set(same_domains)
            result["kept"] = [item for item in result.get("kept") or [] if str(item.get("domain") or "") not in own]
        kept = result.get("kept") or []
        if kept:
            text = CROSS_DOMAIN_HEADING + "\n\n" + render_cross_domain(result)
        else:
            text = (CROSS_DOMAIN_HEADING
                    + f"\n\nNo cross-field transfer passed the check ({result.get('considered', 0)} candidate(s) considered).")
        return {"text": text, "considered": result.get("considered", 0), "kept": len(kept),
                "dropped": result.get("dropped") or {}, "gate_failed": result.get("gate_failed", 0),
                "goal": result.get("goal", "")}
    except Exception as error:  # the second leg is advice; it must never take the review down
        return {"text": CROSS_DOMAIN_HEADING + f"\n\nUnavailable ({type(error).__name__}: {error}).",
                "error": f"{type(error).__name__}: {error}"}


async def review_plan(
    *,
    memory_dir: str | Path,
    project_id: str,
    task: str,
    plan: str,
    model: Any,
    max_suggestions: int = DEFAULT_MAX_SUGGESTIONS,
    record_limit: int = DEFAULT_RECORD_LIMIT,
    call_id: str | None = None,
    cross_domain: bool = False,
    cross_domain_limit: int = CROSS_DOMAIN_LIMIT,
    cross_domain_candidates: int = CROSS_DOMAIN_CANDIDATES,
) -> dict[str, Any]:
    """Review ``plan`` against the library. Returns the parsed review plus what was looked at.

    With ``cross_domain``, a second leg runs alongside the normal review and is appended to the same text:
    other-field records that pass the transfer check, labelled as hypotheses. The caller (the main agent) never
    chooses a domain: it asks for a review and gets both sections.
    """
    import asyncio

    started = time.monotonic()
    if not plan.strip():
        return {"status": "empty_plan", "suggestions": [], "text": "Plan review: the plan is empty."}
    queries = plan_queries(plan)
    per_query: list[list[dict[str, Any]]] = []
    for q in queries:
        per_query.append(await asyncio.to_thread(
            gather_candidates, memory_dir=memory_dir, project_id=project_id, query=q, limit=record_limit, call_id=call_id
        ))
    retrieved = merge_round_robin(per_query)
    # The plan may already name records: those go first so the cut below never drops them.
    found: dict[str, dict[str, Any]] = {}
    named = list(dict.fromkeys(_ID.findall(plan)))
    if named:
        for c in await asyncio.to_thread(
            fetch_candidates, memory_dir=memory_dir, project_id=project_id, ids=named, call_id=call_id
        ):
            found.setdefault(c["id"], c)
    for key, c in retrieved.items():
        found.setdefault(key, c)
    # `found` is already in round-robin order across the plan's queries: each
    # query's best record first, then each query's second, and so on.
    shown = list(found.values())[: max(record_limit * 2, 12)]
    cross_task: asyncio.Task[dict[str, Any]] | None = None
    if cross_domain:
        cross_task = asyncio.create_task(cross_domain_section(
            memory_dir=memory_dir, project_id=project_id, plan=plan, task=task, model=model,
            same_domains=same_field_domains(shown), limit=cross_domain_limit, candidates=cross_domain_candidates,
        ))
    if not found:
        cross = await cross_task if cross_task is not None else None
        text = "Plan review: no stored experience matched this plan."
        if cross is not None:
            text += "\n\n" + cross["text"]
        return {"status": "no_records", "suggestions": [], "queries": queries, "text": text,
                "cross_domain": cross}
    prompt = await asyncio.to_thread(load_policy_prompt, REVIEW_PROMPT_FILENAME)
    filled = prompt.format(task=task.strip()[:20000], plan=plan[:MAX_PLAN_CHARS], records=_render_records(shown),
                           max_suggestions=max_suggestions)
    try:
        response = await model.ainvoke([SystemMessage(content=filled), HumanMessage(content="Review the plan now.")])
    except BaseException:
        if cross_task is not None:
            cross_task.cancel()
        raise
    raw = format_message_content(response).strip()
    allowed = {c["id"] for c in shown}
    attempts = 1
    try:
        review = parse_review(raw, allowed, max_suggestions)
    except ValueError as error:
        # One correction retry: a truncated or prose reply is often recoverable, and a lost review is a lost
        # chance to use the library. The second failure is reported as before.
        correction = (f"\n\n[Correction] Your previous reply could not be parsed as the required JSON object "
                      f"({error}). Return only the JSON object, with no prose or Markdown fences.")
        try:
            response = await model.ainvoke([SystemMessage(content=filled),
                                            HumanMessage(content="Review the plan now." + correction)])
        except BaseException:
            if cross_task is not None:
                cross_task.cancel()
            raise
        raw = format_message_content(response).strip()
        attempts = 2
        try:
            review = parse_review(raw, allowed, max_suggestions)
        except ValueError as second:
            cross = await cross_task if cross_task is not None else None
            text = f"Plan review: the reviewer's output could not be read ({second})."
            if cross is not None:
                text += "\n\n" + cross["text"]
            await emit_trace_async(memory_dir, "plan_review", call_id=call_id, ok=False, error=str(second),
                                   raw_output=raw, prompt_chars=len(filled), attempts=attempts, cross_domain=cross,
                                   seconds=round(time.monotonic() - started, 2))
            return {"status": "unreadable", "suggestions": [], "queries": queries, "text": text,
                    "cross_domain": cross}
    titles = {}
    for c in shown:
        paper = c["record"].get("paper") if isinstance(c.get("record"), dict) else None
        if isinstance(paper, dict) and paper.get("title"):
            titles[c["id"]] = " ".join(str(paper["title"]).split())[:70]
    cross = await cross_task if cross_task is not None else None
    text = render_review(review, titles)
    if cross is not None:
        text += "\n\n" + cross["text"]
    await emit_trace_async(memory_dir, "plan_review", call_id=call_id, ok=True, queries=queries, shown=[c["id"] for c in shown],
                           shown_sizes=record_sizes(shown), found=len(found), prompt_chars=len(filled),
                           plan_chars=len(plan), task_chars=len(task), raw_output=raw,
                           suggestions=review["suggestions"], dropped_without_source=review["dropped_without_source"],
                           cross_domain=cross, attempts=attempts,
                           seconds=round(time.monotonic() - started, 2), usage=response_usage(response))
    return {"status": "ok", "queries": queries, "shown": [c["id"] for c in shown], "text": text,
            "cross_domain": cross, **review}


__all__ = ["CROSS_DOMAIN_HEADING", "REVIEW_PROMPT_FILENAME", "cross_domain_section", "merge_round_robin",
           "parse_review", "plan_design_point", "plan_queries", "render_review", "review_plan", "same_field_domains"]
