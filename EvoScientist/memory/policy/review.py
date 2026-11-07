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


def _render_records(candidates: Sequence[dict[str, Any]]) -> str:
    blocks = []
    for c in candidates:
        text = json.dumps(c["record"], ensure_ascii=False, indent=1, sort_keys=True)
        if len(text) > MAX_RECORD_CHARS:
            text = text[:MAX_RECORD_CHARS] + "\n... [record truncated]"
        blocks.append(f"{c['id']}\n{text}")
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
        kept.append({"about": about, "suggestion": text, "because": because,
                     "sources": list(dict.fromkeys(srcs)), "kind": kind if kind in KINDS else "check"})
    return {"suggestions": kept[:max_suggestions], "dropped_without_source": dropped,
            "no_suggestion_reason": " ".join(str(payload.get("no_suggestion_reason") or "").split())[:300]}


def render_review(review: dict[str, Any], titles: dict[str, str] | None = None) -> str:
    """The review as text for the plan-writer: numbered suggestions, each with the records it rests on."""
    titles = titles or {}
    sugg = review.get("suggestions") or []
    if not sugg:
        why = review.get("no_suggestion_reason") or "no stored experience bears on this plan"
        return f"Plan review: no suggestions ({why})."
    lines = ["Plan review against stored experience. Each item rests on the records named. You decide: take it and "
             "revise the plan, or refuse it and say why in your reply. Check a record with `read_memory` before relying on it."]
    for i, s in enumerate(sugg, 1):
        refs = ", ".join(f"{e}" + (f" ({titles[e]})" if e in titles else "") for e in s["sources"])
        lines.append(f"\n{i}. [{s['kind']}] About: {s['about']}\n   Suggestion: {s['suggestion']}\n   Because: {s['because']}\n   Sources: {refs}")
    return "\n".join(lines)


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
) -> dict[str, Any]:
    """Review ``plan`` against the library. Returns the parsed review plus what was looked at."""
    import asyncio

    started = time.monotonic()
    if not plan.strip():
        return {"status": "empty_plan", "suggestions": [], "text": "Plan review: the plan is empty."}
    queries = plan_queries(plan)
    found: dict[str, dict[str, Any]] = {}
    for q in queries:
        for c in await asyncio.to_thread(
            gather_candidates, memory_dir=memory_dir, project_id=project_id, query=q, limit=record_limit, call_id=call_id
        ):
            found.setdefault(c["id"], c)
    # the plan may already name records; make sure the reviewer sees them
    named = [i for i in _ID.findall(plan) if i not in found]
    if named:
        for c in await asyncio.to_thread(
            fetch_candidates, memory_dir=memory_dir, project_id=project_id, ids=named, call_id=call_id
        ):
            found.setdefault(c["id"], c)
    if not found:
        return {"status": "no_records", "suggestions": [], "queries": queries,
                "text": "Plan review: no stored experience matched this plan."}
    shown = list(found.values())[: max(record_limit * 2, 12)]
    prompt = await asyncio.to_thread(load_policy_prompt, REVIEW_PROMPT_FILENAME)
    filled = prompt.format(task=task.strip()[:20000], plan=plan[:MAX_PLAN_CHARS], records=_render_records(shown),
                           max_suggestions=max_suggestions)
    response = await model.ainvoke([SystemMessage(content=filled), HumanMessage(content="Review the plan now.")])
    raw = format_message_content(response).strip()
    allowed = {c["id"] for c in shown}
    try:
        review = parse_review(raw, allowed, max_suggestions)
    except ValueError as error:
        await emit_trace_async(memory_dir, "plan_review", call_id=call_id, ok=False, error=str(error), raw_output=raw[:2000],
                               seconds=round(time.monotonic() - started, 2))
        return {"status": "unreadable", "suggestions": [], "queries": queries,
                "text": f"Plan review: the reviewer's output could not be read ({error})."}
    titles = {}
    for c in shown:
        paper = c["record"].get("paper") if isinstance(c.get("record"), dict) else None
        if isinstance(paper, dict) and paper.get("title"):
            titles[c["id"]] = " ".join(str(paper["title"]).split())[:70]
    text = render_review(review, titles)
    await emit_trace_async(memory_dir, "plan_review", call_id=call_id, ok=True, queries=queries, shown=[c["id"] for c in shown],
                           suggestions=review["suggestions"], dropped_without_source=review["dropped_without_source"],
                           seconds=round(time.monotonic() - started, 2), usage=response_usage(response))
    return {"status": "ok", "queries": queries, "shown": [c["id"] for c in shown], "text": text, **review}


__all__ = ["REVIEW_PROMPT_FILENAME", "parse_review", "plan_queries", "render_review", "review_plan"]
