"""The push-side gate: decide whether experience is wanted, and phrase the query.

`apply_experience` is a *pull*: the agent has to remember the tool exists, judge
its own need for experience, and hand-write the query. Two things go wrong. The
query is capped by whatever sentence the agent happened to write, which is
usually the vocabulary of its own situation rather than the vocabulary a paper
would use; and the tool is absent from the always-include list, so above the
tool-selector threshold it can be filtered out of the request entirely and the
reuse layer silently stops running.

This module is the other direction. One auxiliary-model call answers both halves
of the question the agent would otherwise have to ask itself -- *is experience
worth retrieving here* and *what should be retrieved* -- and returns the three
retrieval facets plus the task and state strings the synthesis stage wants. The
two halves are merged into one call because they are one judgement: naming the
decision is what tells you whether there is a decision.

Merging them costs a model call on every step where the answer is "no", so the
caller is expected to short-circuit in code first (empty library, mechanical
step, facets unchanged since the last intervention). What remains here is the
judgement that genuinely needs a model.

`need=false` is a first-class answer, mirroring the `verdict: decline` doctrine
one stage down: the honest "stored experience does not bear on this step" is more
useful than a manufactured relevance.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...utils import format_message_content
from .trace import emit_trace_async

logger = logging.getLogger(__name__)

# How much of the trajectory tail the gate reads. The gate is not being asked to
# reconstruct the run -- it needs the original request plus enough of the recent
# steps to name the decision immediately ahead. Paying for the whole trajectory
# on the auxiliary model buys context the gate does not use.
DEFAULT_RECENT_MESSAGES = 6
MAX_MESSAGE_CHARS = 700
MAX_REQUEST_CHARS = 2000


class GateOutputError(Exception):
    """The gate model's output could not be read as a decision."""


def _clean(text: str, limit: int) -> str:
    """Flatten to one paragraph and cap length."""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + " ...[truncated]"


def render_recent(messages: list[Any], *, limit: int = DEFAULT_RECENT_MESSAGES) -> str:
    """Render the trajectory tail as a compact transcript for the gate.

    Tool calls are rendered by name and arguments rather than dropped: "the agent
    just called `search_paper_text` three times" is often the whole signal for
    whether a decision is being approached or already resolved.
    """
    rendered: list[str] = []
    for message in messages[-limit:]:
        role = getattr(message, "type", "") or type(message).__name__
        body = _clean(format_message_content(message), MAX_MESSAGE_CHARS)
        if not body:
            continue
        rendered.append(f"[{role}] {body}")
    return "\n".join(rendered)


def render_library(stats: Mapping[str, Any] | None) -> str:
    """Render library statistics so the gate knows what vocabulary exists.

    Counts and facet values only, never a record listing: the gate runs on every
    step that reaches it, so this block has to stay a fixed size as the library
    grows. Its purpose is to stop the gate from asking for subject matter the
    library provably does not hold.
    """
    if not stats:
        return "[library statistics unavailable]"
    disciplines = ", ".join(
        f"{name}({count})" for name, count in (stats.get("disciplines") or [])[:6]
    )
    domains = ", ".join(
        f"{name}({count})" for name, count in (stats.get("top_domains") or [])[:8]
    )
    lines = [
        f"records: {stats.get('records', 0)} from {stats.get('papers', 0)} papers",
    ]
    if disciplines:
        lines.append(f"disciplines: {disciplines}")
    if domains:
        lines.append(f"domains: {domains}")
    return "\n".join(lines)


def parse_gate_json(raw_output: str) -> dict[str, Any]:
    """Parse the gate's decision, tolerating fenced JSON and trailing prose.

    Raises ``GateOutputError`` rather than guessing. An unreadable gate output
    must not be read as ``need=true`` (which would spend two more model calls on
    a query nobody wrote) nor silently as ``need=false``; the caller decides, and
    records the reason.
    """
    candidate = raw_output.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1] if "\n" in candidate else candidate
        candidate = candidate.removesuffix("```").strip()
        candidate = candidate.removeprefix("json").strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as first_error:
        start = candidate.find("{")
        if start < 0:
            raise GateOutputError("gate did not return a JSON object") from first_error
        try:
            payload, _ = json.JSONDecoder().raw_decode(candidate[start:])
        except json.JSONDecodeError as exc:
            raise GateOutputError(f"gate returned malformed JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise GateOutputError("gate output must be a JSON object")
    return normalize_gate(payload)


def normalize_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce a parsed gate payload into the fixed decision shape.

    ``need`` is read strictly: only a real boolean ``true`` or the string
    ``"true"`` opens the gate. Anything else -- a missing key, a null, the string
    ``"maybe"`` -- closes it, because the failure that matters is a malformed
    output being read as permission to spend two more calls.

    A ``need=true`` with no usable ``topic`` is downgraded to ``need=false``:
    lexical retrieval with an empty topic returns the library in directory order,
    which is not a recall path, it is noise with a confident shape.
    """
    raw_need = payload.get("need")
    need = raw_need is True or (
        isinstance(raw_need, str) and raw_need.strip().lower() == "true"
    )
    topic = _clean(payload.get("topic") or "", 400)
    method = _clean(payload.get("method") or "", 400)
    task = _clean(payload.get("task") or "", 800)
    state = _clean(payload.get("state") or "", 1200)
    reason = _clean(payload.get("reason") or "", 400)
    if need and not topic:
        need = False
        reason = (
            f"{reason} [downgraded: gate asked for retrieval without a topic]"
            if reason
            else "downgraded: gate asked for retrieval without a topic"
        )
    return {
        "need": need,
        "reason": reason,
        "topic": topic,
        "method": method,
        # The writer reads `task` verbatim and it is part of the policy cache
        # key, so an empty one would key every policy on the same string. Fall
        # back to the topic, which is at least the subject being decided.
        "task": task or topic,
        "state": state,
    }


async def decide_experience_need(
    *,
    request_text: str,
    recent_text: str,
    library_text: str,
    model: Any,
    prompt: str,
    memory_dir: str | Path | None = None,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Run the merged gate call and return the normalized decision."""
    filled = prompt.format(
        request=_clean(request_text, MAX_REQUEST_CHARS) or "[no request text]",
        recent=recent_text or "[no prior steps]",
        library=library_text,
    )
    response = await model.ainvoke(
        [SystemMessage(content=filled), HumanMessage(content="Decide now.")]
    )
    output = format_message_content(response).strip()
    try:
        decision = parse_gate_json(output)
    except GateOutputError as error:
        # Closed on an unreadable output: the expensive path must never open on
        # a query the gate did not actually write.
        decision = {
            "need": False,
            "reason": f"gate output unreadable: {error}",
            "topic": "",
            "method": "",
            "task": "",
            "state": "",
        }
    if memory_dir is not None:
        await emit_trace_async(
            memory_dir,
            "gate",
            call_id=call_id,
            raw_output=output,
            **decision,
        )
    return decision


__all__ = [
    "DEFAULT_RECENT_MESSAGES",
    "GateOutputError",
    "decide_experience_need",
    "normalize_gate",
    "parse_gate_json",
    "render_library",
    "render_recent",
]
