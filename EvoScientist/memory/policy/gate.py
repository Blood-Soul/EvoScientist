"""The gate: look at the situation and the library, then say what to consult.

Role. The gate answers two questions in one call: does the decision the agent is
about to take have stored experience that bears on it, and if so, which records. It
is the only stage that knows the agent's situation, so it is the stage that has to
name what is wanted.

What it sees (and why):
* the task brief: facts the task states, each verified against the statement. The
  one place a rule of the task can come from;
* the agent's recent actions and their results: what happened, as recorded, with the
  coach's own earlier guidance removed so the gate cannot read its own output back;
* the agent's recent narration, labelled as the agent's words: it shows where the
  agent is heading, but it is never a source of task rules;
* the library: a map of topics and a short list of candidate records, so the gate
  chooses from what exists instead of guessing the library's vocabulary.

What it returns:
* ``need``;
* ``situation``: what the agent is doing right now, in neutral words;
* ``pick``: the ids of the records it wants (from the candidate list only);
* ``purpose``: what it wants to decide with them.

It does not return topic, method, or state. Topic and method were guesses at the
library's vocabulary; state was a paraphrase of the agent's work that the writer
then read as fact. Candidates replace the former, the writer's own inputs the
latter.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...utils import format_message_content
from .trace import emit_trace_async

logger = logging.getLogger(__name__)

DEFAULT_RECENT_MESSAGES = 6
# Action history shown to the gate: how many recent steps, and how much of each.
DEFAULT_RECENT_ACTIONS = 10
MAX_ACTION_CHARS = 260
MAX_RESULT_CHARS = 260
MAX_NARRATION_CHARS = 500
MAX_PICKED = 8

_GUIDANCE_BLOCK = re.compile(
    r"<experience_guidance>.*?</experience_guidance>", re.DOTALL | re.IGNORECASE
)
# The agent habitually opens a reply by acknowledging the guidance it just got and
# then restating it ("Guidance received. It asks for a balanced objective ...").
# Dropping only the acknowledgement would leave the restatement, which is the
# coach's own advice coming back as if it were the agent's situation. In the DuoRC
# runs 56% of the agent's text blocks opened this way. So a block that opens by
# acknowledging guidance is dropped whole; the actions and results still show what
# the agent then did.
_ECHO_LEAD = re.compile(
    r"^\s*(?:\*\*)?(?:the\s+|this\s+)?guidance(?:\W|'s|\b)",
    re.IGNORECASE,
)


class GateOutputError(Exception):
    """The gate model's output could not be read as a decision."""


def _clean(text: str, limit: int) -> str:
    """Flatten to one paragraph and cap length."""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + " ...[truncated]"


def strip_echo(text: str) -> str:
    """Remove the coach's own guidance and the agent's restatement of it."""
    text = _GUIDANCE_BLOCK.sub(" ", text or "").strip()
    return "" if _ECHO_LEAD.match(text) else text


def _brief_args(args: Any) -> str:
    if isinstance(args, Mapping):
        for key in ("command", "file_path", "path", "pattern", "query", "task", "url"):
            if args.get(key):
                return str(args[key])
        return json.dumps(args, ensure_ascii=False, default=str)
    return str(args or "")


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return " ".join(
        str(part.get("text", ""))
        for part in content or []
        if isinstance(part, Mapping) and part.get("type") == "text"
    )


def render_activity(
    messages: Sequence[Any], *, actions: int = DEFAULT_RECENT_ACTIONS
) -> str:
    """The agent's recent history as the gate reads it.

    Each step is the agent's narration (labelled as such, with the coach's echo
    removed), then what it did, then what came back. Narration is kept because
    without it the gate cannot tell where the agent is in its work; it is labelled
    because it is the agent's own words and can say anything.
    """
    results: dict[str, str] = {}
    for message in messages:
        if getattr(message, "type", "") == "tool":
            results[str(getattr(message, "tool_call_id", ""))] = _clean(
                format_message_content(message), MAX_RESULT_CHARS
            )
    steps: list[str] = []
    for message in messages:
        if getattr(message, "type", "") != "ai":
            continue
        narration = _clean(strip_echo(_message_text(message)), MAX_NARRATION_CHARS)
        calls = getattr(message, "tool_calls", None) or []
        if not calls and not narration:
            continue
        lines = []
        if narration:
            lines.append(f"  agent says: {narration}")
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            action = _clean(
                f"{call.get('name', '')}: {_brief_args(call.get('args'))}",
                MAX_ACTION_CHARS,
            )
            lines.append(f"  did: {action}")
            outcome = results.get(str(call.get("id", "")))
            if outcome:
                lines.append(f"  result: {outcome}")
        steps.append("\n".join(lines))
    shown = steps[-actions:]
    if not shown:
        return "[no steps yet]"
    return "\n".join(f"[step {i + 1}]\n{block}" for i, block in enumerate(shown))


def render_recent(
    messages: Sequence[Any], *, limit: int = DEFAULT_RECENT_MESSAGES
) -> str:
    """Compact transcript of the last few messages."""
    rendered: list[str] = []
    for message in list(messages)[-limit:]:
        role = getattr(message, "type", "") or type(message).__name__
        body = _clean(format_message_content(message), 700)
        if body:
            rendered.append(f"[{role}] {body}")
    return "\n".join(rendered)


def render_library(stats: Mapping[str, Any] | None) -> str:
    """One-line library statistics, for callers that have no catalog."""
    if not stats:
        return "[library statistics unavailable]"
    return f"records: {stats.get('records', 0)} from {stats.get('papers', 0)} papers"


def parse_gate_json(
    raw_output: str, *, allowed_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    """Parse the gate's decision, tolerating fenced JSON and trailing prose.

    Raises ``GateOutputError`` rather than guessing: an unreadable output must not
    be read as permission to spend two more model calls, nor as a silent no.
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
    return normalize_gate(payload, allowed_ids=allowed_ids)


def normalize_gate(
    payload: Mapping[str, Any], *, allowed_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    """Coerce a parsed payload into the fixed decision shape.

    ``need`` opens only on a real ``true``. Picked ids that are not in the candidate
    list are dropped: the gate chooses from what exists. A ``need=true`` that picks
    nothing usable is downgraded, because there is then nothing to consult.
    """
    raw_need = payload.get("need")
    need = raw_need is True or (
        isinstance(raw_need, str) and raw_need.strip().lower() == "true"
    )
    situation = _clean(payload.get("situation") or "", 500)
    purpose = _clean(payload.get("purpose") or "", 500)
    reason = _clean(payload.get("reason") or "", 400)
    raw_pick = payload.get("pick")
    pick: list[str] = []
    if isinstance(raw_pick, list):
        allowed = set(allowed_ids) if allowed_ids is not None else None
        for item in raw_pick:
            identifier = str(item).strip()
            if (
                identifier
                and identifier not in pick
                and (allowed is None or identifier in allowed)
            ):
                pick.append(identifier)
    pick = pick[:MAX_PICKED]
    if need and not pick:
        need = False
        reason = (
            f"{reason} [downgraded: no usable record picked]"
            if reason
            else "downgraded: no usable record picked"
        )
    return {
        "need": need,
        "reason": reason,
        "situation": situation,
        "pick": pick,
        "purpose": purpose or situation,
    }


MAX_SUMMARY_CHARS = 1800


async def summarize_activity(
    *,
    activity_text: str,
    model: Any,
    prompt: str,
    memory_dir: str | Path | None = None,
    call_id: str | None = None,
) -> str:
    """A short account of the recent steps, in place of the steps themselves.

    The gate and the writer read this instead of the last few raw actions, which
    are mostly commands and truncated output. If the summary call fails or comes
    back empty the raw steps are returned unchanged: a coach with an uglier view
    is better than a coach that stops.
    """
    if not activity_text or activity_text == "[no steps yet]":
        return activity_text
    started = time.monotonic()
    try:
        response = await model.ainvoke(
            [
                SystemMessage(content=prompt.format(activity=activity_text)),
                HumanMessage(content="Summarise now."),
            ]
        )
    except Exception:
        logger.debug("Activity summary failed; using the raw steps", exc_info=True)
        return activity_text
    summary = strip_echo(format_message_content(response).strip())[:MAX_SUMMARY_CHARS]
    if memory_dir is not None:
        await emit_trace_async(
            memory_dir,
            "activity_summary",
            call_id=call_id,
            raw_chars=len(activity_text),
            seconds=round(time.monotonic() - started, 2),
            usage=response_usage(response),
            summary=summary,
        )
    return summary or activity_text


async def decide_experience_need(
    *,
    brief_text: str,
    activity_text: str,
    library_map: str,
    candidates_text: str,
    allowed_ids: Sequence[str],
    model: Any,
    prompt: str,
    memory_dir: str | Path | None = None,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Run the gate call and return the normalised decision."""
    filled = prompt.format(
        brief=brief_text or "[no task brief available]",
        activity=activity_text or "[no steps yet]",
        library_map=library_map,
        candidates=candidates_text,
    )
    response = await model.ainvoke(
        [SystemMessage(content=filled), HumanMessage(content="Decide now.")]
    )
    output = format_message_content(response).strip()
    try:
        decision = parse_gate_json(output, allowed_ids=allowed_ids)
    except GateOutputError as error:
        decision = {
            "need": False,
            "reason": f"gate output unreadable: {error}",
            "situation": "",
            "pick": [],
            "purpose": "",
        }
    if memory_dir is not None:
        await emit_trace_async(
            memory_dir, "gate", call_id=call_id, raw_output=output, **decision
        )
    return decision


__all__ = [
    "DEFAULT_RECENT_ACTIONS",
    "DEFAULT_RECENT_MESSAGES",
    "MAX_PICKED",
    "GateOutputError",
    "decide_experience_need",
    "normalize_gate",
    "parse_gate_json",
    "render_activity",
    "render_library",
    "render_recent",
    "summarize_activity",
    "strip_echo",
]
