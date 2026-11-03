"""The task brief: what the task states, compressed once, quoted from the source.

Every step the coach needs to know what the task is and what it fixes. Sending the
whole statement every step costs money and most of it is noise for experience
reuse (evaluation-script source, licence headers, directory layout). Letting the
gate paraphrase it every step was worse: the paraphrase drifted, and the agent's
own choices crept into it as if the user had imposed them.

So the statement is read once, at the first step, and reduced to a short list of
facts, each carrying the exact sentence it came from. Two properties make this safe:

* A fact whose quotation cannot be found in the statement is dropped. The model
  can omit a fact, but it cannot add one that the statement does not contain.
* The brief is the only place a "rule of the task" may come from. Gate and writer
  receive this text, never a paraphrase of it.

The reduction is one auxiliary-model call. When it fails, a mechanical fallback
keeps the sentences that look like rules, so a failed call degrades the brief
instead of removing it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...utils import format_message_content

logger = logging.getLogger(__name__)

# Budget for the model-written brief. A fact is one line; this is a ceiling, not a goal.
MAX_FACTS = 24
MAX_BRIEF_CHARS = 3500
# Statements longer than this are cut before summarising. The AIRS statement is
# about 13k characters including a licence header and the scorer's source.
MAX_STATEMENT_CHARS = 30000
MIN_QUOTE_CHARS = 12

FACT_KINDS = ("goal", "data", "metric", "submission", "resource", "restriction", "time")

BRIEF_PROMPT = """You prepare a task brief for a research agent's experience coach.

The coach needs to know what the task is and what it fixes, and nothing else. Read
the task statement and list its facts.

Rules:
- Every fact MUST carry `quote`: an exact, contiguous sentence or phrase copied from
  the statement, character for character. A fact without a quote is discarded.
- Keep only facts that a researcher choosing a method would need: the goal, the data
  and its fields, how the result is scored, what must be submitted, the compute and
  model restrictions, the time budget.
- Leave out: licence text, scorer or helper source code, directory layout,
  formatting instructions, anything about how to write code.
- State a restriction only if the statement states it. Do not infer one.
- `kind` is one of: goal, data, metric, submission, resource, restriction, time.
- At most {max_facts} facts. One line each, plain words.

Return only JSON:
{{"facts": [{{"kind": "metric", "fact": "scored by accuracy; a has_answer=false row is correct only if no_answer is true", "quote": "..."}}]}}

Task statement:
```
{statement}
```
"""


@dataclass(frozen=True)
class Fact:
    kind: str
    text: str
    quote: str


@dataclass(frozen=True)
class TaskBrief:
    facts: tuple[Fact, ...]
    source: str  # "model" or "fallback"
    dropped: int  # facts discarded because the quotation was not in the statement

    def render(self) -> str:
        """Text for the gate and the writer: facts only, grouped, no quotations."""
        if not self.facts:
            return "[no task brief available]"
        lines = [f"- ({fact.kind}) {fact.text}" for fact in self.facts]
        return _cap("\n".join(lines), MAX_BRIEF_CHARS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "dropped": self.dropped,
            "facts": [
                {"kind": f.kind, "fact": f.text, "quote": f.quote} for f in self.facts
            ],
        }


def _cap(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rsplit("\n", 1)[0]


def _squash(text: str) -> str:
    """Whitespace- and case-insensitive form used to check a quotation."""
    return " ".join(text.replace("`", "").split()).casefold()


def quote_in_statement(quote: str, statement: str) -> bool:
    """Whether ``quote`` appears in ``statement`` (ignoring case and whitespace)."""
    needle = _squash(quote)
    return len(needle) >= MIN_QUOTE_CHARS and needle in _squash(statement)


def _parse(raw: str) -> list[dict[str, Any]]:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].removesuffix("```").strip()
        candidate = candidate.removeprefix("json").strip()
    start = candidate.find("{")
    if start < 0:
        return []
    try:
        payload, _ = json.JSONDecoder().raw_decode(candidate[start:])
    except json.JSONDecodeError:
        return []
    facts = payload.get("facts") if isinstance(payload, dict) else None
    return [f for f in facts if isinstance(f, dict)] if isinstance(facts, list) else []


def verify_facts(
    raw_facts: list[dict[str, Any]], statement: str
) -> tuple[list[Fact], int]:
    """Keep the facts whose quotation really is in the statement."""
    kept: list[Fact] = []
    dropped = 0
    seen: set[str] = set()
    for item in raw_facts:
        text = " ".join(str(item.get("fact") or "").split())
        quote = str(item.get("quote") or "").strip()
        kind = str(item.get("kind") or "").strip().lower()
        if kind not in FACT_KINDS:
            kind = "goal" if not kept else "data"
        if not text or not quote_in_statement(quote, statement):
            dropped += 1
            continue
        if text.casefold() in seen:
            continue
        seen.add(text.casefold())
        kept.append(Fact(kind=kind, text=text[:300], quote=quote[:400]))
        if len(kept) >= MAX_FACTS:
            break
    return kept, dropped


# Sentences that look like rules, for the no-model fallback.
_RULEISH = re.compile(
    r"\b(must|only|not allowed|forbidden|prohibited|do not|don't|never|limit|budget|"
    r"hours?|minutes?|gpu|accuracy|metric|score|submission|evaluate|evaluation|"
    r"test set|train|validation|dataset|model)\b",
    re.IGNORECASE,
)
_NOISE = re.compile(
    r"(copyright|licen[sc]e|all rights reserved|^\s*import |^\s*from \w+ import |"
    r"^\s*def |^\s*class |^\s*#!|^\s*\"\"\")",
    re.IGNORECASE,
)


def fallback_facts(statement: str) -> list[Fact]:
    """Mechanical brief: rule-looking sentences, source code and licence lines removed."""
    facts: list[Fact] = []
    for raw in re.split(r"(?<=[.!?])\s+|\n+", statement):
        line = " ".join(raw.split())
        if len(line) < 25 or len(line) > 300 or _NOISE.search(line):
            continue
        if _RULEISH.search(line):
            facts.append(Fact(kind="goal", text=line, quote=line))
        if len(facts) >= MAX_FACTS:
            break
    return facts


async def build_brief(statement: str, model: Any) -> TaskBrief:
    """Reduce the task statement to verified facts (one auxiliary call)."""
    statement = (statement or "").strip()
    if not statement:
        return TaskBrief(facts=(), source="fallback", dropped=0)
    clipped = statement[:MAX_STATEMENT_CHARS]
    try:
        response = await model.ainvoke(
            [
                SystemMessage(
                    content=BRIEF_PROMPT.format(max_facts=MAX_FACTS, statement=clipped)
                ),
                HumanMessage(content="Write the brief now."),
            ]
        )
        raw_facts = _parse(format_message_content(response).strip())
    except Exception:
        logger.debug("Task brief model call failed", exc_info=True)
        raw_facts = []
    facts, dropped = verify_facts(raw_facts, clipped)
    if facts:
        return TaskBrief(facts=tuple(facts), source="model", dropped=dropped)
    return TaskBrief(
        facts=tuple(fallback_facts(clipped)), source="fallback", dropped=dropped
    )


__all__ = [
    "FACT_KINDS",
    "Fact",
    "TaskBrief",
    "build_brief",
    "fallback_facts",
    "quote_in_statement",
    "verify_facts",
]
