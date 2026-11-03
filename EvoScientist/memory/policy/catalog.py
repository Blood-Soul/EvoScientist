"""What the gate may see of the library, whatever its size.

The old gate saw one line of statistics ("74 records from 6 papers") and had to
guess the library's vocabulary. Retrieval is lexical, so a guess in the wrong
words finds nothing, and a guess in the right words is a coincidence. The gate
should choose from what exists, not invent a query.

Showing the whole library does not scale: one line per record is 77 kB at ten
thousand records. So the gate sees two small, fixed-size things, neither of which
grows with the library:

* a map: the library's topics and how many records sit under each, so the gate
  knows what kind of knowledge exists at all;
* candidates: the records most related to the agent's current situation, one line
  each, found by the same lexical search that already backs `search_experience`.
  The query is built in code from the task brief and the agent's recent actions, so
  the candidates do not depend on the gate having phrased anything correctly.

The gate then names the records it wants. Everything here is deterministic and
costs milliseconds.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..experiences.retrieval import search_experience_records
from ..experiences.store import list_experience_documents
from ..types import ObservationSearchMode

MAP_TOPICS = 24
DEFAULT_CANDIDATES = 24
LINE_CHARS = 170

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{3,}")
# Words that appear in nearly every research step and carry no topic.
_BLAND = frozenset(
    """this that with from have will your their there which what when where into
    file files code script python import print true false none return used using
    result results step steps model models train training data test tests run runs
    output outputs error errors path paths line lines check checking value values
    need needs make makes made also then than more most some each other over under""".split()
)


def normalise_topic(value: str | None) -> str:
    """Fold the spellings of one domain into one key.

    The library's `domain` field is free text and one idea arrives in several
    spellings ("machine_reading_comprehension", "machine reading comprehension",
    "reading_comprehension"). Folding case, separators and plurals keeps the map
    short without any model.
    """
    text = (value or "").casefold().replace("_", " ").replace("-", " ")
    words = [w.rstrip("s") if len(w) > 4 else w for w in text.split()]
    return " ".join(words) or "unknown"


def library_map(documents: list[Any], *, limit: int = MAP_TOPICS) -> str:
    """One line per topic, largest first, with a few example tasks."""
    topics: dict[str, Counter[str]] = {}
    counts: Counter[str] = Counter()
    for document in documents:
        key = normalise_topic(getattr(document, "domain", ""))
        counts[key] += 1
        task = " ".join((getattr(document, "task", "") or "").split())
        if task:
            topics.setdefault(key, Counter())[task] += 1
    lines = []
    for key, count in counts.most_common(limit):
        examples = [t for t, _ in topics.get(key, Counter()).most_common(2)]
        shown = "; ".join(e[:70] for e in examples)
        lines.append(f"- {key} ({count})" + (f": {shown}" if shown else ""))
    rest = len(counts) - limit
    if rest > 0:
        lines.append(f"- ... and {rest} more topics")
    return "\n".join(lines) or "[the library is empty]"


def situation_terms(*texts: str, limit: int = 40) -> str:
    """Query terms for candidate search, taken from text rather than invented.

    The most frequent substantive words across the brief and recent actions; no
    model decides them, so the candidates cannot depend on the gate's phrasing.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        for word in _WORD.findall(text or ""):
            lowered = word.casefold().strip("_-")
            if lowered not in _BLAND:
                counts[lowered] += 1
    return " ".join(word for word, _ in counts.most_common(limit))


def record_line(document: Any) -> str:
    """One line: id, level, task, and the record's own one-line summary."""
    task = " ".join((getattr(document, "task", "") or "").split())
    summary = " ".join((getattr(document, "summary", "") or "").split())
    head = f"{document.observation_id} [{(document.experience_level or '?').upper()}]"
    body = task or summary
    if task and summary and summary.casefold() not in task.casefold():
        body = f"{task} - {summary}"
    line = f"{head} {body}"
    return line if len(line) <= LINE_CHARS else line[: LINE_CHARS - 3] + "..."


def candidate_lines(
    *,
    memory_dir: str | Path,
    project_id: str,
    query: str,
    limit: int = DEFAULT_CANDIDATES,
) -> tuple[list[str], list[str]]:
    """Records most related to ``query``: (their ids, one display line each)."""
    if not query.strip():
        return [], []
    hits = search_experience_records(
        memory_dir=memory_dir,
        project_id=project_id,
        topic=query,
        limit=limit,
        mode=ObservationSearchMode.RANKED,
    )
    ids = [hit["observation_id"] for hit in hits][:limit]
    if not ids:
        return [], []
    by_id = {
        document.observation_id: document
        for document in list_experience_documents(
            memory_dir=memory_dir, project_id=project_id
        )
    }
    ordered = [by_id[i] for i in ids if i in by_id]
    return [d.observation_id for d in ordered], [record_line(d) for d in ordered]


def render_catalog(
    *,
    memory_dir: str | Path,
    project_id: str,
    query: str,
    candidates: int = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    """Everything the gate may see of the library, plus the ids it may name."""
    documents = list_experience_documents(memory_dir=memory_dir, project_id=project_id)
    ids, lines = candidate_lines(
        memory_dir=memory_dir, project_id=project_id, query=query, limit=candidates
    )
    return {
        "records": len(documents),
        "map": library_map(documents),
        "candidate_ids": ids,
        "candidates": "\n".join(lines) if lines else "[nothing in the library matched]",
    }


__all__ = [
    "DEFAULT_CANDIDATES",
    "candidate_lines",
    "library_map",
    "normalise_topic",
    "record_line",
    "render_catalog",
    "situation_terms",
]
