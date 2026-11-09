"""Paper passages as text an agent reads, not as JSON.

`search_paper_text` returns locators (which paper, which section, a snippet of why it matched); `read_paper` returns the
passage itself. Both used to be JSON documents, which turned every line break of the paper into a `\\n` escape. Here the
passage is plain text under a one-line header that says where it came from and whether it was cut.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _s(value: Any) -> str:
    return " ".join(str(value if value is not None else "").split())


def _clip(text: str, limit: int) -> str:
    text = _s(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _label(paper: Mapping[str, Any] | None) -> str:
    paper = paper if isinstance(paper, Mapping) else {}
    title, pid = _clip(paper.get("title") or "", 100), _s(paper.get("paper_id"))
    return f"{title} ({pid})" if title and pid else title or pid or "unknown paper"


def render_passage_hits(results: Sequence[Mapping[str, Any]], *, papers_stored: int | None = None, hint: str = "") -> str:
    """Ranked passage locators from `search_paper_text`."""
    if not results:
        out = ["No passage matched."]
        if papers_stored is not None:
            out.append(f"Papers with stored full text in this project: {papers_stored}.")
        if hint:
            out.append(hint)
        return "\n".join(out)
    lines = [f"{len(results)} passage(s), best match first:", ""]
    for i, r in enumerate(results, 1):
        where = _s(r.get("section_path"))
        lines.append(f"{i}. {_label(r)}" + (f" · section: {where}" if where else ""))
        lines.append(f"   chunk_id: {_s(r.get('chunk_id'))}")
        for snippet in list(r.get("matches") or [])[:2]:
            text = snippet.get("text") if isinstance(snippet, Mapping) else snippet
            if _s(text):
                lines.append(f"   … {_clip(text, 240)}")
        if _s(r.get("url")):
            lines.append(f"   {_s(r.get('url'))}")
    lines += ["", "Read a passage with read_paper(chunk_id=...); expand=section gives its whole section, expand=full needs paper_id."]
    return "\n".join(lines)


def render_passage(result: Mapping[str, Any]) -> str:
    """One passage or a whole paper from `read_paper`: a header line, then the text as written."""
    paper = result.get("paper") if isinstance(result.get("paper"), Mapping) else {}
    if result.get("expand") == "full":
        scope = "full text"
        span = f"{result.get('returned_chars')} of {result.get('char_count')} characters"
    else:
        scope = f"section: {_s(result.get('section_path'))}" if _s(result.get("section_path")) else "passage"
        span = f"characters {result.get('char_start')}–{result.get('char_end')}"
    header = f"{_label(paper)} · {scope} · {span}" + (" · TRUNCATED" if result.get("truncated") else "")
    url = _s(paper.get("url")) if isinstance(paper, Mapping) else ""
    return "\n".join(p for p in (header, url, "", str(result.get("text") or "")) if p is not None)


def render_error(message: str) -> str:
    return f"Error: {_s(message)}"


__all__ = ["render_error", "render_passage", "render_passage_hits"]
