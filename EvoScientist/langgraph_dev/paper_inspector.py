"""A developer-facing inspector for the paper full-text (RAG) chunk store.

Purpose: confirm from a browser that ingestion actually happened -- which
projects hold papers, how each paper was chunked, and what text a given chunk
carries. This is a debug surface, not a product feature: the markup is
deliberately plain, there is no styling, and nothing here is meant for end
users.

Why server-rendered HTML on the langgraph dev app: the WebUI front end lives in
a separate package (``@evoscientist/webui``), so adding a panel there would mean
pulling that repo in. The dev server already mounts a custom Starlette app via
``langgraph.json``'s ``http`` key, so one more route reaches the same origin the
browser is already talking to at zero front-end cost.

Every handler offloads filesystem work with ``asyncio.to_thread``: langgraph
dev's ``blockbuster`` middleware rejects blocking syscalls on the event loop and
would turn a directory walk into a 500.
"""

from __future__ import annotations

import asyncio
import html
from pathlib import Path
from typing import Any
from urllib.parse import quote

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

# A project can hold dozens of papers at tens of chunks each. The index stays
# paper-level and chunk rows are one paper at a time, so no single response has
# to carry a whole store.
_MAX_CHUNK_ROWS = 400


def _memories_dir() -> Path:
    from EvoScientist.paths import MEMORIES_DIR

    return Path(MEMORIES_DIR)


def _fulltext_enabled() -> bool:
    """Report the switch state without gating the inspector on it.

    Turning the feature off stops new ingestion; it does not delete what is
    already stored, and "did anything land before I turned it off" is exactly
    the question this page answers. So the state is displayed, not enforced.
    """
    try:
        from EvoScientist.config import get_effective_config

        return bool(get_effective_config().memory_paper_fulltext_enabled)
    except Exception:
        return True


def _collect_overview() -> dict[str, Any]:
    """Gather paper-level metadata for every project with stored text."""
    from EvoScientist.memory.papers.store import list_paper_projects, list_papers

    memories = _memories_dir()
    projects = []
    for project_id in list_paper_projects(memories):
        papers = list_papers(memory_dir=memories, project_id=project_id)
        projects.append(
            {
                "project_id": project_id,
                "paper_count": len(papers),
                "chunk_total": sum(int(p.get("chunk_count") or 0) for p in papers),
                "papers": [
                    {
                        "paper_id": paper.get("paper_id"),
                        "paper_key": paper.get("paper_key"),
                        "title": paper.get("title"),
                        "url": paper.get("url"),
                        "char_count": paper.get("char_count"),
                        "chunk_count": paper.get("chunk_count"),
                        "section_count": paper.get("section_count"),
                        "max_chunk_chars": paper.get("max_chunk_chars"),
                        "overlap_chars": paper.get("overlap_chars"),
                        "chunking_version": paper.get("chunking_version"),
                        "paper_sha256": paper.get("paper_sha256"),
                    }
                    for paper in papers
                ],
            }
        )
    return {
        "memories_dir": str(memories),
        "fulltext_enabled": _fulltext_enabled(),
        "projects": projects,
    }


def _collect_paper(project_id: str, paper_id: str) -> dict[str, Any] | None:
    """Gather one paper's metadata and chunk rows, text included."""
    from EvoScientist.memory.papers.store import list_paper_chunks

    found = list_paper_chunks(
        memory_dir=_memories_dir(), project_id=project_id, paper_id=paper_id
    )
    if found is None:
        return None
    metadata, chunks = found
    return {
        "project_id": project_id,
        "metadata": metadata,
        "chunk_count": len(chunks),
        "truncated": len(chunks) > _MAX_CHUNK_ROWS,
        "chunks": [
            {
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index"),
                "section": chunk.get("section"),
                "section_path": chunk.get("section_path"),
                "char_start": chunk.get("char_start"),
                "char_end": chunk.get("char_end"),
                "text_chars": len(str(chunk.get("text") or "")),
                "text": str(chunk.get("text") or ""),
            }
            for chunk in chunks[:_MAX_CHUNK_ROWS]
        ],
    }


def _esc(value: Any) -> str:
    """Escape any value for HTML text content.

    Everything rendered here -- titles, URLs, section headings, chunk text --
    comes from a downloaded paper, so it is untrusted input that happens to be
    stored locally. It is escaped, never interpolated raw.
    """
    return html.escape("" if value is None else str(value), quote=True)


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html>\n<html><head><meta charset='utf-8'>"
        f"<title>{_esc(title)}</title>"
        # The only styling: monospace and wrapped preformatted text, so long
        # chunks stay readable without horizontal scrolling.
        "<style>body{font-family:monospace;margin:1rem;line-height:1.4}"
        "pre{white-space:pre-wrap;background:#f4f4f4;padding:.5rem;"
        "border-left:3px solid #999;overflow-wrap:anywhere}"
        "td,th{border:1px solid #ccc;padding:2px 6px;text-align:left;"
        "vertical-align:top}table{border-collapse:collapse}"
        "summary{cursor:pointer}</style></head><body>"
        f"{body}</body></html>"
    )


def _render_overview(data: dict[str, Any]) -> str:
    switch = "enabled" if data["fulltext_enabled"] else "DISABLED (no new ingestion)"
    parts = [
        "<h2>Paper full-text (RAG) chunk store</h2>",
        f"<p>memories dir: <code>{_esc(data['memories_dir'])}</code><br>",
        f"memory_paper_fulltext_enabled: <b>{_esc(switch)}</b><br>",
        "<a href='/debug/papers.json'>raw JSON</a></p>",
    ]
    if not data["projects"]:
        parts.append(
            "<p><b>No paper full text stored yet.</b> Text is persisted when "
            "paper experience extraction runs (background queue or the "
            "foreground extract tool), so run one of those first.</p>"
        )
        return "".join(parts)

    for project in data["projects"]:
        parts.append(
            f"<h3>project {_esc(project['project_id'])} &mdash; "
            f"{project['paper_count']} papers, "
            f"{project['chunk_total']} chunks</h3>"
        )
        parts.append(
            "<table><tr><th>paper_id</th><th>title</th><th>chars</th>"
            "<th>chunks</th><th>sections</th><th>chunk/overlap</th>"
            "<th>v</th></tr>"
        )
        for paper in project["papers"]:
            # paper_key is the URL-safe identifier; paper_id may hold a raw URL.
            link = (
                f"/debug/papers?project={quote(str(project['project_id']))}"
                f"&amp;paper={quote(str(paper['paper_key'] or ''))}"
            )
            parts.append(
                "<tr>"
                f'<td><a href="{link}">{_esc(paper["paper_id"])}</a></td>'
                f"<td>{_esc(paper['title'])}</td>"
                f"<td>{_esc(paper['char_count'])}</td>"
                f"<td>{_esc(paper['chunk_count'])}</td>"
                f"<td>{_esc(paper['section_count'])}</td>"
                f"<td>{_esc(paper['max_chunk_chars'])}"
                f"/{_esc(paper['overlap_chars'])}</td>"
                f"<td>{_esc(paper['chunking_version'])}</td>"
                "</tr>"
            )
        parts.append("</table>")
    return "".join(parts)


def _render_paper(data: dict[str, Any]) -> str:
    metadata = data["metadata"]
    parts = [
        "<p><a href='/debug/papers'>&larr; all papers</a></p>",
        f"<h2>{_esc(metadata.get('title'))}</h2>",
        f"<p>paper_id: <code>{_esc(metadata.get('paper_id'))}</code><br>"
        f"paper_key: <code>{_esc(metadata.get('paper_key'))}</code><br>"
        f"url: {_esc(metadata.get('url'))}<br>"
        f"project: <code>{_esc(data['project_id'])}</code><br>"
        f"chars: {_esc(metadata.get('char_count'))}, "
        f"chunks: {_esc(metadata.get('chunk_count'))}, "
        f"sections: {_esc(metadata.get('section_count'))}<br>"
        f"sha256: <code>{_esc(metadata.get('paper_sha256'))}</code></p>",
    ]
    if data["truncated"]:
        parts.append(
            f"<p><b>Showing the first {_MAX_CHUNK_ROWS} chunks of "
            f"{_esc(metadata.get('chunk_count'))}.</b></p>"
        )
    for chunk in data["chunks"]:
        parts.append(
            "<details><summary>"
            f"#{_esc(chunk['chunk_index'])} "
            f"<code>{_esc(chunk['chunk_id'])}</code> "
            f"[{_esc(chunk['char_start'])}:{_esc(chunk['char_end'])}] "
            f"{_esc(chunk['text_chars'])} chars &mdash; "
            f"{_esc(chunk['section_path'] or '(no section)')}"
            "</summary>"
            f"<pre>{_esc(chunk['text'])}</pre></details>"
        )
    return "".join(parts)


async def get_paper_store_page(request: Request) -> HTMLResponse:
    """Render the chunk-store inspector, or one paper's chunk list."""
    project_id = (request.query_params.get("project") or "").strip()
    paper_id = (request.query_params.get("paper") or "").strip()
    if project_id and paper_id:
        data = await asyncio.to_thread(_collect_paper, project_id, paper_id)
        if data is None:
            return HTMLResponse(
                _page(
                    "paper not found",
                    "<p><a href='/debug/papers'>&larr; all papers</a></p>"
                    "<p><b>No stored paper matched that project and id.</b></p>",
                ),
                status_code=404,
            )
        return HTMLResponse(
            _page(f"chunks: {data['metadata'].get('title')}", _render_paper(data))
        )
    overview = await asyncio.to_thread(_collect_overview)
    return HTMLResponse(_page("paper chunk store", _render_overview(overview)))


async def get_paper_store_json(request: Request) -> JSONResponse:
    """Same data as the page, for scripted checks.

    Chunk text is omitted from the paper view here; the page is the place to
    read text, and a JSON dump of every chunk body would be megabytes.
    """
    project_id = (request.query_params.get("project") or "").strip()
    paper_id = (request.query_params.get("paper") or "").strip()
    if project_id and paper_id:
        data = await asyncio.to_thread(_collect_paper, project_id, paper_id)
        if data is None:
            return JSONResponse({"error": "paper not found"}, status_code=404)
        for chunk in data["chunks"]:
            chunk.pop("text", None)
        return JSONResponse(data)
    return JSONResponse(await asyncio.to_thread(_collect_overview))


__all__ = [
    "get_paper_store_json",
    "get_paper_store_page",
]
