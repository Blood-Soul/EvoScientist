"""Agent tools for retrieving stored paper full text.

Two tools, two granularities. ``search_paper_text`` locates passages and
returns only locators plus matched snippets; ``read_paper`` expands one
locator into its section, or a whole paper. Splitting them keeps a broad
search cheap while leaving deep single-paper reading available -- the finding
that motivated this store is that a whole paper can outperform its distilled
experience, so that path must not be closed off.

These are deliberately separate from ``search_observations``: chunks are
numerous and would otherwise crowd ``E-*`` experience records out of a shared
ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field

from ..memory.papers import (
    describe_chunks,
    list_papers,
    read_paper_chunk,
    read_paper_full,
    search_paper_chunks,
)
from ..memory.runtime_context import runtime_project_id
from ..memory.types import ObservationSearchMode

# A whole paper runs to tens of thousands of characters. The default keeps one
# `read_paper` call from dominating the context window while staying large
# enough to hold a typical full paper body; raise it deliberately per call.
DEFAULT_FULL_PAPER_MAX_CHARS = 24000
MAX_FULL_PAPER_MAX_CHARS = 120000
# Section reads are bounded too: a merged, over-long section can otherwise
# return more than the caller expects from a "read one section" request.
DEFAULT_SECTION_MAX_CHARS = 12000


class SearchPaperTextArgs(BaseModel):
    """Model-facing arguments for the `search_paper_text` tool."""

    query: str = Field(
        min_length=1,
        description=(
            "Distinctive words or a short phrase naming the method, metric, "
            "dataset, or claim to locate in stored paper text. Retrieval is "
            "lexical, so prefer the terminology a paper would itself use and "
            "try 1-3 focused phrasing variants when the first returns little."
        ),
    )
    mode: ObservationSearchMode = Field(
        default=ObservationSearchMode.RANKED,
        description=(
            "ranked scores passages by term overlap and returns the best "
            "matches. regex interprets query as a grep-like pattern and falls "
            "back to literal matching when the pattern is invalid."
        ),
    )
    paper_id: str = Field(
        default="",
        description=(
            "Optional paper ID (as listed in the inlined paper full-text index "
            "or returned by a previous hit) to restrict the search to one paper."
        ),
    )
    limit: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum number of matching passages to return.",
    )
    runtime: Annotated[object | None, InjectedToolArg] = None


class ReadPaperArgs(BaseModel):
    """Model-facing arguments for the `read_paper` tool."""

    chunk_id: str = Field(
        default="",
        description=(
            "A C-* passage ID from `search_paper_text`. Required unless "
            "`paper_id` is given with expand=full."
        ),
    )
    paper_id: str = Field(
        default="",
        description=(
            "A paper ID to read in full. Used with expand=full; ignored when "
            "`chunk_id` is supplied."
        ),
    )
    expand: str = Field(
        default="section",
        description=(
            "section returns the whole section containing the passage "
            "(default, best for understanding a result in context); chunk "
            "returns just the matched passage; full returns the entire paper "
            "and requires `paper_id`."
        ),
    )
    max_chars: int = Field(
        default=0,
        ge=0,
        le=MAX_FULL_PAPER_MAX_CHARS,
        description=(
            "Optional cap on returned characters. 0 uses the default for the "
            "chosen granularity. Truncation is reported in the response."
        ),
    )
    runtime: Annotated[object | None, InjectedToolArg] = None


def create_search_paper_text_tool(
    *, memory_dir: str | Path, project_id: str
) -> BaseTool:
    """Build the `search_paper_text` tool for one project context."""

    def _search_paper_text(
        query: str,
        mode: ObservationSearchMode = ObservationSearchMode.RANKED,
        paper_id: str = "",
        limit: int = 8,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> str:
        active_project = runtime_project_id(runtime, project_id)
        hits = search_paper_chunks(
            memory_dir=memory_dir,
            project_id=active_project,
            query=query,
            limit=limit,
            mode=ObservationSearchMode(mode),
            paper_id=paper_id or None,
        )
        # Resolve each hit's paper and section once so the agent can cite a
        # source without a second call, then hand back locators only -- the
        # passage text itself comes from read_paper.
        described = describe_chunks(
            memory_dir=memory_dir,
            project_id=active_project,
            chunk_ids=[hit["observation_id"] for hit in hits],
        )
        results: list[dict[str, Any]] = []
        for hit in hits:
            chunk_id = hit["observation_id"]
            detail = described.get(chunk_id, {})
            paper = detail.get("paper", {})
            results.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": paper.get("paper_id"),
                    "title": paper.get("title"),
                    "url": paper.get("url"),
                    "section_path": detail.get("section_path") or "",
                    "matches": hit.get("matches", []),
                    "score": hit.get("score"),
                }
            )
        if not results:
            stored = list_papers(memory_dir=memory_dir, project_id=active_project)
            return json.dumps(
                {
                    "results": [],
                    "papers_stored": len(stored),
                    "hint": (
                        "No passage matched. Retrieval is lexical: retry with the "
                        "wording a paper would use, or drop the paper_id filter."
                        if stored
                        else "No paper full text is stored for this project yet."
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        return json.dumps({"results": results}, ensure_ascii=False, sort_keys=True)

    return StructuredTool.from_function(
        func=_search_paper_text,
        name="search_paper_text",
        description=(
            "Search the full text of papers stored for this project and return "
            "locating passages (C-* IDs) with matched snippets. This is the "
            "counterpart to `search_observations`: experiences (E-*) give "
            "transferable judgements and conclusions, while paper text gives "
            "verifiable evidence, concrete numbers, and implementation detail. "
            "Use this when you need to quote a paper's own wording, check a "
            "reported metric or hyperparameter, or when the experience records "
            "are not specific enough to decide. Then read a promising passage "
            "with `read_paper`."
        ),
        args_schema=SearchPaperTextArgs,
        infer_schema=False,
    )


def create_read_paper_tool(*, memory_dir: str | Path, project_id: str) -> BaseTool:
    """Build the `read_paper` tool for one project context."""

    def _read_paper(
        chunk_id: str = "",
        paper_id: str = "",
        expand: str = "section",
        max_chars: int = 0,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> str:
        active_project = runtime_project_id(runtime, project_id)
        requested_chunk = chunk_id.strip()
        requested_paper = paper_id.strip()
        granularity = expand.strip().casefold() or "section"
        if granularity not in {"chunk", "section", "full"}:
            return json.dumps(
                {"error": "expand must be one of: chunk, section, full."},
                ensure_ascii=False,
                sort_keys=True,
            )

        if granularity == "full":
            if not requested_paper:
                return json.dumps(
                    {"error": "expand=full requires paper_id."},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            result = read_paper_full(
                memory_dir=memory_dir,
                project_id=active_project,
                paper_id=requested_paper,
                max_chars=max_chars or DEFAULT_FULL_PAPER_MAX_CHARS,
            )
            if result is None:
                return json.dumps(
                    {
                        "error": (
                            "No stored full text for that paper_id in this project. "
                            "Use search_paper_text to find a stored paper."
                        )
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            if result["truncated"]:
                # Mark the cut inline so the model cannot mistake a truncated
                # body for the paper's real ending.
                result["text"] += (
                    f"\n\n[truncated at {result['returned_chars']} of "
                    f"{result['char_count']} characters; raise max_chars or use "
                    "search_paper_text to locate the remaining sections]"
                )
            return json.dumps(result, ensure_ascii=False, sort_keys=True)

        if not requested_chunk:
            return json.dumps(
                {"error": "chunk_id is required unless expand=full with paper_id."},
                ensure_ascii=False,
                sort_keys=True,
            )
        result = read_paper_chunk(
            memory_dir=memory_dir,
            project_id=active_project,
            chunk_id=requested_chunk,
            expand=granularity,
            max_chars=max_chars or DEFAULT_SECTION_MAX_CHARS,
        )
        if result is None:
            return json.dumps(
                {
                    "error": (
                        "No stored passage with that chunk_id in this project. "
                        "Chunk IDs come from search_paper_text."
                    )
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        if result["truncated"]:
            result["text"] += "\n\n[truncated; raise max_chars to read further]"
        return json.dumps(result, ensure_ascii=False, sort_keys=True)

    return StructuredTool.from_function(
        func=_read_paper,
        name="read_paper",
        description=(
            "Read stored paper text at one of three granularities: a single "
            "passage (chunk_id with expand=chunk), the whole section "
            "containing it (chunk_id with expand=section, the default and "
            "usually the right choice for reading a result in context), or an "
            "entire paper (paper_id with expand=full) when a topic warrants "
            "deep reading rather than targeted lookup. Long responses are "
            "truncated with an explicit marker."
        ),
        args_schema=ReadPaperArgs,
        infer_schema=False,
    )


__all__ = [
    "create_read_paper_tool",
    "create_search_paper_text_tool",
]
