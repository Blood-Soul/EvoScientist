"""Foreground paper-experience extraction for explicit user requests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field

from ..memory.experiences import (
    download_paper_text,
    load_experience_prompts,
    load_paper_experiences,
    run_experience_extraction,
    store_paper_experiences,
)
from ..memory.experiences.store import paper_storage_key
from ..memory.papers import (
    has_paper_text,
    paper_fulltext_settings,
    persist_paper_fulltext,
)
from ..memory.runtime_context import runtime_project_id

ACTIVE_EXTRACTION_CONCURRENCY = 2


class ActivePaperItem(BaseModel):
    url: str = Field(min_length=1, description="Resolvable paper or full-text URL.")
    paper_id: str = Field(default="", description="arXiv, DOI, or other paper ID.")
    title: str = Field(default="", description="Paper title when known.")
    domain_arxiv: str | None = Field(
        default=None,
        description="Primary arXiv category from paper metadata, when known.",
    )


class ExtractPaperExperiencesArgs(BaseModel):
    papers: list[ActivePaperItem] = Field(
        min_length=1,
        max_length=5,
        description="One to five papers selected for foreground extraction.",
    )
    refresh: bool = Field(
        default=False,
        description="Re-extract even when this project already has the paper.",
    )
    runtime: Annotated[object | None, InjectedToolArg] = None


def create_extract_paper_experiences_tool(
    *, memory_dir: str | Path, project_id: str
) -> BaseTool:
    """Build the immediate extraction tool backed by the shared project store."""

    async def _extract_paper_experiences(
        papers: list[ActivePaperItem],
        refresh: bool = False,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> str:
        active_project = runtime_project_id(runtime, project_id)
        semaphore = asyncio.Semaphore(ACTIVE_EXTRACTION_CONCURRENCY)
        prompt_lock = asyncio.Lock()
        prompts: dict[str, str] | None = None

        async def get_prompts() -> dict[str, str]:
            nonlocal prompts
            if prompts is not None:
                return prompts
            async with prompt_lock:
                if prompts is None:
                    prompts = await asyncio.to_thread(load_experience_prompts)
            return prompts

        async def backfill_fulltext(paper: ActivePaperItem, paper_id: str) -> bool:
            """Ensure this paper's text is stored, downloading it if missing.

            Experiences cached before full-text persistence existed have no
            paper.md, and a cache hit never reaches the extraction download, so
            the gap would otherwise never close. Fetching here costs one Jina
            request -- no metered API call, no extraction tokens -- and only
            happens once per paper, since the next hit finds the stored text.
            """
            if not paper_fulltext_settings().enabled:
                return False
            key = paper_storage_key(paper_id, paper.url)
            if await asyncio.to_thread(
                has_paper_text, memory_dir, project_id=active_project, paper_key=key
            ):
                return True
            # Share the extraction semaphore: a batch of cached papers would
            # otherwise fire every backfill download at once, unbounded.
            async with semaphore:
                try:
                    paper_text = await download_paper_text(paper.url)
                except Exception:
                    # The cached experiences are still a valid result; a failed
                    # backfill only means this paper has no full text yet.
                    return False
                stored = await asyncio.to_thread(
                    persist_paper_fulltext,
                    memory_dir=memory_dir,
                    project_id=active_project,
                    paper_id=paper_id,
                    url=paper.url,
                    title=paper.title,
                    paper_text=paper_text,
                    domain_arxiv=paper.domain_arxiv,
                )
            return stored is not None

        async def one(paper: ActivePaperItem) -> dict[str, Any]:
            paper_id = paper.paper_id.strip() or paper.url.strip()
            if not refresh:
                cached = await asyncio.to_thread(
                    load_paper_experiences,
                    memory_dir=memory_dir,
                    project_id=active_project,
                    paper_id=paper_id,
                    url=paper.url,
                )
                if cached is not None:
                    return {
                        "paper_id": paper_id,
                        "title": paper.title,
                        "url": paper.url,
                        "cached": True,
                        "full_text_available": await backfill_fulltext(paper, paper_id),
                        "l1": cached["l1"],
                        "l2": cached["l2"],
                    }
            async with semaphore:
                active_prompts = await get_prompts()
                paper_text = await download_paper_text(paper.url)
                # Persist before extraction so the raw text survives an
                # extraction failure; the call never raises.
                stored = await asyncio.to_thread(
                    persist_paper_fulltext,
                    memory_dir=memory_dir,
                    project_id=active_project,
                    paper_id=paper_id,
                    url=paper.url,
                    title=paper.title,
                    paper_text=paper_text,
                    domain_arxiv=paper.domain_arxiv,
                )
                payloads = await run_experience_extraction(
                    paper_id=paper_id,
                    paper_text=paper_text,
                    prompts=active_prompts,
                    domain_arxiv=paper.domain_arxiv,
                )
                await asyncio.to_thread(
                    store_paper_experiences,
                    memory_dir=memory_dir,
                    project_id=active_project,
                    paper_id=paper_id,
                    url=paper.url,
                    title=paper.title,
                    paper_text=paper_text,
                    prompts=active_prompts,
                    payloads=payloads,
                    domain_arxiv=paper.domain_arxiv,
                )
                return {
                    "paper_id": paper_id,
                    "title": paper.title,
                    "url": paper.url,
                    "cached": False,
                    "full_text_available": stored is not None,
                    "l1": payloads["l1"],
                    "l2": payloads["l2"],
                }

        results = await asyncio.gather(
            *(one(paper) for paper in papers), return_exceptions=True
        )
        completed: list[dict[str, Any]] = []
        failed: list[dict[str, str]] = []
        for paper, result in zip(papers, results, strict=True):
            if isinstance(result, Exception):
                failed.append(
                    {
                        "paper_id": paper.paper_id or paper.url,
                        "title": paper.title,
                        "url": paper.url,
                        "error": str(result),
                    }
                )
            else:
                completed.append(result)
        return json.dumps(
            {
                "project_id": active_project,
                "completed": completed,
                "failed": failed,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    return StructuredTool.from_function(
        coroutine=_extract_paper_experiences,
        name="extract_paper_experiences",
        description=(
            "Immediately download one to five explicitly selected papers, extract "
            "their L1 practical and L2 inductive experiences, persist them (and "
            "each paper's full text, for later `search_paper_text`) in the active "
            "project's store, and return the full payloads. Cached papers whose "
            "full text was not previously stored will have it fetched and saved, "
            "so `full_text_available` becomes true. Use this for explicit "
            "foreground extraction requests; paper-navigator's automatic "
            "background accumulation uses enqueue_paper_experiences."
        ),
        args_schema=ExtractPaperExperiencesArgs,
        infer_schema=False,
    )
