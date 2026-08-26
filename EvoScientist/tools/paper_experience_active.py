"""Foreground paper-experience extraction for explicit user requests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
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


def _runtime_project_id(runtime: ToolRuntime | None, default: str) -> str:
    if runtime is None or not isinstance(runtime.config, Mapping):
        return default
    configurable = runtime.config.get("configurable", {})
    if not isinstance(configurable, Mapping):
        return default
    value = configurable.get("evomemory_project_id")
    return value if isinstance(value, str) and value.strip() else default


def create_extract_paper_experiences_tool(
    *, memory_dir: str | Path, project_id: str
) -> BaseTool:
    """Build the immediate extraction tool backed by the shared project store."""

    async def _extract_paper_experiences(
        papers: list[ActivePaperItem],
        refresh: bool = False,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> str:
        active_project = _runtime_project_id(runtime, project_id)
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
                        "l1": cached["l1"],
                        "l2": cached["l2"],
                    }
            async with semaphore:
                active_prompts = await get_prompts()
                paper_text = await download_paper_text(paper.url)
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
            "their L1 practical and L2 inductive experiences, persist them in the "
            "active project's experience store, and return the full payloads. Use "
            "this for explicit foreground extraction requests; paper-navigator's "
            "automatic background accumulation uses enqueue_paper_experiences."
        ),
        args_schema=ExtractPaperExperiencesArgs,
        infer_schema=False,
    )
