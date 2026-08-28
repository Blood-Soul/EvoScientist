"""Deterministic background worker for queued paper experiences."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, TypedDict

from filelock import AsyncFileLock, Timeout
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..experiences import (
    claim_next_task,
    complete_task,
    download_paper_text,
    fail_task,
    load_experience_prompts,
    queue_worker_lock_path,
    requeue_running_tasks,
    run_experience_extraction,
    store_paper_experiences,
)
from ..papers import persist_paper_fulltext
from ._factory import resolve_memory_agent_paths


class PaperExperienceWorkerState(TypedDict, total=False):
    processed: int
    failed: int


async def drain_paper_experience_queue(
    *, memory_dir: str | Path, project_id: str, model: Any | None = None
) -> dict[str, int]:
    """Drain all currently pending tasks for one project."""
    lock_path = queue_worker_lock_path(memory_dir, project_id)
    # The LangGraph dev event loop rejects synchronous filesystem operations.
    # Queue persistence deliberately uses normal file APIs, so keep that work
    # off the event loop while retaining async download/model calls below.
    await asyncio.to_thread(lock_path.parent.mkdir, parents=True, exist_ok=True)
    try:
        lock = AsyncFileLock(lock_path)
        await lock.acquire(timeout=0)
    except Timeout:
        return {"processed": 0, "failed": 0}
    try:
        await asyncio.to_thread(
            requeue_running_tasks, memory_dir=memory_dir, project_id=project_id
        )
        processed = failed = 0
        while task := await asyncio.to_thread(
            claim_next_task, memory_dir=memory_dir, project_id=project_id
        ):
            try:
                paper_text = await download_paper_text(task.url)
                # Persist before extracting: the full text keeps its value if
                # extraction fails, and a retry then reuses it instead of
                # re-downloading. persist_paper_fulltext never raises, so a
                # storage problem cannot fail a task whose real job is
                # extraction.
                await asyncio.to_thread(
                    persist_paper_fulltext,
                    memory_dir=memory_dir,
                    project_id=project_id,
                    paper_id=task.paper_id or task.url,
                    url=task.url,
                    title=task.title,
                    paper_text=paper_text,
                    domain_arxiv=task.domain_arxiv,
                )
                prompts = await asyncio.to_thread(load_experience_prompts)
                payloads = await run_experience_extraction(
                    paper_id=task.paper_id or task.url,
                    paper_text=paper_text,
                    prompts=prompts,
                    model=model,
                    domain_arxiv=task.domain_arxiv,
                )
                await asyncio.to_thread(
                    store_paper_experiences,
                    memory_dir=memory_dir,
                    project_id=project_id,
                    paper_id=task.paper_id or task.url,
                    url=task.url,
                    title=task.title,
                    paper_text=paper_text,
                    prompts=prompts,
                    payloads=payloads,
                    domain_arxiv=task.domain_arxiv,
                )
                await asyncio.to_thread(complete_task, memory_dir=memory_dir, task=task)
                processed += 1
            except Exception as exc:
                await asyncio.to_thread(
                    fail_task, memory_dir=memory_dir, task=task, error=str(exc)
                )
                failed += 1
        return {"processed": processed, "failed": failed}
    finally:
        await lock.release()


def build_paper_experience_worker_graph(
    *, memory_dir: str | Path | None = None
) -> CompiledStateGraph:
    """Build a one-node graph suitable for LangGraph background dispatch."""
    paths = resolve_memory_agent_paths(memory_dir=memory_dir)

    async def drain(
        _state: PaperExperienceWorkerState, config: RunnableConfig
    ) -> PaperExperienceWorkerState:
        configurable = config.get("configurable", {})
        project_id = str(configurable.get("evomemory_project_id") or "").strip()
        if not project_id:
            raise ValueError("evomemory_project_id is required")
        return await drain_paper_experience_queue(
            memory_dir=paths.memory_dir, project_id=project_id
        )

    graph = StateGraph(PaperExperienceWorkerState)
    graph.add_node("drain", drain)
    graph.add_edge(START, "drain")
    graph.add_edge("drain", END)
    return graph.compile()
