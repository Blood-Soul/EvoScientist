"""End-to-end orchestration: retrieve → rerank → synthesize → cache.

Kept separate from the stages so each stage stays unit-testable without a
model, and so the tool layer has exactly one function to call.

Both model calls are intermediate work on the auxiliary model. Nothing here
produces user-facing prose: the policy is a structured object the acting agent
reads and then answers from with the main model.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

from .prompts import load_rerank_prompt, load_writer_prompt
from .select import (
    DEFAULT_MAX_SELECTED,
    DEFAULT_RETRIEVE_LIMIT,
    gather_candidates,
    rerank_candidates,
)
from .store import load_cached_policy, store_policy
from .synthesize import synthesize_policy
from .trace import emit_trace_async

logger = logging.getLogger(__name__)


def _model_name(model: Any) -> str:
    for attribute in ("model_name", "model", "name"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


async def derive_policy(
    *,
    memory_dir: str | Path,
    project_id: str,
    task: str,
    state: str = "",
    retrieve_limit: int = DEFAULT_RETRIEVE_LIMIT,
    max_selected: int = DEFAULT_MAX_SELECTED,
    model: Any | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Derive one target-bound reuse policy for ``task``.

    Returns a report carrying the policy plus the selection trail, so the
    caller can see which records were considered, which were used, and whether
    the result was synthesized or served from cache. An empty candidate set is
    a normal outcome, reported as ``status="no_candidates"`` rather than an
    error -- a project with no relevant experience should fall through to live
    search, not stall.
    """
    call_id = uuid.uuid4().hex[:12]
    await emit_trace_async(
        memory_dir,
        "request",
        call_id=call_id,
        project_id=project_id,
        task=task.strip(),
        state=state,
        max_selected=max_selected,
        refresh=refresh,
    )

    if model is None:
        from ...EvoScientist import _ensure_auxiliary_chat_model

        model = _ensure_auxiliary_chat_model()

    candidates = await _gather(
        memory_dir=memory_dir,
        project_id=project_id,
        task=task,
        retrieve_limit=retrieve_limit,
        call_id=call_id,
    )
    if not candidates:
        report = {
            "status": "no_candidates",
            "project_id": project_id,
            "task": task.strip(),
            "considered": [],
            "policy": None,
            "hint": (
                "No stored experience matched this task. Retrieval is lexical: "
                "retry with the terminology a paper would use, or proceed with "
                "live search and `search_paper_text`."
            ),
        }
        await emit_trace_async(memory_dir, "report", call_id=call_id, **report)
        return report

    rerank_prompt = await load_rerank_prompt()
    ranked = await rerank_candidates(
        candidates=candidates,
        task=task,
        max_selected=max_selected,
        model=model,
        prompt=rerank_prompt,
        memory_dir=memory_dir,
        call_id=call_id,
    )
    selected = ranked["selected"]
    if not selected:
        report = {
            "status": "no_reusable_memory",
            "project_id": project_id,
            "task": task.strip(),
            "considered": [item["id"] for item in candidates],
            "policy": None,
            "selection_reason": ranked.get("reason", ""),
            "hint": (
                "Candidates were retrieved but none offers a transferable "
                "procedure for this task. Proceed without experience reuse."
            ),
        }
        await emit_trace_async(memory_dir, "report", call_id=call_id, **report)
        return report

    selected_ids = [item["id"] for item in selected]
    if not refresh:
        cached = await _load_cached(
            memory_dir=memory_dir,
            project_id=project_id,
            task=task,
            selected_ids=selected_ids,
        )
        if cached is not None:
            report = {
                "status": "ok",
                "cached": True,
                "project_id": project_id,
                "task": task.strip(),
                "considered": [item["id"] for item in candidates],
                "selected": selected_ids,
                "selection_reason": ranked.get("reason", ""),
                "policy": cached,
            }
            await emit_trace_async(memory_dir, "report", call_id=call_id, **report)
            return report

    policy = await synthesize_policy(
        task=task,
        selected=selected,
        state=state,
        model=model,
        prompt=await load_writer_prompt(),
        memory_dir=memory_dir,
        call_id=call_id,
    )
    await _store(
        memory_dir=memory_dir,
        project_id=project_id,
        task=task,
        selected_ids=selected_ids,
        policy=policy,
        model_name=_model_name(model),
    )
    report = {
        "status": "ok",
        "cached": False,
        "project_id": project_id,
        "task": task.strip(),
        "considered": [item["id"] for item in candidates],
        "selected": selected_ids,
        "selection_reason": ranked.get("reason", ""),
        "policy": policy,
    }
    await emit_trace_async(memory_dir, "report", call_id=call_id, **report)
    return report


async def _gather(
    *,
    memory_dir: str | Path,
    project_id: str,
    task: str,
    retrieve_limit: int,
    call_id: str,
) -> list[dict[str, Any]]:
    return await asyncio.to_thread(
        gather_candidates,
        memory_dir=memory_dir,
        project_id=project_id,
        query=task,
        limit=retrieve_limit,
        call_id=call_id,
    )


async def _load_cached(
    *, memory_dir: str | Path, project_id: str, task: str, selected_ids: list[str]
) -> dict[str, Any] | None:
    return await asyncio.to_thread(
        load_cached_policy,
        memory_dir=memory_dir,
        project_id=project_id,
        task=task,
        selected_ids=selected_ids,
    )


async def _store(
    *,
    memory_dir: str | Path,
    project_id: str,
    task: str,
    selected_ids: list[str],
    policy: dict[str, Any],
    model_name: str,
) -> None:
    """Persist the policy; a cache write failure must not fail the tool."""
    try:
        await asyncio.to_thread(
            store_policy,
            memory_dir=memory_dir,
            project_id=project_id,
            task=task,
            selected_ids=selected_ids,
            policy=policy,
            model_name=model_name,
        )
    except OSError:
        logger.warning("Failed to cache derived reuse policy", exc_info=True)


__all__ = ["derive_policy"]
