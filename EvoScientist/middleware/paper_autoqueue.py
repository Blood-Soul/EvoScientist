"""Queue every paper a literature search returns for experience extraction.

Enqueueing used to depend on the agent: `paper-navigator` asks it to call
`enqueue_paper_experiences` once the final paper set is settled, and in practice
an agent that runs the search scripts one by one never gets to that step. The
library then stays empty and the experience coach has nothing to say.

This middleware takes the decision away from the agent. After an `execute` call
that ran `scholar_search.py` it reads the papers the search found (the pool file
named by `--output`, plus any `arxiv:<id>` in the printed result), queues each
one and starts a drain of the queue in this process. The skill scripts run in
the agent's own shell, which has neither the memory folder nor the project id,
so the queueing has to happen here.

It queues what was *searched*, not what was read, so a run's library grows with
its searches rather than with the few papers the agent opens.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)

_ARXIV_IN_TEXT = re.compile(r"arxiv:(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_OUTPUT_ARG = re.compile(r"--output(?:=|\s+)(\"[^\"]+\"|'[^']+'|\S+)")
# How many times one drain is restarted because papers arrived while it was
# finishing. The lock inside the drain keeps two of them from running at once.
_MAX_REDRAINS = 20


def _result_text(result: Any) -> str:
    content = getattr(result, "content", "")
    if isinstance(content, str):
        return content
    return " ".join(
        str(part.get("text", "")) if isinstance(part, dict) else str(part)
        for part in content or []
    )


def papers_from_pool(path: Path) -> dict[str, str]:
    """`{arxiv id: title}` for the arXiv papers in a search pool (jsonl)."""
    found: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return found
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        identifier = ""
        external = row.get("externalIds")
        if isinstance(external, dict) and external.get("ArXiv"):
            identifier = str(external["ArXiv"])
        else:
            match = _ARXIV_IN_TEXT.search(str(row.get("paperId") or ""))
            identifier = match.group(1) if match else ""
        identifier = re.sub(r"v\d+$", "", identifier.strip())
        if re.fullmatch(r"\d{4}\.\d{4,5}", identifier):
            found.setdefault(identifier, str(row.get("title") or "").strip())
    return found


class PaperAutoQueueMiddleware(AgentMiddleware):
    """Queue the papers of every `scholar_search.py` call and drain the queue."""

    name = "paper_autoqueue"

    def __init__(
        self,
        *,
        memory_dir: str | Path,
        project_id: str,
        workspace_root: str | Path,
        max_papers: int = 80,
        drain: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        super().__init__()
        self._memory_dir = Path(memory_dir).expanduser()
        self._project_id = project_id
        self._root = Path(workspace_root)
        self._max_papers = max_papers
        self._queued = 0
        self._drain_fn = drain or self._drain_queue
        self._drain_task: asyncio.Task[Any] | None = None
        self._redrains = 0

    @property
    def queued(self) -> int:
        return self._queued

    # -- what a search found -------------------------------------------------

    def _pool_path(self, command: str) -> Path | None:
        match = _OUTPUT_ARG.search(command)
        if not match:
            return None
        raw = match.group(1).strip("\"'")
        path = Path(raw)
        return path if path.is_absolute() else self._root / path

    def _found(self, command: str, text: str) -> dict[str, str]:
        found: dict[str, str] = {}
        pool = self._pool_path(command)
        if pool is not None:
            found.update(papers_from_pool(pool))
        for identifier in _ARXIV_IN_TEXT.findall(text):
            found.setdefault(identifier, "")
        return found

    # -- queueing --------------------------------------------------------------

    def _enqueue(self, found: dict[str, str]) -> int:
        from ..memory.experiences import enqueue_paper

        created = 0
        for identifier, title in found.items():
            if self._queued >= self._max_papers:
                break
            try:
                _task, is_new = enqueue_paper(
                    memory_dir=self._memory_dir,
                    project_id=self._project_id,
                    url=f"https://arxiv.org/abs/{identifier}",
                    paper_id=identifier,
                    title=title,
                )
            except Exception:
                logger.warning("Could not queue %s", identifier, exc_info=True)
                continue
            if is_new:
                created += 1
                self._queued += 1
        return created

    def _pending(self) -> int:
        from ..memory.experiences.queue import QUEUE_DIR

        folder = self._memory_dir / QUEUE_DIR / self._project_id / "pending"
        try:
            return sum(1 for _ in folder.glob("*.json"))
        except OSError:
            return 0

    async def _drain_queue(self) -> None:
        from ..EvoScientist import _ensure_auxiliary_chat_model
        from ..memory.agents.paper_experience_worker import (
            drain_paper_experience_queue,
        )
        from .utils import quiet_model

        # Extraction runs inside the agent's graph node: untagged, the model's
        # JSON would be streamed into the transcript as the agent's own text.
        model = quiet_model(_ensure_auxiliary_chat_model())
        return await drain_paper_experience_queue(
            memory_dir=self._memory_dir, project_id=self._project_id, model=model
        )

    def _start_drain(self) -> None:
        if self._drain_task is not None and not self._drain_task.done():
            return
        self._drain_task = asyncio.create_task(self._drain_and_recheck())

    async def _drain_and_recheck(self) -> None:
        try:
            result = await self._drain_fn()
        except Exception:
            logger.warning("Paper experience drain failed", exc_info=True)
            return
        # A drain that did nothing (another worker holds the queue lock, which
        # is then draining it) must not be repeated: it would spin on files it
        # cannot take. A drain that did work may have missed papers queued while
        # it was finishing, so look once more.
        did_work = isinstance(result, dict) and (
            int(result.get("processed", 0)) + int(result.get("failed", 0)) > 0
        )
        # The pending count reads the queue folder: off the event loop, which
        # refuses synchronous file access under `langgraph dev`.
        pending = await asyncio.to_thread(self._pending) if did_work else 0
        if pending and self._redrains < _MAX_REDRAINS:
            self._redrains += 1
            await self._drain_and_recheck()

    # -- middleware surface ----------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        result = await handler(request)
        try:
            call = request.tool_call
            if call.get("name") != "execute":
                return result
            command = str((call.get("args") or {}).get("command") or "")
            if "scholar_search.py" not in command:
                return result
            found = await asyncio.to_thread(
                self._found, command, _result_text(result)
            )
            if found and await asyncio.to_thread(self._enqueue, found):
                self._start_drain()
        except Exception:
            # Queueing is a side effect of a search; it must never fail the search.
            logger.warning("Paper auto-queue failed", exc_info=True)
        return result


def create_paper_autoqueue_middleware(
    *,
    memory_dir: str | Path,
    project_id: str,
    workspace_root: str | Path,
    max_papers: int = 80,
) -> PaperAutoQueueMiddleware:
    return PaperAutoQueueMiddleware(
        memory_dir=memory_dir,
        project_id=project_id,
        workspace_root=workspace_root,
        max_papers=max_papers,
    )


__all__ = [
    "PaperAutoQueueMiddleware",
    "create_paper_autoqueue_middleware",
    "papers_from_pool",
]
