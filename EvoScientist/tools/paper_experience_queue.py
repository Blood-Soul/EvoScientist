"""Runtime tool used by paper-navigator to enqueue final papers."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field

from ..memory.experiences import enqueue_paper
from ..memory.paper_experience_launch import launch_paper_experience_worker

logger = logging.getLogger(__name__)


class PaperQueueItem(BaseModel):
    url: str = Field(min_length=1, description="Resolvable paper or full-text URL.")
    paper_id: str = Field(default="", description="arXiv, DOI, S2, or other paper ID.")
    title: str = Field(default="", description="Paper title when known.")


class EnqueuePaperExperiencesArgs(BaseModel):
    papers: list[PaperQueueItem] = Field(
        min_length=1,
        description="The final papers returned to the user by paper-navigator.",
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


def create_paper_experience_queue_tool(
    *, memory_dir: str | Path, project_id: str
) -> BaseTool:
    """Build the batch enqueue tool without changing paper search behavior."""

    def _enqueue_paper_experiences(
        papers: list[PaperQueueItem],
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> str:
        active_project = _runtime_project_id(runtime, project_id)
        enqueued = existing = 0
        task_ids: list[str] = []
        for paper in papers:
            task, created = enqueue_paper(
                memory_dir=memory_dir,
                project_id=active_project,
                url=paper.url,
                paper_id=paper.paper_id,
                title=paper.title,
            )
            task_ids.append(task.task_id)
            if created:
                enqueued += 1
            else:
                existing += 1
        launched = False
        try:
            launched = launch_paper_experience_worker(active_project) is not None
        except Exception:
            logger.warning("Failed to launch paper experience worker", exc_info=True)
        return json.dumps(
            {
                "enqueued": enqueued,
                "existing": existing,
                "worker_launched": launched,
                "task_ids": task_ids,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    return StructuredTool.from_function(
        func=_enqueue_paper_experiences,
        name="enqueue_paper_experiences",
        description=(
            "Queue the final papers selected by paper-navigator for background "
            "L1/L2 experience extraction. Call once after the final paper set is "
            "determined and before returning the normal answer. This is asynchronous "
            "and must not alter the answer."
        ),
        args_schema=EnqueuePaperExperiencesArgs,
        infer_schema=False,
    )
