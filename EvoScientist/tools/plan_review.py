"""The `review_plan` tool: a finished plan, read against the experience library, with suggestions that cite their records.

The plan-writer calls it once it has written a plan file. The tool reads that file and the task description, finds the
experience records that bear on the plan's choices, and returns suggestions. Each one names the records it rests on. The
plan-writer then revises the plan or refuses a suggestion; nothing is applied for it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field

from ..memory.policy.review import DEFAULT_MAX_SUGGESTIONS, review_plan
from ..memory.runtime_context import runtime_project_id


class ReviewPlanArgs(BaseModel):
    path: str = Field(
        min_length=1,
        description="The plan file you wrote, relative to the workspace root (for example `reports/drafts/plan-x.md`).",
    )
    task: str = Field(
        min_length=1,
        description="The task description you were given for this plan, copied as it was written: the direction, the data "
        "facts, the working conditions. The review reads it next to the plan.",
    )
    runtime: Annotated[object | None, InjectedToolArg] = None


def create_review_plan_tool(
    *, memory_dir: str | Path, project_id: str, workspace_root: str | Path, max_suggestions: int = DEFAULT_MAX_SUGGESTIONS
) -> BaseTool:
    root = Path(workspace_root)

    async def _review_plan(
        path: str, task: str, runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None
    ) -> str:
        target = (root / path.strip().lstrip("/")).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            return json.dumps({"status": "error", "error": f"no plan file at {path}; write the plan first"})
        from ..EvoScientist import _ensure_auxiliary_chat_model
        from ..middleware.utils import quiet_model

        try:
            result = await review_plan(
                memory_dir=memory_dir,
                project_id=runtime_project_id(runtime, project_id),
                task=task,
                plan=target.read_text(encoding="utf-8", errors="replace"),
                model=quiet_model(_ensure_auxiliary_chat_model()),
                max_suggestions=max_suggestions,
            )
        except Exception as error:  # a review is an aid; its failure must not block the plan
            return json.dumps({"status": "error", "error": str(error),
                               "hint": "The review failed. Finish the plan without it."})
        return result["text"]

    return StructuredTool.from_function(
        coroutine=_review_plan,
        name="review_plan",
        description=(
            "Read the plan you just wrote against this project's stored paper experience and get suggestions, each tied to "
            "the experience records it rests on. Call it once, after the plan file is written and before you report. "
            "The suggestions are advice: take one and revise the plan file, or refuse it and say why in your reply. "
            "`read_memory` on an `E-*` id shows the record behind a suggestion."
        ),
        args_schema=ReviewPlanArgs,
        infer_schema=False,
    )


__all__ = ["create_review_plan_tool"]
