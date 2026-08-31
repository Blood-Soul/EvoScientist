"""The `apply_experience` tool: stored experience, rewritten for the task at hand.

Reading an `E-*` record directly delivers a ~2.5k-character account of what one
paper's authors did on their datasets with their models -- the procedure and the
obsolete values arrive together, and an actor copies both. This tool inserts the
reuse step: it selects the relevant records and returns a compact policy stating
what transfers, what must be re-derived here, what blocks reuse, and what to
verify.

Deliberately not folded into `search_observations`: that tool locates records,
this one transforms them, and the transformation costs two aux-model calls. It
must be invoked when a decision is actually being made, not on every search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field

from ..memory.policy import (
    DEFAULT_MAX_SELECTED,
    DEFAULT_RETRIEVE_LIMIT,
    derive_policy,
)
from ..memory.runtime_context import runtime_project_id

MAX_SELECTED_CEILING = 6


class ApplyExperienceArgs(BaseModel):
    """Model-facing arguments for the `apply_experience` tool."""

    task: str = Field(
        min_length=1,
        description=(
            "The decision you are about to make, stated concretely: the goal, "
            "the domain, and the constraint that matters. This is what the "
            "policy is conditioned on, so a specific task ('choose an "
            "evaluation protocol for a 7B retrieval-augmented agent on "
            "materials data') yields a usable policy where a bare topic "
            "('agents') does not."
        ),
    )
    state: str = Field(
        default="",
        description=(
            "What is already true in this project: available compute, data on "
            "hand, decisions already fixed, results already obtained. The "
            "policy uses this to tell what must still be re-derived from what "
            "you already have. Leave empty at the start of a project."
        ),
    )
    # Defaults to None rather than a literal so the configured project value is
    # what applies when the model omits the argument. A default baked in here
    # would be filled in by schema validation and shadow the configuration.
    max_selected: int | None = Field(
        default=None,
        ge=1,
        le=MAX_SELECTED_CEILING,
        description=(
            "How many stored experiences to draw the policy from. More records "
            "widen coverage and surface conflicts; fewer keep the policy tight. "
            "Omit to use this project's configured default."
        ),
    )
    refresh: bool = Field(
        default=False,
        description=(
            "Re-synthesize instead of reusing the cached policy for this exact "
            "task and record set. Use after new papers have been added."
        ),
    )
    runtime: Annotated[object | None, InjectedToolArg] = None


def create_apply_experience_tool(
    *,
    memory_dir: str | Path,
    project_id: str,
    max_selected: int = DEFAULT_MAX_SELECTED,
) -> BaseTool:
    """Build the `apply_experience` tool for one project context.

    ``max_selected`` is the configured default the model gets when it does not
    pass one; the model may still ask for fewer or more, up to
    ``MAX_SELECTED_CEILING``.
    """
    configured_max_selected = min(max(int(max_selected), 1), MAX_SELECTED_CEILING)

    async def _apply_experience(
        task: str,
        state: str = "",
        max_selected: int | None = None,
        refresh: bool = False,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> str:
        active_project = runtime_project_id(runtime, project_id)
        try:
            report = await derive_policy(
                memory_dir=memory_dir,
                project_id=active_project,
                task=task,
                state=state,
                retrieve_limit=DEFAULT_RETRIEVE_LIMIT,
                max_selected=max_selected or configured_max_selected,
                refresh=refresh,
            )
        except Exception as error:
            # Reuse is an enhancement over live search, never a prerequisite.
            # A failed synthesis returns a usable signal instead of breaking
            # the caller's turn.
            return json.dumps(
                {
                    "status": "error",
                    "error": str(error),
                    "hint": (
                        "Policy synthesis failed. Continue with "
                        "`search_observations`, `search_paper_text`, or live "
                        "search rather than blocking on experience reuse."
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        return json.dumps(report, ensure_ascii=False, sort_keys=True)

    return StructuredTool.from_function(
        coroutine=_apply_experience,
        name="apply_experience",
        description=(
            "Turn this project's stored paper experiences into a reuse policy "
            "for the decision you are making now. Returns the procedure that "
            "transfers, the source-specific values you must re-derive rather "
            "than copy (`rebind`), the preconditions, what does not transfer, "
            "the checks to run before concluding, and any conflicts between "
            "records. Use this instead of reading `E-*` records directly "
            "whenever you are planning an experiment, choosing a method, "
            "evaluation protocol, baseline, or parameter, or judging whether a "
            "published result applies here. A `decline` verdict is a valid, "
            "useful answer: it means the stored experience does not apply. "
            "`read_memory` on an `E-*` ID remains the path for auditing the "
            "evidence behind a policy line."
        ),
        args_schema=ApplyExperienceArgs,
        infer_schema=False,
    )


__all__ = ["create_apply_experience_tool"]
