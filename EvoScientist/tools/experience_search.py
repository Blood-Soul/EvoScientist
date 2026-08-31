"""The `search_experience` and `list_experience` tools.

Experience retrieval used to be a mode of `search_observations`. The two stores
answer different questions -- observations record how this agent's tools and
environment behave, experiences record what published papers found -- and
sharing one entry point made the observation store's vocabulary the vocabulary
of both. The result was process-shaped queries ("how do I build an idea from an
abstract", "which APIs can find papers") aimed at records that only ever hold
subject-matter findings, so retrieval returned either nothing or confident
noise. Splitting the entry point is what lets each schema teach its own query
shape, which is the actual fix; the argument lists differ because the questions
differ, not for symmetry.

`list_experience` exists because retrieval here is lexical. A caller who does
not already know the library's vocabulary cannot phrase a query that reaches
it, and for an open-ended task ("what has been found about X") there may be no
single right phrasing. Browsing by structure sidesteps phrasing entirely.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field

from ..memory.experiences.retrieval import (
    browse_experience_facets,
    search_experience_records,
)
from ..memory.experiences.taxonomy import DISCIPLINES
from ..memory.runtime_context import runtime_project_id
from ..memory.types import ExperienceLevel

_DISCIPLINE_LIST = ", ".join(DISCIPLINES)


class SearchExperienceArgs(BaseModel):
    """Model-facing arguments for the `search_experience` tool.

    There is deliberately no `memory_type` and no `scope`. Both describe an
    observation's frontmatter; every `E-*` record is stored as
    semantic/project, so passing either could only ever be a no-op or, as in
    the merged retriever, silently empty the entire result set.
    """

    topic: str = Field(
        default="",
        description=(
            "The research SUBJECT to find experience about: the phenomenon, "
            "system class, problem, or field. This is content, not process. "
            "Good: 'automated scientific discovery', 'protein structure "
            "prediction', 'catalyst screening', 'open-ended agent learning'. "
            "Bad, and will match nothing, because these records come from "
            "papers and never describe your workflow: 'how do I generate ideas "
            "from abstracts', 'which API finds papers', 'what to do when a "
            "download fails'. If your task is to reason about a set of papers, "
            "the topic is what those papers are ABOUT."
        ),
    )
    method: str = Field(
        default="",
        description=(
            "Optional technique, architecture, or algorithm facet: "
            "'reinforcement learning from feedback', 'graph neural network', "
            "'multi-agent orchestration'. Searched separately from `topic` and "
            "fused, so a record matching both ranks above one matching either."
        ),
    )
    task: str = Field(
        default="",
        description=(
            "Optional capability or pipeline-stage facet: 'hypothesis "
            "generation', 'experiment design', 'benchmark construction', "
            "'result validation'. Names what the work accomplishes, still in "
            "subject-matter terms rather than in terms of your own steps."
        ),
    )
    discipline: str = Field(
        default="",
        description=(
            "Optional exact-match filter on the coarse field vocabulary: "
            f"{_DISCIPLINE_LIST}. Use it to keep a cross-disciplinary library "
            "from returning computer-science records for a chemistry question. "
            "Call `list_experience` to see which values actually hold records."
        ),
    )
    domain: str = Field(
        default="",
        description=(
            "Optional exact-match filter on the fine-grained domain label, as "
            "written by extraction (e.g. 'agent_learning'). Free text, so use a "
            "value `list_experience` reported rather than one you guessed; for "
            "approximate subject matching use `topic` instead."
        ),
    )
    level: ExperienceLevel | None = Field(
        default=None,
        description=(
            "Optional layer filter. l1 records one concrete practice from one "
            "paper: what was done, in what setting, with what measured result. "
            "l2 records an inductive claim generalized across observations. Omit "
            "to search both."
        ),
    )
    limit: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum number of experience records to return.",
    )
    runtime: Annotated[object | None, InjectedToolArg] = None


class ListExperienceArgs(BaseModel):
    """Model-facing arguments for the `list_experience` tool."""

    facet: Literal["discipline", "domain", "records"] = Field(
        default="discipline",
        description=(
            "What to list. `discipline` returns the bounded top-level fields "
            "with record counts -- start here. `domain` returns fine-grained "
            "domain labels with counts, ordered by count; narrow it with the "
            "`discipline` argument first. `records` returns individual record "
            "IDs and summaries, which only makes sense once the filters have "
            "narrowed the set to something readable."
        ),
    )
    discipline: str = Field(
        default="",
        description=(
            f"Optional filter applied before counting: one of {_DISCIPLINE_LIST}."
        ),
    )
    domain: str = Field(
        default="",
        description="Optional fine-grained domain filter applied before counting.",
    )
    level: ExperienceLevel | None = Field(
        default=None,
        description="Optional l1/l2 layer filter applied before counting.",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of facet values or records in this page.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Page offset. Every level is paged and reports its total, so a "
            "library of any size is walkable rather than truncated silently."
        ),
    )
    runtime: Annotated[object | None, InjectedToolArg] = None


def create_search_experience_tool(
    *,
    memory_dir: str | Path,
    project_id: str,
) -> BaseTool:
    """Build the read-only `search_experience` tool for one project context."""

    def _search_experience(
        topic: str = "",
        method: str = "",
        task: str = "",
        discipline: str = "",
        domain: str = "",
        level: ExperienceLevel | None = None,
        limit: int = 8,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> str:
        results = search_experience_records(
            memory_dir=memory_dir,
            project_id=runtime_project_id(runtime, project_id),
            topic=topic,
            method=method,
            task=task,
            discipline=discipline,
            domain=domain,
            level=level,
            limit=limit,
        )
        payload: dict[str, object] = {"results": results}
        if not results:
            # An empty page is ambiguous between "nothing matches" and "your
            # phrasing missed", and the second is the common case for a lexical
            # index. Name the recovery path instead of leaving the caller to
            # retry variations of the same wording.
            payload["hint"] = (
                "No experience matched. Retrieval here is lexical, so try "
                "`list_experience` to see which disciplines and domains hold "
                "records, then query with the library's own vocabulary. Broaden "
                "or drop `discipline`/`domain`/`level` filters, and check that "
                "`topic` names a research subject rather than a step in your "
                "own workflow."
            )
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    return StructuredTool.from_function(
        func=_search_experience,
        name="search_experience",
        description=(
            "Search experience extracted from published papers (`E-*`): "
            "methods, measured results, evaluation protocols, and "
            "subject-matter findings, across every discipline in this "
            "project's library.\n\n"
            "Query by SUBJECT, not by process. These records describe what "
            "other researchers did in their own settings; none of them "
            "describes your workflow, your tools, or your environment. If you "
            "are about to ask 'how do I ...' or 'what should I do when ...', "
            "this is the wrong store -- that is `search_observations`.\n\n"
            "Facets (`topic`, `method`, `task`) are searched separately and "
            "fused, so supplying two or three narrows results by agreement "
            "rather than diluting one long query string. `discipline`, "
            "`domain`, and `level` are exact-match filters. When you do not "
            "know the library's vocabulary, browse it with `list_experience` "
            "first. Read a full record with `read_memory`, or -- when you are "
            "making a decision rather than gathering context -- prefer "
            "`apply_experience`, which rebinds the source's fixed values to "
            "your task instead of leaving you to copy them."
        ),
        args_schema=SearchExperienceArgs,
        infer_schema=False,
    )


def create_list_experience_tool(
    *,
    memory_dir: str | Path,
    project_id: str,
) -> BaseTool:
    """Build the read-only `list_experience` tool for one project context."""

    def _list_experience(
        facet: str = "discipline",
        discipline: str = "",
        domain: str = "",
        level: ExperienceLevel | None = None,
        limit: int = 20,
        offset: int = 0,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> str:
        result = browse_experience_facets(
            memory_dir=memory_dir,
            project_id=runtime_project_id(runtime, project_id),
            facet=facet,
            discipline=discipline,
            domain=domain,
            level=level,
            limit=limit,
            offset=offset,
        )
        return json.dumps(result, ensure_ascii=False, sort_keys=True)

    return StructuredTool.from_function(
        func=_list_experience,
        name="list_experience",
        description=(
            "Browse the paper-experience library by structure instead of by "
            "keyword. Use this when you do not know what the library contains "
            "or how it words things -- which is the normal situation for an "
            "open-ended question, where no single phrasing is the right one.\n\n"
            "Walk it top-down: `facet=discipline` for the bounded field list "
            "with counts, then `facet=domain` (optionally filtered by "
            "discipline) for fine-grained subjects, then `facet=records` once "
            "the filters have narrowed to a readable set. Every level reports "
            "its total and accepts `limit`/`offset`, so the library stays "
            "walkable as it grows. Feed the values you find back into "
            "`search_experience` as `discipline`/`domain`, or read a promising "
            "record directly with `read_memory`."
        ),
        args_schema=ListExperienceArgs,
        infer_schema=False,
    )


__all__ = [
    "create_list_experience_tool",
    "create_search_experience_tool",
]
