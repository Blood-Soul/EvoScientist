"""Agent-facing tools over :class:`~.tree.ResearchTree`.

Every tool returns text. A refusal comes back as `ERROR: ...` with what to do
instead, so the agent can correct itself without an exception tearing down
the turn.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from .gitlog import GitError
from .judge import Judge
from .tree import CATEGORIES, KINDS, ResearchTree, TreeError


class InitArgs(BaseModel):
    budget_minutes: float | None = Field(
        default=None,
        description="Total minutes left for the whole task. Not needed when the harness fixed the deadline.",
    )
    metric: str = Field(
        default="score", description="Name of the primary metric, for display."
    )
    direction: str = Field(
        default="max", description="'max' if a higher score is better, 'min' otherwise."
    )
    prediction_file: str = Field(
        default="holdout_predictions.csv",
        description="File each node writes with predictions on the held-out rows; this is what the judge scores.",
    )
    deliverable: str = Field(
        default="submission.csv",
        description="The task's deliverable file. The champion's copy is kept at the project root.",
    )


class HistoryArgs(BaseModel):
    nodes: list[str] = Field(
        default_factory=list,
        description="Node ids to read. Empty reads every finished node, oldest first.",
    )
    full: bool = Field(
        default=False,
        description="False: hypothesis, change, observation, score and outcome. True: the complete record "
        "including the next step and diffstat.",
    )


class NewArgs(BaseModel):
    kind: str = Field(
        description=f"One of {', '.join(KINDS)}. explore: a new line from the root. refine: change one thing in a "
        "finished node (one parent). merge: combine two or more finished nodes. pilot: a small time-boxed "
        "experiment that only produces evidence for a decision; it is not scored."
    )
    title: str = Field(description="Short name of the step.")
    hypothesis: str = Field(
        description="What you expect and why, falsifiable: 'X should raise the metric because Y; it fails if Z'."
    )
    category: str = Field(
        default="method",
        description=f"One of {', '.join(CATEGORIES)}: whether this changes the method, the way it is evaluated, "
        "or tests how sensitive it is to a component.",
    )
    parents: list[str] = Field(
        default_factory=list,
        description="Finished, scored nodes this one starts from (see kind).",
    )
    based_on: list[str] = Field(
        default_factory=list,
        description="Finished node ids and insight ids (I1, ...) whose evidence led to this step. Required once "
        "there is history. If the history does not bear on the step, run a pilot first and cite it.",
    )
    minutes: float = Field(
        default=60.0,
        description="Time budget for the node. Clipped to the time left and to the kind's limits.",
    )
    gpu: bool = Field(
        default=True,
        description="False for work that does not need the GPU; only one GPU node runs at a time.",
    )


class FinishArgs(BaseModel):
    node: str = Field(description="The running node to close.")
    change: str = Field(
        description="What was changed, at the level of components (not parameter values)."
    )
    observation: str = Field(
        description="What was seen: what the errors look like, what surprised you. The score is added by the "
        "judge; do not state one as fact here."
    )
    next_step: str = Field(
        default="", description="What this result suggests trying next."
    )
    source: str = Field(
        default="", description="Paper or experience id the idea came from, if any."
    )
    abandon_reason: str = Field(
        default="",
        description="Set to drop the node without scoring it (dead end, no longer needed).",
    )


class InsightArgs(BaseModel):
    text: str = Field(
        default="",
        description="State and decision: 'when the situation looks like X, the step that worked or failed was Y'.",
    )
    evidence: list[str] = Field(
        default_factory=list, description="Finished node ids that show it."
    )
    status: str = Field(default="open", description="open, supported or contradicted.")
    insight_id: str = Field(
        default="",
        description="Set to update an existing insight (I1, ...) instead of adding one.",
    )


class NoArgs(BaseModel):
    pass


def create_research_tree_tools(
    root: str | Path,
    *,
    judge: Judge | None = None,
    now: Callable[[], float] | None = None,
) -> list[BaseTool]:
    """Build the research-tree tools for the project at `root`."""
    kwargs = {"judge": judge}
    if now is not None:
        kwargs["now"] = now
    tree = ResearchTree(root, **kwargs)

    def guarded(fn: Callable[..., str]) -> Callable[..., str]:
        def run(**args) -> str:
            try:
                return fn(**args)
            except (TreeError, GitError) as e:
                return f"ERROR: {e}"

        run.__name__ = fn.__name__
        run.__doc__ = fn.__doc__
        return run

    def tool(fn, name: str, description: str, schema: type[BaseModel]) -> BaseTool:
        return StructuredTool.from_function(
            func=guarded(fn),
            name=name,
            description=description,
            args_schema=schema,
            infer_schema=False,
        )

    return [
        tool(
            lambda **a: tree.init(**a),
            "tree_init",
            "Start the research tree for this task. Call once, at the start, after the task data is in place. "
            "It creates the git record and starts the clock.",
            InitArgs,
        ),
        tool(
            lambda **a: tree.status(),
            "tree_status",
            "Read the clock, the running node, the champion, every node with its score and outcome, which pairs of "
            "nodes are worth merging, and the insights. Call it before every decision.",
            NoArgs,
        ),
        tool(
            lambda **a: tree.history(a.get("nodes") or None, full=a.get("full", False)),
            "tree_history",
            "Read the record of finished nodes: what was tried, what was seen, what it scored, and what became of "
            "it. This is the evidence for your next decision; read the nodes relevant to it, in order. The same "
            "record is in git (`git log`, `git show node/<id>`).",
            HistoryArgs,
        ),
        tool(
            lambda **a: tree.new_node(**a),
            "tree_new",
            "Open a node: one hypothesis, one git branch and worktree at nodes/<id>/. Then hand that directory to a "
            "sub-agent to build it. Refused without based_on once there is history, with a second GPU node, or "
            "without time.",
            NewArgs,
        ),
        tool(
            lambda **a: tree.finish(a.pop("node"), **a),
            "tree_finish",
            "Close a node. The held-out predictions are scored by the frozen judge; the score, deltas against the "
            "parent and champion, and the outcome are written by this tool. You supply the narrative. An improved "
            "champion's deliverable is copied to the project root.",
            FinishArgs,
        ),
        tool(
            lambda **a: tree.insight(**a),
            "tree_insight",
            "Record or update a lesson drawn from several nodes. Cite the nodes that show it; mark it contradicted "
            "when a later node refutes it. Insights can be cited by later nodes.",
            InsightArgs,
        ),
        tool(
            lambda **a: tree.finalize(),
            "tree_finalize",
            "End the run: settle every node's outcome, tag the final solution, make sure the champion's deliverable "
            "is at the project root, and write TRAJECTORY.md. Do not call it while a node is running.",
            NoArgs,
        ),
    ]
