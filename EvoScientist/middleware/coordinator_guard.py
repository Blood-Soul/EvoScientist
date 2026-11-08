"""Keep the coordinating agent from writing code files itself, and make the tree start.

In a run where the research tree is mandatory (`research_tree_require_plans`), the main agent plans,
reviews and decides; code is written inside a node directory by a sub-agent. The first end-to-end
run showed that a prompt alone does not hold: the main agent wrote its own preprocessing and training
code in the project root and never used the tree. This middleware is on the main agent only and turns
a `write_file` / `edit_file` aimed at a code file into an error that says what to do instead.
Sub-agents have their own middleware stack and are not affected.

A later run showed the other half of the same failure: the model read the "Start ... call tree_init"
line, then spent its first half hour writing scripts with `execute` heredocs and training a baseline
in the project root, never calling `tree_init`. `execute` cannot be refused (it is how real work is
launched), so instead: once the agent has used several tool calls and still has no tree, every model
call carries a note that the tree must be started before further work.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelRequest, ToolCallRequest

GUARDED_TOOLS = frozenset({"write_file", "edit_file"})
CODE_SUFFIXES = frozenset(
    {".py", ".ipynb", ".sh", ".bash", ".js", ".ts", ".c", ".cc", ".cpp", ".cu", ".r", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
)

# A few calls of latitude for "put the task data in place"; after that a run
# that has not opened its tree is off the rails and is told so every turn.
START_AFTER_CALLS = 6

NOT_STARTED = (
    "## The research tree is not started\n\n"
    "This run is coordinated through the research tree and it still has no tree. Do not keep "
    "building in the project root: call `tree_init` now with the direction, metric, deliverable "
    "and time budget, then `tree_survey`, before any further work. The first baseline is a node: "
    "open it with `tree_new` and its code is written by a sub-agent."
)

REFUSAL = (
    "Refused: in this run you coordinate and do not write code files. Code is written inside a node "
    "directory (nodes/<id>/) by a sub-agent. Ask plan-writer (task) for a plan, record it with "
    "tree_plan(path), open a node with tree_new(plan_id=...), and give the sub-agent the brief. If you "
    "only need a placeholder deliverable, create it with a single `execute` command."
)


def refuses(tool_call: dict[str, Any]) -> bool:
    """True for a write or edit of a code file."""
    if tool_call.get("name") not in GUARDED_TOOLS:
        return False
    args = tool_call.get("args") or {}
    path = str(args.get("file_path") or args.get("path") or "")
    return PurePosixPath(path).suffix.lower() in CODE_SUFFIXES


def _refusal(request: ToolCallRequest) -> ToolMessage:
    call = request.tool_call
    return ToolMessage(
        content=REFUSAL,
        tool_call_id=call["id"],
        name=call.get("name", "write_file"),
        status="error",
    )


class CoordinatorGuardMiddleware(AgentMiddleware):
    """Main agent only: no code files written by the coordinator, tree started early."""

    name = "coordinator_guard"

    def __init__(self, workspace_root: str | Path | None = None,
                 start_after_calls: int = START_AFTER_CALLS) -> None:
        # A missing root only disables the start note (tests build the guard bare).
        self._root = Path(workspace_root) if workspace_root is not None else None
        self._start_after = start_after_calls

    def _tree_started(self) -> bool:
        # tree_init writes .research/state.json (research_tree/tree.py); reading
        # the fact off disk, not the transcript, keeps it true after compaction.
        return self._root is not None and (self._root / ".research" / "state.json").exists()

    def _tool_calls(self, state: Any) -> int:
        messages = state.get("messages") if isinstance(state, dict) else None
        if not isinstance(messages, (list, tuple)):
            return 0
        return sum(len(m.tool_calls or []) for m in messages if isinstance(m, AIMessage))

    def _prepare(self, request: ModelRequest) -> ModelRequest:
        if self._root is None or self._tree_started():
            return request
        if self._tool_calls(request.state) < self._start_after:
            return request
        from deepagents.middleware._utils import append_to_system_message

        return request.override(
            system_message=append_to_system_message(request.system_message, NOT_STARTED)
        )

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:
        return handler(self._prepare(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[Any]]
    ) -> Any:
        return await handler(self._prepare(request))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if refuses(request.tool_call):
            return _refusal(request)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if refuses(request.tool_call):
            return _refusal(request)
        return await handler(request)
