"""Keep the coordinating agent from writing code files itself.

In a run where the research tree is mandatory (`research_tree_require_plans`), the main agent plans,
reviews and decides; code is written inside a node directory by a sub-agent. The first end-to-end
run showed that a prompt alone does not hold: the main agent wrote its own preprocessing and training
code in the project root and never used the tree. This middleware is on the main agent only and turns
a `write_file` / `edit_file` aimed at a code file into an error that says what to do instead.
Sub-agents have their own middleware stack and are not affected.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ToolCallRequest

GUARDED_TOOLS = frozenset({"write_file", "edit_file"})
CODE_SUFFIXES = frozenset(
    {".py", ".ipynb", ".sh", ".bash", ".js", ".ts", ".c", ".cc", ".cpp", ".cu", ".r", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
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
    """Main agent only: no code files written by the coordinator."""

    name = "coordinator_guard"

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
