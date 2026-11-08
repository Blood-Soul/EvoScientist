"""The coordinator guard also makes sure the research tree is started.

`execute` heredocs cannot be refused, so a run where the model spends its first
half hour building in the project root — as r26 did — is steered back by a note
on the model call once several tool calls have passed without `tree_init`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from EvoScientist.middleware.coordinator_guard import CoordinatorGuardMiddleware


def _request(calls: int) -> ModelRequest:
    messages = [HumanMessage("do the task")]
    messages += [
        AIMessage(content="", tool_calls=[{"name": "execute", "args": {"command": "ls"}, "id": f"c{i}"}])
        for i in range(calls)
    ]
    return ModelRequest(
        model=SimpleNamespace(),
        messages=messages,
        state={"messages": messages},
        runtime=SimpleNamespace(),
        system_message=SystemMessage("base prompt"),
    )


def _seen(guard: CoordinatorGuardMiddleware, request: ModelRequest) -> ModelRequest:
    seen: list[ModelRequest] = []

    def handler(req: ModelRequest) -> str:
        seen.append(req)
        return "model"

    assert guard.wrap_model_call(request, handler) == "model"
    return seen[0]


def _started(tmp_path: Path) -> Path:
    (tmp_path / ".research").mkdir()
    (tmp_path / ".research" / "state.json").write_text("{}")
    return tmp_path


def test_a_bare_guard_does_not_change_the_request() -> None:
    request = _request(20)
    assert _seen(CoordinatorGuardMiddleware(), request) is request


def test_several_calls_without_a_tree_are_told_to_start_it(tmp_path: Path) -> None:
    out = _seen(CoordinatorGuardMiddleware(tmp_path), _request(6))
    assert "tree_init" in out.system_message.text and "base prompt" in out.system_message.text


def test_early_calls_are_left_alone(tmp_path: Path) -> None:
    request = _request(5)
    assert _seen(CoordinatorGuardMiddleware(tmp_path), request) is request


def test_a_started_tree_is_left_alone(tmp_path: Path) -> None:
    request = _request(20)
    assert _seen(CoordinatorGuardMiddleware(_started(tmp_path)), request) is request


@pytest.mark.asyncio
async def test_the_note_is_also_on_the_async_path(tmp_path: Path) -> None:
    guard = CoordinatorGuardMiddleware(tmp_path)
    seen: list[ModelRequest] = []

    async def handler(req: ModelRequest) -> str:
        seen.append(req)
        return "model"

    assert await guard.awrap_model_call(_request(6), handler) == "model"
    assert "tree_init" in seen[0].system_message.text
