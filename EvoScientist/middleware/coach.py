"""Experience coach: push stored paper experience at the decisions that need it.

`apply_experience` is a *pull*. The agent has to notice that it is making a
decision, remember the tool exists, and write the retrieval query itself. In
practice all three fail together, so this middleware inverts the direction: it
works out what experience bears on the agent's next decision and puts the result in
front of the agent, without being asked.

Roles (see ``memory/policy``):

* the **task brief** is built once from the task statement, every fact carrying a
  quotation checked against the statement. It is the only source of what the task
  requires;
* the **gate** reads the brief, the agent's recent actions and a catalog of the
  library, decides whether stored experience bears on the decision, and names the
  records it wants;
* the **reranker** narrows the named records; the **writer** turns them into
  guidance for this decision.

Timing. The three model calls take about a minute, far longer than an agent step, so
nothing waits for them by default:

* a derivation **starts** only after a substantive action (writing or editing a
  file, launching a job), or when the agent has gone ``idle_trigger_steps`` steps
  without one. Reading files and polling processes are not decisions;
* it runs **beside** the agent. The result is injected into the first model call
  after it finishes;
* if it has been outstanding for ``max_wait_steps`` agent steps the agent is made to
  wait for it, so guidance never arrives arbitrarily late;
* guidance delivered after the agent has acted since the snapshot starts with a line
  saying what the agent did meanwhile, so the agent can judge whether it still
  applies. It is never dropped silently for being old.

Three properties are deliberate:

**The injection is per-call, never persisted.** `ModelRequest.override` returns a
new request; only the returned `ModelResponse` is written to graph state. So the
guidance is visible to the model exactly once, and the trajectory the next step
caches is byte-identical to the one this step started from.

**Most steps must cost nothing.** Before any model is asked, code answers for free:
an empty library has nothing to say; the agent's last action was a pure lookup; no
substantive action has happened since the last derivation.

**Failure is silence.** Reuse enhances the agent's own reasoning; a coach that cannot
run must be invisible, not disruptive.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

_STATS_TTL_SECONDS = 30.0
_HISTORY_LIMIT = 8

# Tools that gather context rather than commit to anything. A step whose last action
# was one of these is mid-lookup: the agent is still reading.
_READ_ONLY_TOOLS = frozenset(
    {
        "list_experience",
        "ls",
        "read_file",
        "read_memory",
        "read_paper",
        "search_experience",
        "search_observations",
        "search_paper_text",
        "glob",
        "grep",
        "check_process",
    }
)

# Actions that commit the agent to something. A derivation is started after one of
# these: it is the moment a decision has just been taken or is about to be.
_SUBSTANTIVE_TOOLS = frozenset(
    {"write_file", "edit_file", "run_in_background", "execute", "task"}
)


class ExperienceCoachMiddleware(AgentMiddleware):
    """Decide per step whether stored experience should shape the next action."""

    name = "experience_coach"

    def __init__(
        self,
        *,
        memory_dir: str | Path,
        project_id: str,
        model: Any | None = None,
        max_selected: int = 4,
        recent_messages: int = 6,
        background: bool = False,
        max_wait_steps: int = 5,
        max_wait_seconds: float = 90.0,
        idle_trigger_steps: int = 12,
        recent_actions: int = 10,
        candidate_limit: int = 24,
    ) -> None:
        super().__init__()
        self._memory_dir = Path(memory_dir).expanduser()
        self._project_id = project_id
        self._model = model
        self._max_selected = max_selected
        self._recent_messages = recent_messages
        self._recent_actions = recent_actions
        self._candidate_limit = candidate_limit
        self._stats: dict[str, Any] | None = None
        self._stats_at: float = 0.0
        self._gate_prompt: str | None = None
        self._brief: Any | None = None
        self._brief_for: str = ""
        self._last_picked: tuple[str, ...] | None = None
        self._last_suggestion: str = ""
        self._history: deque[dict[str, Any]] = deque(maxlen=_HISTORY_LIMIT)
        # Background mode and its bookkeeping.
        self._background = background
        self._max_wait_steps = max_wait_steps
        # A forced wait must end: if a derivation hangs (model endpoint stuck, proxy
        # down) the agent would otherwise hang with it. After this many seconds the
        # derivation is cancelled and the agent carries on without guidance.
        self._max_wait_seconds = max_wait_seconds
        self._idle_trigger_steps = idle_trigger_steps
        self._step = 0
        self._pending: asyncio.Task[str] | None = None
        self._pending_step = 0
        self._pending_started = 0.0
        self._pending_actions: list[str] = []
        self._substantive_seen = 0
        self._last_started_substantive = 0
        # Idle time is counted from the start of the run, so nothing is derived
        # before the agent has done anything at all.
        self._last_started_step = 0
        self._recorded_tool_ids: set[str] = set()
        self._since_snapshot: list[str] = []

    # -- inspection ---------------------------------------------------------

    @property
    def interventions(self) -> list[dict[str, Any]]:
        """Interventions so far this process, oldest first (for tests and debug)."""
        return list(self._history)

    # -- cheap, model-free precondition checks ------------------------------

    def _cached_stats(self) -> dict[str, Any] | None:
        """Library counts, refreshed at most once per TTL (disk walk, run in a thread)."""
        now = time.monotonic()
        if self._stats is None or (now - self._stats_at) > _STATS_TTL_SECONDS:
            try:
                from ..memory.experiences import experience_library_stats

                self._stats = experience_library_stats(
                    memory_dir=self._memory_dir, project_id=self._project_id
                )
            except Exception:
                logger.debug("Experience library stats unavailable", exc_info=True)
                self._stats = {}
            self._stats_at = now
        return self._stats

    @staticmethod
    def _last_tool_names(messages: list[Any]) -> set[str]:
        """Names of the tools whose results the current step is reading."""
        names: set[str] = set()
        for message in reversed(messages):
            if getattr(message, "type", "") == "tool":
                name = getattr(message, "name", "") or ""
                if name:
                    names.add(name)
                continue
            break
        return names

    def _request_text(self, messages: list[Any]) -> str:
        """The originating request: the first real human turn."""
        from ..utils import format_message_content

        for message in messages:
            if isinstance(message, HumanMessage) and not (
                message.additional_kwargs.get("lc_source")
            ):
                return format_message_content(message)
        return ""

    def _skip_reason(self, messages: list[Any], stats: dict[str, Any] | None) -> str:
        """Return a reason to skip without any model call, or ``""`` to proceed."""
        if not messages:
            return "no messages yet"
        if not stats or not stats.get("records"):
            return "experience library is empty"
        tool_names = self._last_tool_names(messages)
        if tool_names and tool_names <= _READ_ONLY_TOOLS:
            return f"mid-lookup after {', '.join(sorted(tool_names))}"
        return ""

    # -- the task brief -----------------------------------------------------

    def _resolve_model(self) -> Any:
        if self._model is None:
            from ..EvoScientist import _ensure_auxiliary_chat_model

            self._model = _ensure_auxiliary_chat_model()
        from .utils import quiet_model

        self._model = quiet_model(self._model)
        return self._model

    async def _trace(self, event: str, **fields: Any) -> None:
        from ..memory.policy.trace import emit_trace_async

        await emit_trace_async(self._memory_dir, event, **fields)

    async def _task_brief(self, statement: str) -> Any:
        """Build the brief once per distinct task statement."""
        if self._brief is not None and self._brief_for == statement:
            return self._brief
        from ..memory.policy import build_brief

        brief = await build_brief(statement, self._resolve_model())
        self._brief, self._brief_for = brief, statement
        await self._trace("coach_brief", **brief.to_dict())
        return brief

    async def _load_gate_prompt(self) -> str:
        if self._gate_prompt is None:
            from ..memory.policy import load_gate_prompt

            self._gate_prompt = await load_gate_prompt()
        return self._gate_prompt

    # -- the coaching step --------------------------------------------------

    async def _suggestion(self, messages: list[Any]) -> str:
        """Run the coach for one decision point and return the guidance to inject.

        Returns ``""`` whenever nothing should be injected, which is the common
        case. Every failure mode also returns ``""``.
        """
        call_id = uuid.uuid4().hex[:12]
        stats = await asyncio.to_thread(self._cached_stats)
        skip = self._skip_reason(messages, stats)
        if skip:
            await self._trace("coach_skip", call_id=call_id, reason=skip)
            return ""

        from ..memory.policy import (
            decide_experience_need,
            derive_policy,
            render_activity,
            render_catalog,
            render_suggestion,
            situation_terms,
        )

        try:
            brief = await self._task_brief(self._request_text(messages))
            brief_text = brief.render()
            activity = render_activity(messages, actions=self._recent_actions)
            catalog = await asyncio.to_thread(
                render_catalog,
                memory_dir=self._memory_dir,
                project_id=self._project_id,
                # Built in code from text that exists, so the candidates do not
                # depend on the gate having phrased anything correctly.
                query=situation_terms(brief_text, activity),
                candidates=self._candidate_limit,
            )
            decision = await decide_experience_need(
                brief_text=brief_text,
                activity_text=activity,
                library_map=catalog["map"],
                candidates_text=catalog["candidates"],
                allowed_ids=catalog["candidate_ids"],
                model=self._resolve_model(),
                prompt=await self._load_gate_prompt(),
                memory_dir=self._memory_dir,
                call_id=call_id,
            )
        except Exception:
            logger.debug("Experience gate failed; skipping coach", exc_info=True)
            await self._trace("coach_skip", call_id=call_id, reason="gate call failed")
            return ""

        if not decision["need"]:
            return ""

        picked = tuple(decision["pick"])
        if picked == self._last_picked and self._last_suggestion:
            await self._trace("coach_reuse", call_id=call_id, pick=list(picked))
            return self._last_suggestion

        try:
            report = await derive_policy(
                memory_dir=self._memory_dir,
                project_id=self._project_id,
                task=decision["purpose"],
                state=activity,
                max_selected=self._max_selected,
                model=self._resolve_model(),
                brief=brief_text,
                situation=decision["situation"],
                picked_ids=list(picked),
            )
        except Exception:
            logger.debug("Policy derivation failed; skipping coach", exc_info=True)
            await self._trace(
                "coach_skip", call_id=call_id, reason="derive_policy failed"
            )
            return ""

        suggestion = render_suggestion(report)
        await self._trace(
            "coach_inject",
            call_id=call_id,
            situation=decision["situation"],
            purpose=decision["purpose"],
            pick=list(picked),
            status=report.get("status"),
            cached=report.get("cached"),
            selected=report.get("selected") or [],
            injected=bool(suggestion),
            suggestion=suggestion,
        )
        if not suggestion:
            return ""
        self._last_picked = picked
        self._last_suggestion = suggestion
        self._history.append(
            {
                "call_id": call_id,
                "situation": decision["situation"],
                "purpose": decision["purpose"],
                "reason": decision["reason"],
                "selected": report.get("selected") or [],
            }
        )
        return suggestion

    # -- middleware surface -------------------------------------------------

    @staticmethod
    def _inject(request: ModelRequest, suggestion: str) -> ModelRequest:
        """Append the guidance as one extra human turn for this call only."""
        return request.override(
            messages=[
                *request.messages,
                HumanMessage(
                    content=suggestion,
                    additional_kwargs={"lc_source": "experience_coach"},
                ),
            ]
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # The coach is async end to end; the sync path is a pass-through.
        return handler(request)

    # -- background timing --------------------------------------------------

    def _observe_actions(self, messages: list[Any]) -> None:
        """Count new substantive actions and remember what the agent did meanwhile."""
        for message in messages:
            if getattr(message, "type", "") != "ai":
                continue
            for call in getattr(message, "tool_calls", None) or []:
                identifier = str(call.get("id", "")) if isinstance(call, dict) else ""
                name = str(call.get("name", "")) if isinstance(call, dict) else ""
                if not identifier or identifier in self._recorded_tool_ids:
                    continue
                self._recorded_tool_ids.add(identifier)
                if name in _SUBSTANTIVE_TOOLS:
                    self._substantive_seen += 1
                    args = call.get("args") or {}
                    target = ""
                    if isinstance(args, dict):
                        target = str(
                            args.get("file_path")
                            or args.get("command")
                            or args.get("path")
                            or ""
                        )[:80]
                    self._since_snapshot.append(f"{name} {target}".strip())

    def _should_start(self) -> bool:
        """Start a derivation after a substantive action, or after a long idle stretch."""
        if self._pending is not None:
            return False
        if self._substantive_seen > self._last_started_substantive:
            return True
        return self._step - self._last_started_step >= self._idle_trigger_steps

    def _start_background(self, messages: list[Any]) -> None:
        self._pending_step = self._step
        self._pending_started = time.monotonic()
        self._last_started_substantive = self._substantive_seen
        self._last_started_step = self._step
        self._since_snapshot = []
        self._pending = asyncio.create_task(self._derive_quietly(messages))

    async def _derive_quietly(self, messages: list[Any]) -> str:
        """Background body: never raises, so a failed derivation is just silence."""
        try:
            return await self._suggestion(messages)
        except Exception:
            logger.debug("Experience coach failed in background", exc_info=True)
            return ""

    async def _collect(self, *, block: bool) -> str:
        """Take the finished guidance; with ``block`` wait for it to finish."""
        task = self._pending
        if task is None:
            return ""
        if not task.done():
            if not block:
                return ""
            await asyncio.wait({task}, timeout=self._max_wait_seconds)
            if not task.done():
                task.cancel()
                self._pending = None
                await self._trace(
                    "coach_timeout",
                    waited_seconds=round(time.monotonic() - self._pending_started, 1),
                    age_steps=self._step - self._pending_step,
                )
                return ""
        self._pending = None
        try:
            suggestion = task.result()
        except BaseException:  # cancelled or failed: stay silent
            return ""
        if not suggestion:
            return ""
        age_steps = self._step - self._pending_step
        meanwhile = list(self._since_snapshot)
        await self._trace(
            "coach_deliver",
            age_steps=age_steps,
            age_seconds=round(time.monotonic() - self._pending_started, 1),
            blocked=block,
            actions_meanwhile=meanwhile[:6],
            chars=len(suggestion),
        )
        if meanwhile:
            note = (
                "Note: this guidance was prepared before your last "
                f"{len(meanwhile)} commit(s) ("
                + "; ".join(meanwhile[-4:])
                + "). Check it against what you have already done; it may no "
                "longer apply."
            )
            return f"{note}\n\n{suggestion}"
        return suggestion

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        messages = list(request.messages)
        if self._background:
            self._step += 1
            self._observe_actions(messages)
            overdue = (
                self._pending is not None
                and self._step - self._pending_step >= self._max_wait_steps
            )
            suggestion = await self._collect(block=overdue)
            if self._should_start():
                self._start_background(messages)
            if suggestion:
                request = self._inject(request, suggestion)
            return await handler(request)
        try:
            suggestion = await self._suggestion(messages)
        except Exception:
            logger.debug("Experience coach failed; proceeding", exc_info=True)
            suggestion = ""
        if suggestion:
            request = self._inject(request, suggestion)
        return await handler(request)


def create_experience_coach_middleware(
    *,
    memory_dir: str | Path,
    project_id: str,
    model: Any | None = None,
    max_selected: int = 4,
    recent_messages: int = 6,
    background: bool | None = None,
) -> ExperienceCoachMiddleware:
    """Factory for the experience coach (main agent only).

    ``background`` defaults to on: the coach derives guidance beside the agent and
    makes it wait only when a derivation has been outstanding for several steps. Set
    ``EVOSCIENTIST_COACH_BLOCKING=1`` (or pass ``background=False``) for blocking
    behaviour, which is what an ablation of the change needs.
    """
    if background is None:
        blocking = os.environ.get("EVOSCIENTIST_COACH_BLOCKING", "").strip().lower()
        background = blocking not in {"1", "true", "yes", "on"}
    return ExperienceCoachMiddleware(
        memory_dir=memory_dir,
        project_id=project_id,
        model=model,
        max_selected=max_selected,
        recent_messages=recent_messages,
        background=background,
    )


__all__ = [
    "ExperienceCoachMiddleware",
    "create_experience_coach_middleware",
]
