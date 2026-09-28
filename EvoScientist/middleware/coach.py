"""Experience coach: push stored paper experience at the step that needs it.

`apply_experience` is a *pull*. The agent has to notice that it is making a
decision, remember the tool exists, and write the retrieval query itself. In
practice all three fail together: the tool is not in the always-include set, so
above the tool-selector threshold it can be filtered out of the request entirely;
and when it does get called, the query is whatever sentence the agent wrote about
its own situation, which is rarely the vocabulary a paper's abstract would use.

This middleware inverts the direction. On every model request it decides, on the
auxiliary model, whether stored experience bears on the step about to be taken --
and if it does, derives the retrieval facets itself, runs the existing reuse
pipeline, and injects the resulting *guidance* into that one call. The agent is
not asked to do anything; it finds the advice already in front of it.

Three properties are deliberate:

**The reuse pipeline is unchanged.** `derive_policy` still does the work and the
nine-field policy is still the product. The coach supplies a better-phrased query
and better timing; it does not re-implement query-conditioned reuse. `apply_experience`
stays registered for the subagents whose YAML grants it, so push and pull share
one implementation and one cache.

**The injection is per-call, never persisted.** `ModelRequest.override` returns a
new request; only the returned `ModelResponse` is written to graph state. So the
guidance is visible to the model exactly once, and the trajectory the next step
caches is byte-identical to the one this step started from. Any scheme that wrote
the guidance into history -- or, worse, rewrote an older one -- would invalidate
every cached token after the edit point, on every step.

**Most steps must cost nothing.** The gate is one auxiliary call, which is cheap
but not free, and it runs on every step. So the short-circuits below run first, in
code, without a model: an empty library has no answer to give; a step that
follows a pure lookup is not a decision point; and facets identical to the last
intervention's reuse the guidance already derived. A step that survives all three
costs one call to ask, two more only if the answer is yes.

Cost per step, worst case: 0 short-circuited, 1 gate says no, 2 nothing rerank
selects, 3 a full hit that misses the policy cache.
"""

from __future__ import annotations

import asyncio
import logging
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

# Library statistics are ~2ms on a real library and already computed once per
# model request by the memory index block, but the coach runs on a different
# schedule and must not add a second disk walk per step.
_STATS_TTL_SECONDS = 30.0

# How many past interventions to remember. This exists to answer "did we already
# say this?" -- not to build a history the model reads. It is instance state, so
# it never enters the trajectory and never affects the prompt prefix.
_HISTORY_LIMIT = 8

# Tools that gather context rather than commit to anything. A step whose last
# action was one of these is mid-lookup: the agent is still reading, and a
# decision it has not reached yet cannot be coached. Writes, code execution and
# delegation are absent on purpose -- those are exactly the commitments worth
# advising before, not after.
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
    }
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
    ) -> None:
        super().__init__()
        self._memory_dir = Path(memory_dir).expanduser()
        self._project_id = project_id
        self._model = model
        self._max_selected = max_selected
        self._recent_messages = recent_messages
        self._stats: dict[str, Any] | None = None
        self._stats_at: float = 0.0
        self._prompt: str | None = None
        # Last intervention, kept so an unchanged decision does not pay for the
        # same derivation twice in a row.
        self._last_facets: tuple[str, str] | None = None
        self._last_suggestion: str = ""
        self._history: deque[dict[str, Any]] = deque(maxlen=_HISTORY_LIMIT)

    # -- inspection ---------------------------------------------------------

    @property
    def interventions(self) -> list[dict[str, Any]]:
        """Interventions so far this process, oldest first (for tests and debug)."""
        return list(self._history)

    # -- cheap, model-free precondition checks ------------------------------

    def _cached_stats(self) -> dict[str, Any] | None:
        """Library counts, refreshed at most once per TTL.

        Walks the library directory, so the async path calls this through
        ``asyncio.to_thread`` -- never inline on the event loop, where it would
        also trip langgraph dev's blockbuster detector.
        """
        now = time.monotonic()
        if self._stats is None or (now - self._stats_at) > _STATS_TTL_SECONDS:
            try:
                from ..memory.experiences import experience_library_stats

                self._stats = experience_library_stats(
                    memory_dir=self._memory_dir, project_id=self._project_id
                )
            except Exception:
                # An unreadable library is a reason to stay quiet, not to fail
                # the agent's turn. Cache the failure so a broken path does not
                # get re-walked on every step.
                logger.debug("Experience library stats unavailable", exc_info=True)
                self._stats = {}
            self._stats_at = now
        return self._stats

    @staticmethod
    def _last_tool_names(messages: list[Any]) -> set[str]:
        """Names of the tools whose results the current step is reading.

        Read backwards over the trailing ToolMessages to the AIMessage that
        requested them, so a parallel batch is judged as one step. An agent that
        fired `search_experience` and `read_memory` together is still looking
        things up.
        """
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
        """The originating request: the first real human turn.

        Synthetic HumanMessages (summarization, injected context) carry an
        ``lc_source`` marker and are skipped -- the gate needs the goal the run
        was given, which a summary paraphrases and an injection replaces.
        """
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

    # -- the coaching step --------------------------------------------------

    async def _load_prompt(self) -> str:
        if self._prompt is None:
            from ..memory.policy import load_gate_prompt

            self._prompt = await load_gate_prompt()
        return self._prompt

    def _resolve_model(self) -> Any:
        if self._model is None:
            from ..EvoScientist import _ensure_auxiliary_chat_model

            self._model = _ensure_auxiliary_chat_model()
        return self._model

    async def _trace(self, event: str, **fields: Any) -> None:
        from ..memory.policy.trace import emit_trace_async

        await emit_trace_async(self._memory_dir, event, **fields)

    async def _suggestion(self, messages: list[Any]) -> str:
        """Run the coach for one step and return the guidance to inject.

        Returns ``""`` whenever nothing should be injected, which is the common
        case. Every failure mode also returns ``""``: reuse is an enhancement to
        the agent's own reasoning, so a coach that cannot run must be invisible
        rather than disruptive.
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
            render_library,
            render_recent,
            render_suggestion,
        )

        try:
            decision = await decide_experience_need(
                request_text=self._request_text(messages),
                recent_text=render_recent(messages, limit=self._recent_messages),
                library_text=render_library(stats),
                model=self._resolve_model(),
                prompt=await self._load_prompt(),
                memory_dir=self._memory_dir,
                call_id=call_id,
            )
        except Exception:
            logger.debug("Experience gate failed; skipping coach", exc_info=True)
            await self._trace("coach_skip", call_id=call_id, reason="gate call failed")
            return ""

        if not decision["need"]:
            return ""

        facets = (decision["topic"], decision["method"])
        if facets == self._last_facets and self._last_suggestion:
            # Same question as last time. Retrieval and rerank would re-run two
            # aux calls before the policy cache is even consulted, and return the
            # guidance already in hand.
            await self._trace(
                "coach_reuse", call_id=call_id, topic=facets[0], method=facets[1]
            )
            return self._last_suggestion

        try:
            report = await derive_policy(
                memory_dir=self._memory_dir,
                project_id=self._project_id,
                task=decision["task"],
                state=decision["state"],
                method=decision["method"],
                max_selected=self._max_selected,
                model=self._resolve_model(),
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
            topic=decision["topic"],
            method=decision["method"],
            task=decision["task"],
            status=report.get("status"),
            cached=report.get("cached"),
            selected=report.get("selected") or [],
            injected=bool(suggestion),
            suggestion=suggestion,
        )
        if not suggestion:
            return ""

        self._last_facets = facets
        self._last_suggestion = suggestion
        self._history.append(
            {
                "call_id": call_id,
                "topic": decision["topic"],
                "method": decision["method"],
                "reason": decision["reason"],
                "selected": report.get("selected") or [],
            }
        )
        return suggestion

    # -- middleware surface -------------------------------------------------

    @staticmethod
    def _inject(request: ModelRequest, suggestion: str) -> ModelRequest:
        """Append the guidance as one extra human turn for this call only.

        It goes at the end of ``messages`` rather than into the system message
        because the advice is about *this* step: the system prompt is a stable
        prefix shared by every call, and appending there would put step-specific
        text in front of the whole trajectory, re-caching it each step. A
        trailing message is legal wherever the loop pauses for a model call --
        tool results have already landed by then -- and is dropped as soon as the
        call returns, since only the response is persisted.
        """
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
        # The coach is async end to end (two awaited model calls and threaded
        # disk reads). Driving that from the sync path would mean owning an event
        # loop inside a middleware, so the sync path stays a pass-through and the
        # coach is an async-execution feature.
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        try:
            suggestion = await self._suggestion(list(request.messages))
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
) -> ExperienceCoachMiddleware:
    """Factory for the experience coach (main agent only).

    ``model`` is left unset by callers that want the configured auxiliary model,
    resolved lazily on first use. Worth knowing: ``auxiliary_model`` defaults to
    the main model, so the coach's cost argument only holds once a cheap
    auxiliary is actually configured.
    """
    return ExperienceCoachMiddleware(
        memory_dir=memory_dir,
        project_id=project_id,
        model=model,
        max_selected=max_selected,
        recent_messages=recent_messages,
    )


__all__ = [
    "ExperienceCoachMiddleware",
    "create_experience_coach_middleware",
]
