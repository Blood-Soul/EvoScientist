"""Deliver coordination notes (:mod:`EvoScientist.research_tree.coord`) at the end of an agent's context.

A note says that something changed: the GPU went to another node or came free, a node asked a question or asked
for the GPU, a background job finished. Each one is delivered once, to the agent it is addressed to, as a message
after the agent's last one, before its next model call. Nothing earlier in the context changes, so the prompt
cache of everything before the note stays valid; a note never goes into the system prompt.

On the main agent the audience is ``main``; on a node agent it is the node its background job runs
(:data:`~EvoScientist.research_tree.coord.CURRENT_NODE`). An agent outside both (any other sub-agent) gets nothing.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage

from ..research_tree.coord import CURRENT_NODE, MAIN, Hub

HEADER = "## Notes from the run (what changed since your last step)"


class CoordNotesMiddleware(AgentMiddleware):
    """Append the notes for this agent, once each, after its last message."""

    name = "coord_notes"

    def __init__(self, hub: Hub, *, main: bool) -> None:
        super().__init__()
        self.hub = hub
        self.main = main

    def _audience(self) -> str:
        return MAIN if self.main else CURRENT_NODE.get()

    def _update(self, state: Any) -> dict[str, Any] | None:
        audience = self._audience()
        if not audience:
            return None
        notes = self.hub.take_notes(audience)
        if not notes:
            return None
        return {"messages": [HumanMessage(content=HEADER + "\n\n" + "\n".join(f"- {n}" for n in notes))]}

    def before_model(self, state, runtime):
        return self._update(state)

    async def abefore_model(self, state, runtime):
        return self._update(state)
