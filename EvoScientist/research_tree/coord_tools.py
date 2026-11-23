"""Tools over the coordination hub (:mod:`.coord`): the node agent's side and the main agent's side.

Node agent: `request_gpu(reason)` (waits in its own session until the lease is granted), `release_gpu()`,
`ask_main(question)` (waits for the answer, or continues with the skeleton's choice after a timeout).
Main agent: `gpu_grant(node)`, `gpu_revoke(node, reason)`, `answer_node(question_id, answer)`.

A node agent's tools act for the node its background job runs (:data:`.coord.CURRENT_NODE`); called outside one
they say so instead of guessing a node.
"""

from __future__ import annotations

import asyncio
import time

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from .coord import CURRENT_NODE, POLL_SECONDS, Hub

NODE_TOOLS = ("request_gpu", "release_gpu", "ask_main")
MAIN_TOOLS = ("gpu_grant", "gpu_revoke", "answer_node")


class RequestArgs(BaseModel):
    reason: str = Field(
        description="What the GPU is for, in one sentence: a check on the full data, a profiling run, the full "
        "training. The main agent decides who is next from it."
    )


class AskArgs(BaseModel):
    question: str = Field(
        description="One question, with what you found and the options you see, so it can be answered in one "
        "reply. Ask only when the data contradicts a premise of the skeleton or a choice in it should be overturned."
    )


class GrantArgs(BaseModel):
    node: str = Field(description="The open node to give the GPU (n3, ...).")


class RevokeArgs(BaseModel):
    node: str = Field(description="The node holding the GPU.")
    reason: str = Field(description="Why it is taken back; the node is told.")


class AnswerArgs(BaseModel):
    question_id: str = Field(description="The question's id (Q1, ...).")
    answer: str = Field(description="The answer, with the decision if the question asked for one.")


class NoArgs(BaseModel):
    pass


def _node() -> str:
    return CURRENT_NODE.get()


NOT_A_NODE = "Error: this agent is not running a node, so there is no GPU lease or main agent to ask for it."


def create_node_tools(hub: Hub) -> list[BaseTool]:
    """The node agent's tools."""

    def granted_text(nid: str) -> str:
        return (f"{nid} holds the GPU now. Commands you start from now on see it; a process started before does "
                "not. Call release_gpu as soon as the GPU work is done, so the next node can start.")

    def request_check(reason: str) -> tuple[str, str] | str:
        nid = _node()
        if not nid:
            return NOT_A_NODE
        if not reason.strip():
            return "Error: give the reason the node needs the GPU."
        if not hub.is_open(nid):
            return f"Error: {nid} is closed."
        return nid, hub.request(nid, reason.strip())

    def waited(nid: str, state: str) -> str:
        if state == "granted":
            return granted_text(nid)
        if state == "closed":
            return f"{nid} was closed while it waited for the GPU. Stop and give your final reply."
        return (f"Still waiting for the GPU after {hub.wait_seconds / 60:.0f} min; the request stays queued. Do any "
                "work that needs no GPU, then call request_gpu again to keep waiting.")

    def request_sync(reason: str) -> str:
        got = request_check(reason)
        if isinstance(got, str):
            return got
        nid, state = got
        end = time.monotonic() + hub.wait_seconds
        while state == "queued" and time.monotonic() < end:
            time.sleep(POLL_SECONDS)
            state = hub.lease(nid)
        return waited(nid, state)

    async def request_async(reason: str) -> str:
        got = request_check(reason)
        if isinstance(got, str):
            return got
        nid, state = got
        end = time.monotonic() + hub.wait_seconds
        while state == "queued" and time.monotonic() < end:
            await asyncio.sleep(POLL_SECONDS)
            state = hub.lease(nid)
        return waited(nid, state)

    def release() -> str:
        nid = _node()
        if not nid:
            return NOT_A_NODE
        try:
            killed = hub.release(nid)
        except ValueError as e:
            return f"Error: {e}."
        return "GPU released" + (f"; {len(killed)} process(es) still in the node directory were stopped" if killed else "") + "."

    def ask_check(question: str) -> tuple[str, str] | str:
        nid = _node()
        if not nid:
            return NOT_A_NODE
        if not question.strip():
            return "Error: the question is empty."
        return nid, hub.ask(nid, question.strip())

    def answered(qid: str) -> str:
        q = hub.answer_of(qid)
        if q["status"] == "answered":
            return f"The main agent's answer to {qid}: {q['answer']}"
        if q["status"] == "dropped":
            return "The node was closed before the question was answered. Stop and give your final reply."
        hub.expire(qid)
        return (f"No answer to {qid} within {hub.answer_seconds / 60:.0f} min. Continue with the skeleton's choice, "
                "and record in DESIGN.md that you asked, got no answer and kept the default. A late answer arrives "
                "as a note.")

    def ask_sync(question: str) -> str:
        got = ask_check(question)
        if isinstance(got, str):
            return got
        _nid, qid = got
        end = time.monotonic() + hub.answer_seconds
        while hub.answer_of(qid)["status"] == "open" and time.monotonic() < end:
            time.sleep(POLL_SECONDS)
        return answered(qid)

    async def ask_async(question: str) -> str:
        got = ask_check(question)
        if isinstance(got, str):
            return got
        _nid, qid = got
        end = time.monotonic() + hub.answer_seconds
        while hub.answer_of(qid)["status"] == "open" and time.monotonic() < end:
            await asyncio.sleep(POLL_SECONDS)
        return answered(qid)

    return [
        StructuredTool.from_function(
            func=request_sync, coroutine=request_async, name="request_gpu", args_schema=RequestArgs,
            infer_schema=False,
            description="Ask for the GPU, with the reason. Until you hold it your commands see no GPU. If it is free "
            "and nobody is waiting you get it at once; otherwise you wait here until the main agent gives it to you "
            "(or the earliest request gets it after a while). Ask only when the code is tested on a small slice on "
            "the CPU, or when a check really needs the GPU.",
        ),
        StructuredTool.from_function(
            func=release, name="release_gpu", args_schema=NoArgs, infer_schema=False,
            description="Give the GPU back as soon as the GPU work is done; processes left in the node directory "
            "are stopped. You can ask again later.",
        ),
        StructuredTool.from_function(
            func=ask_sync, coroutine=ask_async, name="ask_main", args_schema=AskArgs, infer_schema=False,
            description="Ask the main agent, who wrote the skeleton, and wait for the answer. For a premise the data "
            "contradicts or a skeleton choice you think is wrong, not for engineering details you can settle. "
            "Without an answer after a while you continue with the skeleton's choice.",
        ),
    ]


def create_main_tools(hub: Hub) -> list[BaseTool]:
    """The main agent's tools."""

    def grant(node: str) -> str:
        try:
            return hub.grant(node.strip())
        except ValueError as e:
            return f"ERROR: {e}."

    def revoke(node: str, reason: str) -> str:
        if not reason.strip():
            return "ERROR: say why in `reason`; the node is told."
        try:
            killed = hub.revoke(node.strip(), reason.strip())
        except ValueError as e:
            return f"ERROR: {e}."
        return (f"GPU taken back from {node}" + (f"; {len(killed)} of its processes were stopped" if killed else "")
                + ". Give it to the next node with gpu_grant.")

    def answer(question_id: str, answer: str) -> str:
        if not answer.strip():
            return "ERROR: the answer is empty."
        try:
            return hub.answer(question_id.strip(), answer.strip())
        except ValueError as e:
            return f"ERROR: {e}."

    return [
        StructuredTool.from_function(
            func=grant, name="gpu_grant", args_schema=GrantArgs, infer_schema=False,
            description="Give the free GPU to an open node, usually one that asked for it (tree_status lists the "
            "requests with their reasons). Refused while another node holds it.",
        ),
        StructuredTool.from_function(
            func=revoke, name="gpu_revoke", args_schema=RevokeArgs, infer_schema=False,
            description="Take the GPU back from the node holding it: its processes in the node directory are "
            "stopped and it is told why. Its agent keeps working and can ask again.",
        ),
        StructuredTool.from_function(
            func=answer, name="answer_node", args_schema=AnswerArgs, infer_schema=False,
            description="Answer a node agent's question (the note names its id). The node is waiting for it.",
        ),
    ]
