"""The reuse step: write a target-bound policy from selected experiences.

This is the operation the QCR study isolates. Retrieval and selection decide
*whether* usable history reaches the actor; this decides whether the actor can
apply it without importing source-side values. Both model calls in the pipeline
(rerank, write) are intermediate work and run on the auxiliary model -- only
the answer the user reads comes from the main model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...utils import format_message_content
from .schema import PolicyOutputError, normalize_policy
from .trace import emit_trace_async

# One record's JSON runs 4-5k characters. Four records plus the prompt stays
# well inside an aux-model context while keeping every field the writer needs
# (bindings, applicable_when, scope, evidence) intact rather than pre-trimmed.
MAX_RECORD_CHARS = 6000


def _render_records(selected: Sequence[Mapping[str, Any]]) -> str:
    """Render selected records as the writer's evidence block."""
    blocks: list[str] = []
    for item in selected:
        record = item.get("record", {})
        text = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
        if len(text) > MAX_RECORD_CHARS:
            text = text[:MAX_RECORD_CHARS] + "\n... [record truncated]"
        blocks.append(text)
    return "\n\n".join(blocks)


def parse_policy_json(raw_output: str) -> dict[str, Any]:
    """Parse and normalize the writer's output, tolerating fenced JSON."""
    candidate = raw_output.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1] if "\n" in candidate else candidate
        candidate = candidate.removesuffix("```").strip()
        candidate = candidate.removeprefix("json").strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as first_error:
        start = candidate.find("{")
        if start < 0:
            raise PolicyOutputError(
                "policy writer did not return a JSON object"
            ) from first_error
        try:
            payload, _ = json.JSONDecoder().raw_decode(candidate[start:])
        except json.JSONDecodeError as exc:
            raise PolicyOutputError(
                f"policy writer returned malformed JSON: {exc.msg}"
            ) from exc
    if not isinstance(payload, dict):
        raise PolicyOutputError("policy writer output must be a JSON object")
    return normalize_policy(payload)


async def synthesize_policy(
    *,
    task: str,
    selected: Sequence[Mapping[str, Any]],
    state: str,
    model: Any,
    prompt: str,
    memory_dir: str | Path | None = None,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Invoke the writer model and return the normalized policy."""
    filled = prompt.format(
        task=task.strip(),
        state=state.strip() or "[no project state provided]",
        records=_render_records(selected),
    )
    response = await model.ainvoke(
        [SystemMessage(content=filled), HumanMessage(content="Write the policy now.")]
    )
    output = format_message_content(response).strip()
    try:
        policy = parse_policy_json(output)
    except PolicyOutputError:
        if memory_dir is not None:
            await emit_trace_async(
                memory_dir,
                "synthesize",
                call_id=call_id,
                task=task.strip(),
                selected_ids=[item.get("id") for item in selected],
                raw_output=output,
                parsed=False,
            )
        raise
    if memory_dir is not None:
        await emit_trace_async(
            memory_dir,
            "synthesize",
            call_id=call_id,
            task=task.strip(),
            selected_ids=[item.get("id") for item in selected],
            raw_output=output,
            parsed=True,
            policy=policy,
        )
    return policy


__all__ = [
    "MAX_RECORD_CHARS",
    "parse_policy_json",
    "synthesize_policy",
]
