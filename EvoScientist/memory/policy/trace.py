"""Debug-only trace log for the retrieve → rerank → synthesize chain.

Opt-in and deliberately outside the config schema: this exists so a developer
can see what `apply_experience` actually did on a given call -- which records
were recalled, what the reranker kept and why, what the synthesis model wrote
before it was parsed, and the final policy that lands in the agent's tool
result. It has no bearing on production behavior when unset and is expected to
be deleted once prompt iteration on this layer settles.

Enable with ``EVOSCIENTIST_POLICY_TRACE=1``. Every write is best-effort: a
trace failure must never break the pipeline it is observing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENV_ENABLED = "EVOSCIENTIST_POLICY_TRACE"
_ENV_PATH = "EVOSCIENTIST_POLICY_TRACE_PATH"
_DEFAULT_RELATIVE_PATH = "policies/trace.jsonl"


def trace_enabled() -> bool:
    """Read the opt-in switch straight from the environment."""
    raw = os.environ.get(_ENV_ENABLED, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def trace_path(memory_dir: str | Path) -> Path:
    """Resolve where trace lines get appended.

    Defaults to a single file under the memory dir so one ``tail -f`` catches
    every project; override with ``EVOSCIENTIST_POLICY_TRACE_PATH`` to point
    it somewhere else (e.g. out of a shared memory dir).
    """
    override = os.environ.get(_ENV_PATH, "").strip()
    if override:
        return Path(override).expanduser()
    return Path(memory_dir).expanduser() / _DEFAULT_RELATIVE_PATH


def emit_trace(memory_dir: str | Path, event: str, **fields: Any) -> None:
    """Append one JSON line describing ``event``, if tracing is enabled."""
    if not trace_enabled():
        return
    try:
        path = trace_path(memory_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.time(), "event": event, **fields}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    except OSError:
        logger.warning("policy trace write failed", exc_info=True)


async def emit_trace_async(memory_dir: str | Path, event: str, **fields: Any) -> None:
    """Async-safe wrapper around `emit_trace`.

    `emit_trace` does synchronous file IO (``mkdir`` plus an append write).
    Called directly from inside an ``async def`` on the event loop -- as the
    pipeline, rerank, and synthesize call sites do -- that IO blocks the loop
    and trips `langgraph dev`'s blocking-call detector once tracing is turned
    on. The disabled check stays inline so the common case (tracing off) costs
    nothing; only an enabled call is handed to a thread.
    """
    if not trace_enabled():
        return
    await asyncio.to_thread(emit_trace, memory_dir, event, **fields)


__all__ = ["emit_trace", "emit_trace_async", "trace_enabled", "trace_path"]
