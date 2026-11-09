"""Per-call LLM usage ledger.

Set ``EVOSCIENTIST_USAGE_LEDGER`` to a file path and every chat model built by
:func:`EvoScientist.llm.get_chat_model` writes one JSON line per completed (or
failed) LLM call: tokens, cache counters, the model, and enough context to say
where the call came from. The ledger is off unless the variable is set, so
ordinary CLI runs and tests are unaffected.

The alternative -- adding token counters to each call site -- cannot cover
"all of it": the main agent, sub-agent graphs, background tasks, the coach and
the paper-extraction pipeline all reach the provider through the same factory,
and new call sites keep appearing. Attaching one callback to the model object
catches them all, including calls made by middleware or by direct
``model.ainvoke`` outside any graph.

Counting rules:

* ``input_tokens`` / ``output_tokens`` are what the provider reported.
* ``cache_read`` / ``cache_creation`` come from ``usage_metadata``'s
  ``input_token_details`` (Anthropic-style prompt caching). For Anthropic,
  ``input_tokens`` excludes those cached tokens; for OpenAI-compatible
  providers ``input_tokens`` usually already includes ``cache_read``. The raw
  fields are recorded so the reader can apply the right rule per provider.
* ``cache_hit`` is true when the call read at least one cached prompt token,
  which is the per-call answer to "did this request hit the cache?".

A ledger line is never allowed to break the run: every failure while
extracting or writing is swallowed (and logged once at warning level).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

LEDGER_ENV = "EVOSCIENTIST_USAGE_LEDGER"

logger = logging.getLogger(__name__)

# One append-only writer per path, shared by every handler in the process.
_sinks: dict[Path, _Sink] = {}
_sinks_lock = threading.Lock()


class _Sink:
    """Serialises JSON-line writes to one ledger file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._file: Any = None
        self._broken = False

    def write(self, record: dict[str, Any]) -> None:
        if self._broken:
            return
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            logger.warning("usage ledger: cannot serialise a record", exc_info=True)
            return
        with self._lock:
            if self._broken:
                return
            try:
                if self._file is None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self._file = open(self.path, "a", encoding="utf-8")
                self._file.write(line + "\n")
                self._file.flush()
            except OSError:
                # A full or read-only filesystem must not take the agent down.
                self._broken = True
                logger.warning(
                    "usage ledger: cannot write %s", self.path, exc_info=True
                )


def ledger_path() -> Path | None:
    """The configured ledger file, or ``None`` when the ledger is off."""
    raw = os.environ.get(LEDGER_ENV, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _sink(path: Path) -> _Sink:
    with _sinks_lock:
        sink = _sinks.get(path)
        if sink is None:
            sink = _sinks[path] = _Sink(path)
            sink.write(
                {
                    "kind": "ledger_start",
                    "ts": round(time.time(), 3),
                    "pid": os.getpid(),
                    "cwd": os.getcwd(),
                }
            )
        return sink


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _usage_counts(usage: dict[str, Any]) -> dict[str, Any]:
    """Normalise the several shapes providers use for token usage."""
    details = (
        usage.get("input_token_details") or usage.get("prompt_tokens_details") or {}
    )
    output_details = (
        usage.get("output_token_details")
        or usage.get("completion_tokens_details")
        or {}
    )
    input_tokens = _as_int(usage.get("input_tokens") or usage.get("prompt_tokens"))
    output_tokens = _as_int(
        usage.get("output_tokens") or usage.get("completion_tokens")
    )
    cache_read = _as_int(
        details.get("cache_read") if isinstance(details, dict) else None
    ) or _as_int(details.get("cached_tokens") if isinstance(details, dict) else None)
    cache_creation = _as_int(
        details.get("cache_creation") if isinstance(details, dict) else None
    )
    reasoning = _as_int(
        output_details.get("reasoning") if isinstance(output_details, dict) else None
    )
    total = usage.get("total_tokens")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read": cache_read,
        "cache_creation": cache_creation,
        "reasoning_tokens": reasoning,
        "total_tokens": _as_int(total) if total is not None else None,
        "cache_hit": cache_read > 0,
    }


def _usage_from_response(response: Any) -> dict[str, Any] | None:
    """Usage from the first message that carries it, else ``llm_output``."""
    for generations in getattr(response, "generations", None) or []:
        for generation in generations:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None)
            if isinstance(usage, dict) and usage:
                return usage
            info = getattr(generation, "generation_info", None)
            if isinstance(info, dict):
                for key in ("usage", "token_usage"):
                    candidate = info.get(key)
                    if isinstance(candidate, dict) and candidate:
                        return candidate
    output = getattr(response, "llm_output", None)
    if isinstance(output, dict):
        for key in ("token_usage", "usage"):
            candidate = output.get(key)
            if isinstance(candidate, dict) and candidate:
                return candidate
    return None


def _caller_frame(limit: int = 80) -> str:
    """First stack frame outside the installed libraries, as module:function:line.

    The handler is called from LangChain's callback machinery, so without this
    every line would name the same library function. For calls made inside a
    LangGraph graph the frames below the libraries are the graph runner; the
    ``langgraph_node`` metadata recorded alongside is the better attribution
    there, and :func:`summarize_ledger` prefers it.
    """
    frame = sys._getframe(1)
    steps = 0
    while frame is not None and steps < limit:
        filename = frame.f_code.co_filename
        if (
            "/site-packages/" not in filename
            and "/.venv/" not in filename
            and filename != __file__
        ):
            module = frame.f_globals.get("__name__", "")
            return f"{module}:{frame.f_code.co_name}:{frame.f_lineno}"
        frame = frame.f_back
        steps += 1
    return ""


class UsageLedgerHandler(BaseCallbackHandler):
    """Writes one ledger line per LLM call."""

    def __init__(self, path: Path, *, provider: str = "", model: str = "") -> None:
        super().__init__()
        self._sink = _sink(path)
        self._provider = provider
        self._model = model
        self._lock = threading.Lock()
        self._starts: dict[UUID, float] = {}

    # Both sync and async managers call the sync hooks; on_chat_model_start is
    # used by chat models, on_llm_start by plain LLMs.
    def on_chat_model_start(
        self,
        serialized: Any,
        messages: Any,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._mark_start(run_id)

    def on_llm_start(
        self,
        serialized: Any,
        prompts: Any,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._mark_start(run_id)

    def on_llm_end(
        self, response: Any, *, run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        self._record(response=response, run_id=run_id, error=None, kwargs=kwargs)

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        self._record(response=None, run_id=run_id, error=error, kwargs=kwargs)

    def _mark_start(self, run_id: UUID | None) -> None:
        if run_id is None:
            return
        with self._lock:
            self._starts[run_id] = time.time()

    def _record(
        self,
        *,
        response: Any,
        run_id: UUID | None,
        error: BaseException | None,
        kwargs: dict[str, Any],
    ) -> None:
        try:
            record: dict[str, Any] = {
                "kind": "usage",
                "ts": round(time.time(), 3),
                "pid": os.getpid(),
                "provider": self._provider,
                "model": self._model,
                "run_id": str(run_id) if run_id is not None else None,
                "parent_run_id": str(kwargs.get("parent_run_id"))
                if kwargs.get("parent_run_id") is not None
                else None,
                "thread": threading.current_thread().name,
                "source": _caller_frame(),
                # Present on every record, zero when the call failed before a
                # provider response existed, so aggregates can sum blindly.
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read": 0,
                "cache_creation": 0,
                "reasoning_tokens": 0,
                "cache_hit": False,
            }
            metadata = kwargs.get("metadata")
            if isinstance(metadata, dict):
                for key in (
                    "langgraph_node",
                    "langgraph_step",
                    "checkpoint_ns",
                    "ls_model_name",
                    "ls_provider",
                ):
                    value = metadata.get(key)
                    if value is not None and not isinstance(value, (dict, list)):
                        record[key] = value
            tags = kwargs.get("tags")
            if tags:
                record["tags"] = [str(tag) for tag in tags]
            name = kwargs.get("name")
            if name:
                record["name"] = str(name)

            usage = _usage_from_response(response) if response is not None else None
            if usage is not None:
                record.update(_usage_counts(usage))

            if run_id is not None:
                started = self._starts.pop(run_id, None)
                if started is not None:
                    record["latency_ms"] = int((time.time() - started) * 1000)
            if error is not None:
                record["error"] = type(error).__name__
                record["error_message"] = str(error)[:500]
            self._sink.write(record)
        except Exception:  # never let accounting break the run
            logger.warning("usage ledger: failed to record a call", exc_info=True)


def attach_usage_ledger(
    model: Any, *, provider: str = "", model_name: str = ""
) -> None:
    """Attach the ledger handler to *model* when the ledger is enabled.

    Mutates the model's own ``callbacks`` field, so no wrapper object is
    introduced and attribute access on the model keeps working. Models that
    do not expose that field are left alone.
    """
    path = ledger_path()
    if path is None:
        return
    try:
        fields = getattr(type(model), "model_fields", None) or {}
        if "callbacks" not in fields:
            logger.debug(
                "usage ledger: %s has no callbacks field; not instrumented",
                type(model).__name__,
            )
            return
        handler = UsageLedgerHandler(path, provider=provider, model=model_name)
        existing = getattr(model, "callbacks", None)
        model.callbacks = [*existing, handler] if existing else [handler]
    except Exception:  # instrumentation must never prevent model creation
        logger.warning("usage ledger: could not attach to the model", exc_info=True)


def summarize_ledger(path: str | Path, *, top: int = 12) -> dict[str, Any]:
    """Aggregate a ledger file: totals, cache behaviour and cost sources.

    ``source`` is the first non-library stack frame (or, inside a graph, the
    ``langgraph_node`` metadata) that made the call; it answers "where did the
    tokens go" without the caller having to tag every call site.
    """
    records: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("kind") == "usage":
                    records.append(record)
    except OSError:
        return {"available": False, "calls": 0}

    totals = {
        "calls": len(records),
        "errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read": 0,
        "cache_creation": 0,
        "reasoning_tokens": 0,
        "calls_with_cache_hit": 0,
        "latency_ms": 0,
    }
    by_source: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}

    def bucket(table: dict[str, dict[str, int]], key: str) -> dict[str, int]:
        entry = table.setdefault(
            key or "unknown",
            {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read": 0,
                "cache_creation": 0,
                "cache_hits": 0,
            },
        )
        return entry

    for record in records:
        if record.get("error"):
            totals["errors"] += 1
        for field in (
            "input_tokens",
            "output_tokens",
            "cache_read",
            "cache_creation",
            "reasoning_tokens",
            "latency_ms",
        ):
            totals[field] += _as_int(record.get(field))
        if record.get("cache_hit"):
            totals["calls_with_cache_hit"] += 1
        source = record.get("langgraph_node") or record.get("source") or "unknown"
        for table, key in (
            (by_source, str(source)),
            (by_model, str(record.get("model") or "unknown")),
        ):
            entry = bucket(table, key)
            entry["calls"] += 1
            entry["input_tokens"] += _as_int(record.get("input_tokens"))
            entry["output_tokens"] += _as_int(record.get("output_tokens"))
            entry["cache_read"] += _as_int(record.get("cache_read"))
            entry["cache_creation"] += _as_int(record.get("cache_creation"))
            entry["cache_hits"] += 1 if record.get("cache_hit") else 0

    prompt_tokens = (
        totals["input_tokens"] + totals["cache_read"] + totals["cache_creation"]
    )
    summary: dict[str, Any] = {
        "available": True,
        **totals,
        "prompt_tokens_anthropic_style": prompt_tokens,
        "cache_hit_rate": round(totals["calls_with_cache_hit"] / len(records), 4)
        if records
        else 0.0,
        "cache_read_share": round(totals["cache_read"] / prompt_tokens, 4)
        if prompt_tokens
        else 0.0,
        "by_source": dict(
            sorted(by_source.items(), key=lambda item: -(item[1]["calls"]))[:top]
        ),
        "by_model": dict(
            sorted(by_model.items(), key=lambda item: -(item[1]["calls"]))[:top]
        ),
    }
    return summary


__all__ = [
    "LEDGER_ENV",
    "UsageLedgerHandler",
    "attach_usage_ledger",
    "ledger_path",
    "summarize_ledger",
]
