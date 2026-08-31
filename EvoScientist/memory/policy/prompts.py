"""Prompt loading for the reuse-policy pipeline.

Mirrors ``experiences.extraction.load_experience_prompts``: prompts live as
editable Markdown under ``prompt/`` rather than as string literals, so the
reuse behaviour can be tuned without a code change. The search order matches
the experience prompts so a packaged install resolves both the same way.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

RERANK_PROMPT_FILENAME = "policy_rerank.md"
WRITER_PROMPT_FILENAME = "policy_write.md"


def _prompt_dir_candidates() -> list[Path]:
    from ... import paths

    candidates: list[Path] = []
    configured = os.environ.get("EVOSCIENTIST_POLICY_PROMPT_DIR", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            paths.WORKSPACE_ROOT / "prompt",
            Path(__file__).resolve().parents[3] / "prompt",
            Path(sys.prefix) / "share" / "evoscientist" / "prompt",
        ]
    )
    return candidates


def load_policy_prompt(filename: str) -> str:
    """Load one policy prompt file by name."""
    searched = _prompt_dir_candidates()
    for directory in searched:
        candidate = directory / filename
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    joined = ", ".join(str(path) for path in searched)
    raise FileNotFoundError(f"Policy prompt {filename} not found; searched: {joined}")


async def load_rerank_prompt() -> str:
    """Load the candidate-reranker prompt off the event loop."""
    return await asyncio.to_thread(load_policy_prompt, RERANK_PROMPT_FILENAME)


async def load_writer_prompt() -> str:
    """Load the policy-writer prompt off the event loop."""
    return await asyncio.to_thread(load_policy_prompt, WRITER_PROMPT_FILENAME)


__all__ = [
    "RERANK_PROMPT_FILENAME",
    "WRITER_PROMPT_FILENAME",
    "load_policy_prompt",
    "load_rerank_prompt",
    "load_writer_prompt",
]
