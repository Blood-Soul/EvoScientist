"""Prompt loading for plan review.

Mirrors ``experiences.extraction.load_experience_prompts``: prompts live as
editable Markdown under ``prompt/`` rather than as string literals, so the
reuse behaviour can be tuned without a code change. The search order matches
the experience prompts so a packaged install resolves both the same way.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _prompt_dir_candidates() -> list[Path]:
    from ... import paths

    candidates: list[Path] = []
    configured = os.environ.get("EVOSCIENTIST_POLICY_PROMPT_DIR", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            paths.process_workspace().root / "prompt",
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


__all__ = ["load_policy_prompt"]
