"""Paper download and prompt-driven L1/L2 experience extraction."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from ...utils import format_message_content
from ..types import ExperienceLevel

PROMPT_FILENAMES = {"l1": "l1_extract.md", "l2": "l2_inductive.md"}


class ExperienceOutputError(ValueError):
    """Raised when extraction output does not match the existing prompt schema."""


def _prompt_dir_candidates() -> list[Path]:
    from ... import paths

    candidates: list[Path] = []
    configured = os.environ.get("EVOSCIENTIST_EXPERIENCE_PROMPT_DIR", "").strip()
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


def load_experience_prompts() -> dict[ExperienceLevel, str]:
    """Load the existing L1/L2 prompts without changing their schemas."""
    for directory in _prompt_dir_candidates():
        paths = {level: directory / name for level, name in PROMPT_FILENAMES.items()}
        if all(path.is_file() for path in paths.values()):
            return {
                level: path.read_text(encoding="utf-8") for level, path in paths.items()
            }
    searched = ", ".join(str(path) for path in _prompt_dir_candidates())
    raise FileNotFoundError(f"Experience prompts not found; searched: {searched}")


async def download_paper_text(url: str, *, timeout: float = 90.0) -> str:
    """Download full text through Jina Reader, with a direct-text fallback."""
    headers = {
        "User-Agent": "EvoScientist/0.2 paper-experience-worker",
        "Accept": "text/markdown,text/plain;q=0.9,*/*;q=0.1",
    }
    jina_key = os.environ.get("JINA_API_KEY", "").strip()
    jina_headers = {
        **headers,
        **({"Authorization": f"Bearer {jina_key}"} if jina_key else {}),
    }
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, headers=headers
    ) as client:
        reader_url = (
            url if url.startswith("https://r.jina.ai/") else f"https://r.jina.ai/{url}"
        )
        try:
            reader = await client.get(reader_url, headers=jina_headers)
            reader.raise_for_status()
            text = reader.text.strip()
        except httpx.HTTPError as reader_error:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").casefold()
            if "text/plain" not in content_type and "text/markdown" not in content_type:
                raise reader_error
            text = response.text.strip()
    if len(text) < 500:
        raise ValueError("Downloaded paper full text is unexpectedly short")
    return text


def parse_experience_json(raw_output: str, *, level: ExperienceLevel) -> dict[str, Any]:
    """Parse plain or fenced JSON and enforce only the prompt's top-level shape."""
    candidate = raw_output.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.IGNORECASE | re.DOTALL
    )
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as first_error:
        start = candidate.find("{")
        if start < 0:
            raise ExperienceOutputError(
                f"{level.upper()} extraction did not contain a JSON object"
            ) from first_error
        try:
            payload, _ = json.JSONDecoder().raw_decode(candidate[start:])
        except json.JSONDecodeError as exc:
            raise ExperienceOutputError(
                f"{level.upper()} extraction returned malformed JSON: {exc.msg}"
            ) from exc
    if not isinstance(payload, dict):
        raise ExperienceOutputError(f"{level.upper()} output must be a JSON object")
    experiences = payload.get("experiences")
    if not isinstance(experiences, list) or not all(
        isinstance(row, dict) for row in experiences
    ):
        raise ExperienceOutputError(
            f"{level.upper()} output must contain an 'experiences' object list"
        )
    if not isinstance(payload.get("paper_id"), str) or not payload["paper_id"].strip():
        raise ExperienceOutputError(
            f"{level.upper()} output must contain a non-empty 'paper_id'"
        )
    return payload


async def run_experience_extraction(
    *,
    paper_id: str,
    paper_text: str,
    prompts: Mapping[ExperienceLevel, str] | None = None,
    model: Any | None = None,
) -> dict[ExperienceLevel, dict[str, Any]]:
    """Invoke the existing L1 and L2 prompts concurrently."""
    if model is None:
        from ...EvoScientist import _ensure_auxiliary_chat_model

        model = _ensure_auxiliary_chat_model()
    loaded = dict(prompts or load_experience_prompts())

    async def one(level: ExperienceLevel) -> dict[str, Any]:
        response = await model.ainvoke(
            [
                SystemMessage(content=loaded[level]),
                HumanMessage(content=f"[paper_id] {paper_id}\n\n{paper_text}"),
            ]
        )
        return parse_experience_json(
            format_message_content(response).strip(), level=level
        )

    levels: tuple[ExperienceLevel, ...] = ("l1", "l2")
    results = await asyncio.gather(
        *(one(level) for level in levels), return_exceptions=True
    )
    failures = [
        f"{level.upper()}: {result}"
        for level, result in zip(levels, results, strict=True)
        if isinstance(result, Exception)
    ]
    if failures:
        raise ExperienceOutputError(
            "Experience extraction failed (" + "; ".join(failures) + ")"
        )
    return {
        level: result
        for level, result in zip(levels, results, strict=True)
        if isinstance(result, dict)
    }
