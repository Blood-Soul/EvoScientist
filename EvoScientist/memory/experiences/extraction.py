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

# Matches a References/Bibliography heading line on its own (optionally numbered,
# optionally under a Markdown "#" heading) — not the word appearing inside a
# sentence. Jina's PDF-to-Markdown conversion renders it as e.g. "## References".
_REFERENCES_HEADING_RE = re.compile(
    r"^[ \t]*#{0,6}[ \t]*(?:[0-9]+[.)]?[ \t]*)?"
    r"(references|bibliography|works cited)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

PROMPT_FILENAMES = {"l1": "l1_extract.md", "l2": "l2_inductive.md"}


class ExperienceOutputError(ValueError):
    """Raised when extraction output does not match the existing prompt schema."""


_COMMON_LLM_KEYS = {
    "domain",
    "task",
    "statement",
    "applicable_when",
    "not_applicable_when",
    "scope",
    "action",
    "effect",
    "evidence",
}
_L1_LLM_KEYS = _COMMON_LLM_KEYS | {"practice_trace"}
_L2_LLM_KEYS = _COMMON_LLM_KEYS | {"claim_type", "rationale", "rationale_depth"}

# Fields the reuse layer consumes, added after the first ~100 records were
# already extracted. Optional on read so those records stay usable without
# re-extraction: `policy.select.transferable_core()` falls back to the head of
# `statement`, and the policy writer mines bindings out of the prose when the
# structured list is absent.
#
# `transferable_core`: the claim with every paper-specific value removed. Used
#   as the rerank descriptor, where a truncated `statement` opening is a poor
#   signal.
# `bindings`: the source-fixed values themselves, tagged by kind. Gives the
#   writer's `rebind` step a structured input instead of prose mining, and lets
#   the A/B harness count stale-binding hits without an LLM judge.
_OPTIONAL_LLM_KEYS = {"transferable_core", "bindings"}

_BINDING_KINDS = {
    "dataset",
    "model",
    "scale",
    "hyperparam",
    "baseline",
    "metric",
    "toolchain",
    "other",
}


def _validate_bindings(value: Any, *, level: ExperienceLevel) -> None:
    """Validate the optional bindings list when the model supplies it."""
    if not isinstance(value, list):
        raise ExperienceOutputError(f"{level.upper()} bindings must be an array")
    for row in value:
        if not isinstance(row, Mapping):
            raise ExperienceOutputError(
                f"{level.upper()} bindings entries must be objects"
            )
        if not str(row.get("name") or "").strip():
            raise ExperienceOutputError(
                f"{level.upper()} bindings entries need a non-empty name"
            )
        kind = str(row.get("kind") or "").strip().casefold()
        if kind not in _BINDING_KINDS:
            raise ExperienceOutputError(
                f"{level.upper()} binding kind {kind!r} is not recognized"
            )


def _validate_llm_experience(
    item: Mapping[str, Any], *, level: ExperienceLevel
) -> None:
    """Validate only fields the current prompt asks the model to produce."""
    required = _L1_LLM_KEYS if level == "l1" else _L2_LLM_KEYS
    keys = set(item)
    missing = sorted(required - keys)
    extra = sorted(keys - required - _OPTIONAL_LLM_KEYS)
    if missing or extra:
        raise ExperienceOutputError(
            f"{level.upper()} experience keys mismatch; missing={missing}, extra={extra}"
        )
    if "bindings" in item:
        _validate_bindings(item["bindings"], level=level)
    if "transferable_core" in item and not isinstance(item["transferable_core"], str):
        raise ExperienceOutputError(
            f"{level.upper()} transferable_core must be a string"
        )
    for field in ("applicable_when", "not_applicable_when", "evidence"):
        if not isinstance(item[field], list):
            raise ExperienceOutputError(f"{level.upper()} {field} must be an array")
    if level == "l1":
        trace = item["practice_trace"]
        if not isinstance(trace, list) or any(
            not isinstance(pair, Mapping) or set(pair) != {"action", "feedback"}
            for pair in trace
        ):
            raise ExperienceOutputError(
                "L1 practice_trace must be an array of action/feedback objects"
            )
    if any(
        not isinstance(source, Mapping) or set(source) != {"section", "quote"}
        for source in item["evidence"]
    ):
        raise ExperienceOutputError(
            f"{level.upper()} evidence must contain section/quote objects"
        )
    if level == "l2" and item["claim_type"] not in {
        "property",
        "relation",
        "trend",
        "conditional",
    }:
        raise ExperienceOutputError("L2 claim_type is invalid")
    if level == "l2" and item["rationale_depth"] not in {"deep", "shallow", None}:
        raise ExperienceOutputError("L2 rationale_depth is invalid")


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
    return _strip_references_section(text)


def _strip_references_section(text: str) -> str:
    """Drop the References/Bibliography section onward (citation lists carry no
    extractable experience; anything appended after them, e.g. appendices, is
    dropped too since it sits past the body-text boundary we care about)."""
    match = _REFERENCES_HEADING_RE.search(text)
    if not match:
        return text
    # A short prefix means "References" matched inside the abstract/intro (rare
    # false positive, e.g. citing a paper titled "References for X") rather than
    # the real trailing section — keep the full text in that case.
    if match.start() < 500:
        return text
    return text[: match.start()].rstrip()


def _paper_slug(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", paper_id.strip()).strip("_") or "paper"


def _initial_confidence(item: Mapping[str, Any], *, level: ExperienceLevel) -> float:
    """Estimate single-paper evidence completeness; aggregation may revise it."""
    score = 0.45
    evidence = item.get("evidence", [])
    if evidence:
        score += min(0.20, 0.05 * len(evidence))
        if all(
            isinstance(row.get("quote"), str) and len(row["quote"]) >= 150
            for row in evidence
        ):
            score += 0.10
    if level == "l1" and len(item.get("practice_trace", [])) >= 3:
        score += 0.10
    if level == "l2" and item.get("rationale"):
        score += 0.05
    return round(min(score, 0.85), 2)


def _normalize_current_payload(
    payload: dict[str, Any],
    *,
    level: ExperienceLevel,
    paper_id: str,
    domain_arxiv: str | None = None,
) -> dict[str, Any]:
    """Inject deterministic identity, provenance, lifecycle, and confidence fields."""
    experiences = payload["experiences"]
    normalized: list[dict[str, Any]] = []
    slug = _paper_slug(paper_id)
    for index, raw in enumerate(experiences, start=1):
        _validate_llm_experience(raw, level=level)
        item = dict(raw)
        item["id"] = f"{level}_{slug}_{index:02d}"
        item["layer"] = level.upper()
        item["domain_arxiv"] = domain_arxiv
        item["utility"] = None
        item["confidence"] = _initial_confidence(item, level=level)
        item["evidence"] = [
            {"source_id": paper_id, **dict(source)} for source in item["evidence"]
        ]
        normalized.append(item)
    return {"paper_id": paper_id, "experiences": normalized}


def parse_experience_json(
    raw_output: str,
    *,
    level: ExperienceLevel,
    paper_id: str | None = None,
    domain_arxiv: str | None = None,
) -> dict[str, Any]:
    """Parse and normalize current output; retain compatibility with legacy payloads."""
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
    supplied_id = payload.get("paper_id")
    canonical_id = paper_id or (supplied_id if isinstance(supplied_id, str) else "")
    if not canonical_id.strip():
        raise ExperienceOutputError(
            f"{level.upper()} paper_id is required by the caller"
        )
    current = not experiences or any(
        "statement" in item or "domain" in item or "layer" in item
        for item in experiences
    )
    if current:
        return _normalize_current_payload(
            payload,
            level=level,
            paper_id=canonical_id.strip(),
            domain_arxiv=domain_arxiv,
        )
    if supplied_id is not None and supplied_id != canonical_id:
        raise ExperienceOutputError("model paper_id does not match the requested paper")
    return payload


async def run_experience_extraction(
    *,
    paper_id: str,
    paper_text: str,
    prompts: Mapping[ExperienceLevel, str] | None = None,
    model: Any | None = None,
    domain_arxiv: str | None = None,
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
            format_message_content(response).strip(),
            level=level,
            paper_id=paper_id,
            domain_arxiv=domain_arxiv,
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
