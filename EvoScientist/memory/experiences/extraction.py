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

from langchain_core.messages import HumanMessage, SystemMessage

from ...utils import format_message_content
from ..types import ExperienceLevel
from .fulltext import FulltextError, download_fulltext
from .taxonomy import DISCIPLINES, normalize_discipline

# Matches a References/Bibliography heading line on its own (optionally numbered,
# optionally under a Markdown "#" heading) — not the word appearing inside a
# sentence. Jina's PDF-to-Markdown conversion renders it as e.g. "## References".
_REFERENCES_HEADING_RE = re.compile(
    r"^[ \t]*#{0,6}[ \t]*(?:[0-9]+[.)]?[ \t]*)?"
    r"(references|bibliography|works cited)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

PROMPT_FILENAMES = {"l1": "l1_extract.md", "l2": "l2_inductive.md"}

# Not keyed by `ExperienceLevel`: this prompt induces L2 records from a paper's
# own already-extracted L1 records rather than reading `paper_text`, so it
# does not fit the level->prompt map above. Its output is still stored under
# level "l2" (see `run_experience_extraction`); only prompt *selection* needs a
# third key. Loading it is best-effort — see `load_experience_prompts`.
INDUCTION_PROMPT_KEY = "l2_induced"
INDUCTION_PROMPT_FILENAME = "l2_induced_from_l1.md"


class ExperienceOutputError(ValueError):
    """Raised when extraction output does not match the existing prompt schema."""


def _without_lone_surrogates(text: str) -> str:
    """Replace UTF-16 surrogate code points, which UTF-8 cannot encode.

    Paper text scraped from PDFs and model replies can carry them; handing them
    to the chat API fails with "'utf-8' codec can't encode character ...:
    surrogates not allowed", which loses the whole paper.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-8", "replace").decode("utf-8")
    return text


def _clean_surrogates(value: Any) -> Any:
    if isinstance(value, str):
        return _without_lone_surrogates(value)
    if isinstance(value, list):
        return [_clean_surrogates(item) for item in value]
    if isinstance(value, dict):
        return {k: _clean_surrogates(v) for k, v in value.items()}
    return value


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

# Added after ~2000 papers were extracted under the 12-field prompts, so they are
# validated when present but not required: re-extracting the existing corpus is
# not free, and the scorer distinguishes a field a record's prompt never asked
# for from one the model failed to produce. Promote to required only once the
# corpus is uniformly re-extracted.
#
# `trigger_context`: the obstacle in pre-solution wording, indexed so a later
#   agent searching with the problem it faces can reach a record whose other
#   fields are all phrased as conclusions.
# `evidence_scope`: how widely this claim was actually tested in its own paper.
#   Separates "observed once on a toy dataset" from "ablated across five
#   benchmarks" -- the peer-reviewed experimental coverage that substitutes for
#   the rerun this domain cannot perform.
_LATE_ADDED_LLM_KEYS = {"trigger_context", "evidence_scope"}

_VERIFICATION_STRENGTHS = {
    "single-setting",
    "multi-setting",
    "ablated",
    "multi-setting-ablated",
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
# `discipline`: the coarse top-level retrieval facet. Optional because the
#   runtime can derive it from `domain_arxiv`, but the model reading the paper
#   is the better source: `domain_arxiv` is caller-supplied at enqueue time and
#   absent for anything that did not come from arXiv, which is most of the
#   non-CS literature this library is meant to hold.
# `source_l1_ids`: only produced by the L1->L2 induction prompt, never by
#   direct L1/L2 extraction. Marks an L2 record as induced from this paper's
#   own L1 records rather than read from `paper_text` directly.
_OPTIONAL_LLM_KEYS = {"transferable_core", "bindings", "discipline", "source_l1_ids"}

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


def _validate_evidence_scope(value: Any, *, level: ExperienceLevel) -> None:
    """Validate the evidence_scope object when the model supplies it."""
    if not isinstance(value, Mapping):
        raise ExperienceOutputError(f"{level.upper()} evidence_scope must be an object")
    for field in ("datasets", "models"):
        entries = value.get(field)
        if not isinstance(entries, list) or not all(
            isinstance(entry, str) for entry in entries
        ):
            raise ExperienceOutputError(
                f"{level.upper()} evidence_scope {field} must be an array of strings"
            )
    for field in ("has_ablation", "has_baseline_comparison"):
        if not isinstance(value.get(field), bool):
            raise ExperienceOutputError(
                f"{level.upper()} evidence_scope {field} must be a boolean"
            )
    strength = value.get("verification_strength")
    if strength not in _VERIFICATION_STRENGTHS:
        raise ExperienceOutputError(
            f"{level.upper()} evidence_scope verification_strength {strength!r} "
            f"is not one of {', '.join(sorted(_VERIFICATION_STRENGTHS))}"
        )


def _validate_llm_experience(
    item: Mapping[str, Any], *, level: ExperienceLevel
) -> None:
    """Validate only fields the current prompt asks the model to produce."""
    required = _L1_LLM_KEYS if level == "l1" else _L2_LLM_KEYS
    keys = set(item)
    missing = sorted(required - keys)
    extra = sorted(keys - required - _OPTIONAL_LLM_KEYS - _LATE_ADDED_LLM_KEYS)
    if missing or extra:
        raise ExperienceOutputError(
            f"{level.upper()} experience keys mismatch; missing={missing}, extra={extra}"
        )
    if "bindings" in item:
        _validate_bindings(item["bindings"], level=level)
    if "evidence_scope" in item:
        _validate_evidence_scope(item["evidence_scope"], level=level)
    if "trigger_context" in item and not (
        isinstance(item["trigger_context"], str) and item["trigger_context"].strip()
    ):
        raise ExperienceOutputError(
            f"{level.upper()} trigger_context must be a non-empty string"
        )
    if "source_l1_ids" in item:
        ids = item["source_l1_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or not all(isinstance(entry, str) and entry.strip() for entry in ids)
        ):
            raise ExperienceOutputError(
                f"{level.upper()} source_l1_ids must be a non-empty array of strings"
            )
    if "transferable_core" in item and not isinstance(item["transferable_core"], str):
        raise ExperienceOutputError(
            f"{level.upper()} transferable_core must be a string"
        )
    if "discipline" in item:
        # Reject rather than silently normalize: `discipline` is an exact-match
        # retrieval filter, so an out-of-vocabulary value produces a facet no
        # query can name, and a whole record that browsing cannot reach.
        if normalize_discipline(item["discipline"]) is None:
            raise ExperienceOutputError(
                f"{level.upper()} discipline {item['discipline']!r} is not one of "
                f"{', '.join(DISCIPLINES)}"
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
            paths.process_workspace().root / "prompt",
            Path(__file__).resolve().parents[3] / "prompt",
            Path(sys.prefix) / "share" / "evoscientist" / "prompt",
        ]
    )
    return candidates


def load_experience_prompts() -> dict[str, str]:
    """Load the existing L1/L2 prompts without changing their schemas.

    Also loads the L1->L2 induction prompt under `INDUCTION_PROMPT_KEY` when a
    matching file sits alongside l1/l2 in the same directory. That file is
    optional: its absence never raises, since the induction step itself
    degrades gracefully when its prompt is missing (see
    `run_experience_extraction`).
    """
    for directory in _prompt_dir_candidates():
        paths = {level: directory / name for level, name in PROMPT_FILENAMES.items()}
        if all(path.is_file() for path in paths.values()):
            loaded = {
                level: path.read_text(encoding="utf-8") for level, path in paths.items()
            }
            induction_path = directory / INDUCTION_PROMPT_FILENAME
            if induction_path.is_file():
                loaded[INDUCTION_PROMPT_KEY] = induction_path.read_text(
                    encoding="utf-8"
                )
            return loaded
    searched = ", ".join(str(path) for path in _prompt_dir_candidates())
    raise FileNotFoundError(f"Experience prompts not found; searched: {searched}")


async def download_paper_text(url: str, *, timeout: float = 90.0) -> str:
    """Download a paper's full text through the shared acquisition chain.

    Backend order lives in `fulltext.py`: MinerU when `MINERU_API_KEY` is set,
    then the arXiv PDF parsed locally (pymupdf4llm, pymupdf, pypdf), then Jina
    Reader, then a direct text/plain or text/markdown response. Every attempt
    is named in the exception, so a failed fallback is not hidden behind the
    first error the way the Jina-first version hid it.
    """
    try:
        result = await download_fulltext(url, timeout=timeout)
    except FulltextError as exc:
        raise RuntimeError(str(exc)) from exc
    text = result.text.strip()
    if len(text) < 500:
        raise ValueError(
            "Downloaded paper full text is unexpectedly short "
            f"({len(text)} chars via {result.backend})"
        )
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
        if "discipline" in item:
            # Validation accepted case and whitespace variants; store the
            # canonical spelling so the facet value on disk is the one a
            # `discipline=` filter is written against.
            item["discipline"] = normalize_discipline(item["discipline"])
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
        return _clean_surrogates(
            _normalize_current_payload(
                payload,
                level=level,
                paper_id=canonical_id.strip(),
                domain_arxiv=domain_arxiv,
            )
        )
    if supplied_id is not None and supplied_id != canonical_id:
        raise ExperienceOutputError("model paper_id does not match the requested paper")
    return _clean_surrogates(payload)


async def _induce_l2_from_l1(
    *,
    paper_id: str,
    l1_experiences: list[dict[str, Any]],
    prompt: str,
    model: Any,
    domain_arxiv: str | None,
) -> list[dict[str, Any]]:
    """Run the L1->L2 induction prompt and return its normalized L2 records.

    Failures here are logged and swallowed rather than raised: induction is an
    additional, coexisting path (per the approved design) alongside direct L2
    extraction, so it must never take down extraction for a paper that already
    produced valid direct L1/L2 output.
    """
    response = await model.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(
                content=_without_lone_surrogates(
                    f"[paper_id] {paper_id}\n\n"
                    + json.dumps(l1_experiences, ensure_ascii=False)
                )
            ),
        ]
    )
    result = parse_experience_json(
        format_message_content(response).strip(),
        level="l2",
        paper_id=paper_id,
        domain_arxiv=domain_arxiv,
    )
    return result["experiences"]


def _json_was_cut_off(exc: ExperienceOutputError) -> bool:
    """A missing or unterminated JSON object is usually a reply cut off at the output limit."""
    message = str(exc)
    return "malformed JSON" in message or "did not contain a JSON object" in message


_TRUNCATED_RETRY_HINT = (
    "\n\n[Correction] Your previous reply stopped before the JSON object was complete. "
    "Return the same extraction again with fewer or shorter records, so that the whole JSON fits."
)


async def run_experience_extraction(
    *,
    paper_id: str,
    paper_text: str,
    prompts: Mapping[str, str] | None = None,
    model: Any | None = None,
    domain_arxiv: str | None = None,
) -> dict[ExperienceLevel, dict[str, Any]]:
    """Invoke the existing L1 and L2 prompts concurrently, then, when the L1->L2
    induction prompt is available, induce additional L2 records from this
    paper's own L1 output and merge them into the L2 result."""
    if model is None:
        from ...EvoScientist import _ensure_auxiliary_chat_model

        model = _ensure_auxiliary_chat_model()
    loaded = dict(prompts or load_experience_prompts())
    paper_text = _without_lone_surrogates(paper_text)

    async def one(level: ExperienceLevel) -> dict[str, Any]:
        base = f"[paper_id] {paper_id}\n\n{paper_text}"
        note = ""
        while True:  # at most two passes: a reply cut off at the output limit is worth one retry
            response = await model.ainvoke(
                [SystemMessage(content=loaded[level]), HumanMessage(content=base + note)]
            )
            try:
                return parse_experience_json(
                    format_message_content(response).strip(),
                    level=level,
                    paper_id=paper_id,
                    domain_arxiv=domain_arxiv,
                )
            except ExperienceOutputError as exc:
                if note or not _json_was_cut_off(exc):
                    raise
                note = _TRUNCATED_RETRY_HINT

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
    output = {
        level: result
        for level, result in zip(levels, results, strict=True)
        if isinstance(result, dict)
    }

    induction_prompt = loaded.get(INDUCTION_PROMPT_KEY)
    l1_experiences = output.get("l1", {}).get("experiences", [])
    if induction_prompt and l1_experiences:
        try:
            induced = await _induce_l2_from_l1(
                paper_id=paper_id,
                l1_experiences=l1_experiences,
                prompt=induction_prompt,
                model=model,
                domain_arxiv=domain_arxiv,
            )
        except Exception as exc:  # degrade, never block direct output
            print(
                f"[extraction] L1->L2 induction skipped for {paper_id!r}: {exc}",
                file=sys.stderr,
            )
        else:
            output["l2"]["experiences"].extend(induced)

    return output
