"""Paper download and prompt-driven L1/L2 experience extraction."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from collections.abc import Collection, Mapping, Sequence
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


V2_PROMPT_FILENAME = "experience_extract_v2.md"
V2_L2_PROMPT_FILENAME = "l2_extract_v2.md"
V2_INDUCED_PROMPT_FILENAME = "l2_from_l1_v2.md"
V2_INDUCED_PROMPT_KEY = "l2_from_l1"

# The loaded prompts are a plain `Mapping[str, str]` because the store hashes one entry per level.
# This extra entry tags the schema so `run_experience_extraction` can tell the v2 trio from the legacy
# L1/L2 pair without inspecting prompt text; it is never hashed because it is not a level.
V2_SCHEMA_KEY = "schema"
V2_SCHEMA_MARKER = "v2"

_V2_MAX_PAPER_CHARS = 200_000


def load_experience_prompts() -> dict[str, str]:
    """Load the v2 prompts: L1 (facts from tested practices), L2 (mechanism-level findings) and induction.

    The three have separate jobs. L1 reads a paper's controlled comparisons; L2 reads the whole paper for the
    interpretive claims around them; the induction prompt reads this paper's own L1 output and induces findings that
    need more than one record. The store writes one level file per level, so the induction prompt is kept under its
    own key and merged into L2 before storing.

    A prompts mapping without an ``l2`` entry still selects the single-level path in
    `run_experience_extraction`; the older v1 L1/L2 pair is `load_legacy_experience_prompts`.
    """
    for directory in _prompt_dir_candidates():
        l1, l2 = directory / V2_PROMPT_FILENAME, directory / V2_L2_PROMPT_FILENAME
        if l1.is_file() and l2.is_file():
            loaded = {
                V2_SCHEMA_KEY: V2_SCHEMA_MARKER,
                "l1": l1.read_text(encoding="utf-8"),
                "l2": l2.read_text(encoding="utf-8"),
            }
            induction = directory / V2_INDUCED_PROMPT_FILENAME
            if induction.is_file():
                loaded[V2_INDUCED_PROMPT_KEY] = induction.read_text(encoding="utf-8")
            return loaded
    searched = ", ".join(str(path) for path in _prompt_dir_candidates())
    raise FileNotFoundError(
        f"Experience prompts {V2_PROMPT_FILENAME} + {V2_L2_PROMPT_FILENAME} not found; searched: {searched}"
    )


def load_legacy_experience_prompts() -> dict[str, str]:
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


# ---------------------------------------------------------------- single-level records

_V2_L1_KEYS = (
    "domain", "task", "problem", "method", "requires", "source_setting", "limits_stated",
    "action", "effect", "comparison", "locator",
)
# L2 keeps the L1 fields whose job is the same (domain/task/method/requires/limits_stated/action) and replaces the
# measured outcome with the claim's three supports: the options that were compared, the mechanism, and the provenance
# (a locator in the paper, or the L1 ids the induction drew on). There is no source_setting: an L2 finding is stated
# as a condition (`requires`), not as a value from the source experiment.
_V2_L2_KEYS = (
    "domain", "task", "problem", "method", "action", "compared_options", "mechanism",
    "requires", "limits_stated", "origin", "locator", "source_l1_ids",
)
_V2_COMPARISONS = {"controlled", "vs_baseline", "absolute_only"}
_V2_PARAM_BASES = {"swept", "derived", "prior_work", "default", "unstated"}
_V2_REQUIRES_BASES = {"stated", "inferred"}
_V2_ORIGINS = {"paper", "induced"}
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _collapse_numbers(text: str) -> str:
    """One spelling for numbers in a PDF conversion: '0 .2' -> '0.2', '36,000' -> '36000'."""
    text = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", text)
    return re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", text)


def _ungrounded_numbers(claimed: Any, source_text: str) -> tuple[int, int]:
    """(numbers in the claimed text that the source never writes, numbers checked)."""
    if not source_text:
        return 0, 0
    claimed_text = claimed if isinstance(claimed, str) else json.dumps(claimed, ensure_ascii=False)
    wanted = {n for n in _NUMBER_RE.findall(_collapse_numbers(claimed_text)) if "." in n or len(n) >= 2}
    if not wanted:
        return 0, 0
    have = set(_NUMBER_RE.findall(_collapse_numbers(source_text)))
    return len(wanted - have), len(wanted)


def _ungrounded_share(record: Mapping[str, Any], paper_text: str) -> tuple[int, int]:
    """L1 grounding: the numbers in `effect` and the knob values must be the paper's own."""
    return _ungrounded_numbers(
        [record.get("effect"), (record.get("source_setting") or {}).get("params")], paper_text
    )


def _ungrounded_share_l2(record: Mapping[str, Any], source_text: str) -> tuple[int, int]:
    """L2 grounding: numbers in the compared options and the claim must come from the paper or the cited L1 records."""
    return _ungrounded_numbers([record.get("compared_options"), record.get("action")], source_text)


def _validate_conditions(value: Any, field: str) -> None:
    for item in value if isinstance(value, list) else [None]:
        if (
            not isinstance(item, Mapping)
            or item.get("basis") not in _V2_REQUIRES_BASES
            or not str(item.get("text") or "").strip()
        ):
            raise ExperienceOutputError(f"{field} entries need text and a basis of stated or inferred")


def _validate_v2_l1_record(item: Mapping[str, Any]) -> None:
    missing = [k for k in _V2_L1_KEYS if k not in item]
    if missing:
        raise ExperienceOutputError(f"record keys missing: {missing}")
    for key in ("domain", "task", "problem", "action", "locator"):
        if not isinstance(item[key], str) or not item[key].strip():
            raise ExperienceOutputError(f"{key} must be a non-empty string")
    method = item["method"]
    if not isinstance(method, list) or not method or not all(isinstance(m, str) and m.strip() for m in method):
        raise ExperienceOutputError("method must be a non-empty list of strings")
    if item["comparison"] not in _V2_COMPARISONS:
        raise ExperienceOutputError(f"comparison {item['comparison']!r} is not one of {sorted(_V2_COMPARISONS)}")
    effect = item["effect"]
    if not isinstance(effect, Mapping) or not str(effect.get("compared") or "").strip():
        raise ExperienceOutputError("effect.compared is required")
    _validate_conditions(item["requires"], "requires")
    setting = item["source_setting"]
    if not isinstance(setting, Mapping) or not isinstance(setting.get("environment"), list) or not isinstance(setting.get("params"), list):
        raise ExperienceOutputError("source_setting needs environment and params lists")
    for param in setting["params"]:
        if not isinstance(param, Mapping) or param.get("basis") not in _V2_PARAM_BASES or not str(param.get("name") or "").strip():
            raise ExperienceOutputError("params entries need a name and a basis from the allowed set")
    if not isinstance(item["limits_stated"], list):
        raise ExperienceOutputError("limits_stated must be a list")


def _validate_v2_l2_record(
    item: Mapping[str, Any], *, source_ids: Collection[str] | None = None
) -> None:
    """Validate one L2 record; `source_ids` is the set of `l1-*` labels the induction was given."""
    missing = [k for k in _V2_L2_KEYS if k not in item]
    if missing:
        raise ExperienceOutputError(f"record keys missing: {missing}")
    for key in ("domain", "task", "problem", "action"):
        if not isinstance(item[key], str) or not item[key].strip():
            raise ExperienceOutputError(f"{key} must be a non-empty string")
    method = item["method"]
    if not isinstance(method, list) or not method or not all(isinstance(m, str) and m.strip() for m in method):
        raise ExperienceOutputError("method must be a non-empty list of strings")
    if not isinstance(item["mechanism"], str):
        raise ExperienceOutputError("mechanism must be a string (empty when the paper gives no reason)")
    options = item["compared_options"]
    if not isinstance(options, list):
        raise ExperienceOutputError("compared_options must be a list")
    for option in options:
        if (
            not isinstance(option, Mapping)
            or not str(option.get("option") or "").strip()
            or not str(option.get("result") or "").strip()
        ):
            raise ExperienceOutputError("compared_options entries need a non-empty option and result")
    _validate_conditions(item["requires"], "requires")
    _validate_conditions(item["limits_stated"], "limits_stated")
    origin = item["origin"]
    if origin not in _V2_ORIGINS:
        raise ExperienceOutputError(f"origin {origin!r} is not one of {sorted(_V2_ORIGINS)}")
    if origin == "paper":
        if not isinstance(item["locator"], str) or not item["locator"].strip():
            raise ExperienceOutputError("origin=paper needs a locator")
    else:
        cited = item["source_l1_ids"]
        if not isinstance(cited, list) or not cited or not all(isinstance(c, str) and c.strip() for c in cited):
            raise ExperienceOutputError("origin=induced needs non-empty source_l1_ids")
        if source_ids is not None:
            unknown = [c for c in cited if c not in source_ids]
            if unknown:
                raise ExperienceOutputError(f"source_l1_ids name records that were not given: {unknown}")
        if isinstance(item.get("locator"), str) and item["locator"].strip():
            # An induced record has no page of its own; a locator here would point at the paper it never read.
            item["locator"] = ""


def parse_experience_v2(
    raw_output: str,
    *,
    paper_id: str,
    paper_text: str = "",
    domain_arxiv: str | None = None,
    level: str = "l1",
    source_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    """Parse a v2 extraction reply into the payload that is stored.

    `level` selects the record shape: ``l1`` (a tested practice and its measured outcome) or ``l2`` (a mechanism-level
    finding). `source_ids` is the set of ``l1-*`` labels an induction reply may cite; it is ``None`` for the two
    paper-reading prompts, which have no L1 records to point at.

    Records that break the shape, or whose numbers mostly never occur in the source text, are dropped one by one so a
    single bad record does not cost the paper; if the reply held records and none survived, that is an error and the
    caller retries.
    """
    candidate = raw_output.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    start = candidate.find("{")
    if start < 0:
        raise ExperienceOutputError("extraction did not contain a JSON object")
    try:
        payload, _ = json.JSONDecoder().raw_decode(candidate[start:])
    except json.JSONDecodeError as exc:
        raise ExperienceOutputError(f"extraction returned malformed JSON: {exc.msg}") from exc
    rows = payload.get("experiences") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ExperienceOutputError("output must contain an 'experiences' list")
    kept: list[dict[str, Any]] = []
    first_error: ExperienceOutputError | None = None
    for row in rows:
        try:
            if not isinstance(row, dict):
                raise ExperienceOutputError("record is not an object")
            if level == "l2":
                row.setdefault("compared_options", [])
                row.setdefault("mechanism", "")
                row.setdefault("requires", [])
                row.setdefault("limits_stated", [])
                row.setdefault("locator", "")
                row.setdefault("source_l1_ids", [])
                _validate_v2_l2_record(row, source_ids=source_ids)
                if paper_text:
                    bad, total = _ungrounded_share_l2(row, paper_text)
                    if total >= 3 and bad / total > 0.5:
                        raise ExperienceOutputError(f"{bad} of {total} numbers never occur in the source")
            else:
                row.setdefault("limits_stated", [])
                _validate_v2_l1_record(row)
                if paper_text:
                    bad, total = _ungrounded_share(row, paper_text)
                    if total >= 3 and bad / total > 0.5:
                        raise ExperienceOutputError(f"{bad} of {total} numbers never occur in the paper")
        except ExperienceOutputError as exc:
            first_error = first_error or exc
            continue
        row["layer"] = level
        kept.append(row)
    if rows and not kept:
        raise ExperienceOutputError(f"no record survived validation: {first_error}")
    return _clean_surrogates({"paper_id": paper_id, "domain_arxiv": domain_arxiv, "experiences": kept})


def _induction_input(l1_experiences: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Label each L1 record with a local id so an induced claim can cite the records it drew on.

    The labels are positional (``l1-1`` …) and are rewritten to the stored ``E-*`` ids when the paper is saved, so
    nothing downstream has to guess which record a citation meant.
    """
    labelled, ids = [], []
    for index, record in enumerate(l1_experiences, start=1):
        label = f"l1-{index}"
        ids.append(label)
        labelled.append({**record, "id": label})
    return labelled, ids


def _record_tokens(text: Any) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(w) > 2}


def _same_options(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """True when two findings compare (almost) the same set of options.

    Wording varies enough that comparing `problem`/`action` text misses near-duplicates (measured on a trial: the pair
    restating one ablation in different words scored below 0.25 on combined problem+action). The option names do not
    vary that much, because both prompts copy the paper's names for the variants, so they carry the signal.
    """

    def names(record: Mapping[str, Any]) -> set[str]:
        options = record.get("compared_options")
        if not isinstance(options, list):
            return set()
        return _record_tokens(" ".join(str(o.get("option") or "") for o in options if isinstance(o, Mapping)))

    left_names, right_names = names(left), names(right)
    if not left_names or not right_names:
        return False
    return len(left_names & right_names) / len(left_names | right_names) >= 0.45


def _merge_induced(direct: list[dict[str, Any]], induced: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep both origins, but drop an induced record that restates a directly extracted one.

    The two prompts work on different material and are meant to coexist. Duplicates still arise when the paper states a
    finding and its own L1 records make the same claim. The direct record wins: it carries a locator, and an induced
    record can only restate what the L1 records already say, so dropping it loses no information.
    """
    def key(record: Mapping[str, Any]) -> str:
        return " ".join(str(record.get("problem") or "").lower().split())

    kept = list(direct)
    seen = {key(record) for record in direct if key(record)}
    for record in induced:
        record_key = key(record)
        if record_key and record_key in seen:
            continue
        if any(_same_options(record, existing) for existing in kept):
            continue
        seen.add(record_key)
        kept.append(record)
    return kept


async def _run_single_level_extraction(
    *, paper_id: str, paper_text: str, prompt: str, model: Any, domain_arxiv: str | None
) -> dict[str, dict[str, Any]]:
    text = _without_lone_surrogates(paper_text)[:_V2_MAX_PAPER_CHARS]
    base = f"[paper_id] {paper_id}\n\n{text}"
    note = ""
    last: Exception | None = None
    for _attempt in range(3):
        response = await model.ainvoke([SystemMessage(content=prompt), HumanMessage(content=base + note)])
        try:
            payload = parse_experience_v2(
                format_message_content(response).strip(), paper_id=paper_id, paper_text=text, domain_arxiv=domain_arxiv
            )
            return {"l1": payload}
        except ExperienceOutputError as exc:
            last = exc
            note = _TRUNCATED_RETRY_HINT if _json_was_cut_off(exc) else (
                f"\n\n[Correction] Your previous reply was rejected: {exc}. Return the extraction again in the exact shape given."
            )
    raise ExperienceOutputError(f"Experience extraction failed: {last}")


async def _invoke_v2(
    *,
    prompt: str,
    human: str,
    level: str,
    paper_id: str,
    paper_text: str,
    model: Any,
    domain_arxiv: str | None,
    source_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    """Run one v2 prompt, retrying a rejected reply up to three times with the reason fed back."""
    note = ""
    last: Exception | None = None
    for _attempt in range(3):
        response = await model.ainvoke([SystemMessage(content=prompt), HumanMessage(content=human + note)])
        try:
            return parse_experience_v2(
                format_message_content(response).strip(),
                paper_id=paper_id,
                paper_text=paper_text,
                domain_arxiv=domain_arxiv,
                level=level,
                source_ids=source_ids,
            )
        except ExperienceOutputError as exc:
            last = exc
            note = _TRUNCATED_RETRY_HINT if _json_was_cut_off(exc) else (
                f"\n\n[Correction] Your previous reply was rejected: {exc}. Return the extraction again in the exact shape given."
            )
    raise ExperienceOutputError(f"{level.upper()} extraction failed: {last}")


async def _induce_v2(
    *,
    paper_id: str,
    l1_experiences: Sequence[Mapping[str, Any]],
    prompt: str,
    model: Any,
    domain_arxiv: str | None,
) -> list[dict[str, Any]]:
    labelled, source_ids = _induction_input(l1_experiences)
    source_text = json.dumps(labelled, ensure_ascii=False)
    payload = await _invoke_v2(
        prompt=prompt,
        human=f"[paper_id] {paper_id}\n\n{source_text}",
        level="l2",
        paper_id=paper_id,
        paper_text=source_text,
        model=model,
        domain_arxiv=domain_arxiv,
        source_ids=source_ids,
    )
    return payload["experiences"]


async def _run_v2_pipeline(
    *,
    paper_id: str,
    paper_text: str,
    prompts: Mapping[str, str],
    model: Any,
    domain_arxiv: str | None,
) -> dict[str, dict[str, Any]]:
    """L1 and L2 read the paper in parallel, then induction adds findings that need several L1 records.

    L1 is the backbone: if it fails, the paper fails and the worker retries. L2 is allowed to degrade to an empty list
    and is logged loudly when it does — a mechanism-level finding is additional value, not the reason the paper was
    queued. Induction always degrades: it depends on L1 succeeding first.
    """
    text = _without_lone_surrogates(paper_text)[:_V2_MAX_PAPER_CHARS]
    human = f"[paper_id] {paper_id}\n\n{text}"

    async def read(level: ExperienceLevel) -> dict[str, Any]:
        return await _invoke_v2(
            prompt=prompts[level],
            human=human,
            level=level,
            paper_id=paper_id,
            paper_text=text,
            model=model,
            domain_arxiv=domain_arxiv,
        )

    l1_result, l2_result = await asyncio.gather(
        read("l1"), read("l2"), return_exceptions=True
    )
    if isinstance(l1_result, Exception):
        raise ExperienceOutputError(f"L1 extraction failed: {l1_result}")
    if isinstance(l2_result, Exception):
        print(
            f"[extraction] L2 (paper) skipped for {paper_id!r}: {l2_result}",
            file=sys.stderr,
        )
        l2_result = {"paper_id": paper_id, "domain_arxiv": domain_arxiv, "experiences": []}

    l1_payload: dict[str, Any] = l1_result
    l2_payload: dict[str, Any] = l2_result
    induction_prompt = prompts.get(V2_INDUCED_PROMPT_KEY)
    if induction_prompt and l1_payload["experiences"]:
        try:
            induced = await _induce_v2(
                paper_id=paper_id,
                l1_experiences=l1_payload["experiences"],
                prompt=induction_prompt,
                model=model,
                domain_arxiv=domain_arxiv,
            )
        except Exception as exc:  # degrade, never block a paper whose L1 is good
            print(
                f"[extraction] L1->L2 induction skipped for {paper_id!r}: {exc}",
                file=sys.stderr,
            )
        else:
            l2_payload["experiences"] = _merge_induced(l2_payload["experiences"], induced)

    return {"l1": l1_payload, "l2": l2_payload}


async def run_experience_extraction(
    *,
    paper_id: str,
    paper_text: str,
    prompts: Mapping[str, str] | None = None,
    model: Any | None = None,
    domain_arxiv: str | None = None,
) -> dict[ExperienceLevel, dict[str, Any]]:
    """Run the configured extraction.

    Three shapes are supported, selected by the prompts that were loaded:

    * v2 pair (current): ``l1`` + ``l2`` + the L1->L2 induction prompt. L1 records tested practices, L2 records
      mechanism-level findings read from the whole paper, and induction adds findings that span several L1 records.
    * single level (r30 fallback): only ``l1``; the old comparison-centred schema, no L2.
    * v1 pair (legacy): the original L1/L2/induction prompts, kept so old libraries can be re-checked.
    """
    if model is None:
        from ...EvoScientist import _ensure_auxiliary_chat_model

        model = _ensure_auxiliary_chat_model()
    loaded = dict(prompts or load_experience_prompts())
    if "l2" not in loaded:
        return await _run_single_level_extraction(
            paper_id=paper_id, paper_text=paper_text, prompt=loaded["l1"], model=model, domain_arxiv=domain_arxiv
        )
    if loaded.get(V2_SCHEMA_KEY) == V2_SCHEMA_MARKER:
        return await _run_v2_pipeline(
            paper_id=paper_id, paper_text=paper_text, prompts=loaded, model=model, domain_arxiv=domain_arxiv
        )
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
