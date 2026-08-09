"""Session-scoped L1/L2 experience extraction for full-text papers.

The extractor is exposed as a runtime-aware LangChain tool so it can obtain the
current LangGraph thread ID without asking the model to copy session metadata
into a shell command.  Parsed JSON is persisted under EvoMemory's root in a
dedicated, session-isolated namespace; only readable Markdown is returned to
the agent context.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import unquote, urlparse

from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field

from ..utils import format_message_content

_PROMPT_FILENAMES = {"l1": "l1_extract.md", "l2": "l2_inductive.md"}
_STORE_VERSION = 1
ExperienceLevel = Literal["l1", "l2"]
_EXTRACTION_LOCKS: dict[str, asyncio.Lock] = {}
_EXTRACTION_LOCKS_GUARD = threading.Lock()


class PaperExperienceArgs(BaseModel):
    """Model-facing arguments for the paper experience extraction tool."""

    paper_file: str = Field(
        min_length=1,
        description=(
            "Workspace-relative Markdown path printed by paper-navigator's "
            "fetch_paper.py after it saves the full paper."
        ),
    )
    paper_id: str = Field(
        min_length=1,
        description=(
            "Canonical paper identifier from paper-navigator, such as an arXiv "
            "ID, DOI, Semantic Scholar ID, or CorpusId."
        ),
    )
    runtime: Annotated[object | None, InjectedToolArg] = None


class PaperExperienceBatchItem(BaseModel):
    """One final paper to enrich in a batch."""

    paper_file: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)


class PaperExperienceBatchArgs(BaseModel):
    """Arguments for concurrent enrichment of the final ranked papers."""

    papers: list[PaperExperienceBatchItem] = Field(min_length=1)
    max_concurrency: int | None = Field(default=None, ge=1, le=64)
    runtime: Annotated[object | None, InjectedToolArg] = None


class ExperienceOutputError(ValueError):
    """Raised when an extraction model does not return the required JSON shape."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _strip_arxiv_version(value: str) -> str:
    return re.sub(r"v\d+$", "", value, flags=re.IGNORECASE)


def paper_identifier(value: str) -> str:
    """Normalize a URL or common scholarly ID for prompts and cache metadata."""
    original = value.strip()
    parsed = urlparse(original)
    hostname = (parsed.hostname or "").lower()
    if hostname.endswith("arxiv.org"):
        arxiv_match = re.match(r"/(?:abs|pdf|html)/(.+?)(?:\.pdf)?$", parsed.path)
        if arxiv_match:
            return _strip_arxiv_version(unquote(arxiv_match.group(1)))

    doi_match = re.search(
        r"(?:doi\.org/|/doi/(?:abs/|full/)?)(10\.[^?#]+)",
        original,
        flags=re.IGNORECASE,
    )
    if doi_match:
        return unquote(doi_match.group(1)).rstrip("/")

    arxiv_match = re.fullmatch(
        r"(?:arxiv\s*:\s*)?((?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?)",
        original,
        flags=re.IGNORECASE,
    )
    if arxiv_match:
        return _strip_arxiv_version(arxiv_match.group(1))

    prefixed_doi = re.fullmatch(r"doi\s*:\s*(10\..+)", original, re.IGNORECASE)
    if prefixed_doi:
        return prefixed_doi.group(1).rstrip("/")
    return original


def _canonical_paper_key(paper_id: str) -> str:
    normalized = paper_identifier(paper_id).strip()
    lowered = normalized.lower()
    if lowered.startswith("10."):
        return f"doi:{lowered}"
    if re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})", lowered):
        return f"arxiv:{lowered}"
    if lowered.startswith("arxiv:"):
        return f"arxiv:{_strip_arxiv_version(lowered.removeprefix('arxiv:'))}"
    return lowered


def _safe_storage_segment(value: str, *, prefix: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._-") or prefix
    digest = _sha256_text(value)[:12]
    return f"{slug[:48]}-{digest}"


def _prompt_dir_candidates() -> list[Path]:
    from .. import paths

    candidates: list[Path] = []
    configured = os.environ.get("EVOSCIENTIST_EXPERIENCE_PROMPT_DIR", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            paths.WORKSPACE_ROOT / "prompt",
            Path(__file__).resolve().parents[2] / "prompt",
            Path(__file__).resolve().parents[1] / "prompt",
            Path(sys.prefix) / "share" / "evoscientist" / "prompt",
        ]
    )
    return candidates


def load_experience_prompts() -> dict[ExperienceLevel, str]:
    """Load the L1 and L2 extraction prompts from the configured prompt folder."""
    for prompt_dir in _prompt_dir_candidates():
        prompt_paths = {
            level: prompt_dir / filename
            for level, filename in _PROMPT_FILENAMES.items()
        }
        if all(path.is_file() for path in prompt_paths.values()):
            return {
                level: path.read_text(encoding="utf-8")
                for level, path in prompt_paths.items()
            }
    searched = ", ".join(str(path) for path in _prompt_dir_candidates())
    raise FileNotFoundError(
        f"Could not find {tuple(_PROMPT_FILENAMES.values())!r}; "
        f"searched prompt folders: {searched}"
    )


def _get_experience_model():
    """Use the configured auxiliary model, falling back to the main model."""
    from ..EvoScientist import _ensure_auxiliary_chat_model

    return _ensure_auxiliary_chat_model()


def _experience_concurrency_limit(requested: int | None = None) -> int:
    """Resolve the batch limit, preferring an explicit value then environment."""
    if requested is not None:
        return requested
    raw = os.environ.get("PAPER_NAV_EXPERIENCE_CONCURRENCY", "4")
    try:
        return max(1, min(64, int(raw)))
    except ValueError:
        return 4


async def _run_extraction(
    model: Any,
    system_prompt: str,
    paper_id: str,
    paper_markdown: str,
) -> str:
    response = await model.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"[paper_id] {paper_id}\n\n{paper_markdown}"),
        ]
    )
    return format_message_content(response).strip()


def parse_experience_json(raw_output: str, *, level: ExperienceLevel) -> dict[str, Any]:
    """Parse plain or fenced model JSON and validate its cacheable shape."""
    candidate = raw_output.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.IGNORECASE | re.DOTALL
    )
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as first_error:
        # Some providers still add a short preface despite the strict prompt.
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
        raise ExperienceOutputError(
            f"{level.upper()} extraction must be a top-level JSON object"
        )
    experiences = payload.get("experiences")
    if not isinstance(experiences, list) or not all(
        isinstance(item, dict) for item in experiences
    ):
        raise ExperienceOutputError(
            f"{level.upper()} extraction must contain an 'experiences' object list"
        )
    paper_id = payload.get("paper_id")
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise ExperienceOutputError(
            f"{level.upper()} extraction must contain a non-empty 'paper_id'"
        )
    return payload


async def run_experience_extraction(
    paper_id: str,
    paper_markdown: str,
    *,
    levels: tuple[ExperienceLevel, ...] = ("l1", "l2"),
    prompts: Mapping[ExperienceLevel, str] | None = None,
    model: Any | None = None,
) -> dict[ExperienceLevel, dict[str, Any] | Exception]:
    """Run requested prompt levels concurrently and parse each result independently."""
    loaded_prompts = dict(prompts or load_experience_prompts())
    extraction_model = model or _get_experience_model()
    normalized_id = paper_identifier(paper_id)

    async def _one(level: ExperienceLevel) -> dict[str, Any]:
        raw = await _run_extraction(
            extraction_model,
            loaded_prompts[level],
            normalized_id,
            paper_markdown,
        )
        return parse_experience_json(raw, level=level)

    values = await asyncio.gather(
        *(_one(level) for level in levels), return_exceptions=True
    )
    return dict(zip(levels, values, strict=True))


def _display(value: Any) -> str:
    if value is None or value == "":
        return "Not provided"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _keywords(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return _display(value)


def _source_paper(row: Mapping[str, Any], paper_id: str) -> str:
    paper_name = row.get("_paper_name")
    row_paper_id = row.get("_paper_id") or paper_id
    if paper_name:
        return f"{_display(paper_name)} ({_display(row_paper_id)})"
    return _display(row_paper_id)


def _format_practice_trace(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        return [f"- Practice trace: {_display(value)}"]
    lines = ["- Practice trace:"]
    for index, item in enumerate(value, start=1):
        if isinstance(item, Mapping):
            lines.extend(
                [
                    f"  {index}. Action: {_display(item.get('action'))}",
                    f"     Feedback: {_display(item.get('feedback'))}",
                ]
            )
        else:
            lines.append(f"  {index}. {_display(item)}")
    return lines


def format_l1_experiences(payload: Mapping[str, Any], paper_id: str) -> str:
    """Render parsed L1 JSON as prompt-oriented Markdown rather than raw JSON."""
    sections: list[str] = []
    for index, row in enumerate(payload.get("experiences", []), start=1):
        lines = [
            f"### L1-{index:03d}",
            f"- Source paper: {_source_paper(row, paper_id)}",
            f"- Granularity: {_display(row.get('granularity'))}",
            f"- Domain: {_display(row.get('domain'))}",
            f"- Keywords: {_keywords(row.get('keywords'))}",
        ]
        task = row.get("t")
        if isinstance(task, Mapping):
            lines.extend(
                [
                    f"- Task summary: {_display(task.get('summary'))}",
                    f"- Data/modality: {_display(task.get('modality'))}",
                    f"- Scale: {_display(task.get('scale'))}",
                    f"- Constraints/limitations: {_display(task.get('constraint'))}",
                ]
            )
        else:
            lines.append(f"- Task: {_display(task)}")
        lines.append(f"- Evaluation/evidence: {_display(row.get('e'))}")
        lines.extend(_format_practice_trace(row.get("practice_trace")))
        lines.extend(
            [
                f"- Reusable research experience:\n{_display(row.get('narrative'))}",
                f"- Source section: {_display(row.get('source_section'))}",
                f"- Supporting quote: {_display(row.get('source_quote'))}",
            ]
        )
        sections.append("\n".join(lines))
    return "\n\n".join(sections) or "No L1 experiences were extracted from this paper."


def format_l2_experiences(payload: Mapping[str, Any], paper_id: str) -> str:
    """Render parsed L2 JSON as readable evidence with scope and mechanism."""
    sections: list[str] = []
    for index, row in enumerate(payload.get("experiences", []), start=1):
        lines = [
            f"### L2-{index:03d}",
            f"- Source paper: {_source_paper(row, paper_id)}",
            f"- Claim type: {_display(row.get('claim_type'))}",
            f"- Domain: {_display(row.get('domain'))}",
            f"- Keywords: {_keywords(row.get('keywords'))}",
            f"- Keyword summary: {_display(row.get('keywords_summary'))}",
            f"- Core declaration: {_display(row.get('declaration'))}",
        ]
        context = row.get("context")
        if isinstance(context, Mapping):
            lines.extend(
                [
                    f"- Context summary: {_display(context.get('summary'))}",
                    f"- Modality: {_display(context.get('modality'))}",
                    f"- Scale: {_display(context.get('scale'))}",
                    f"- Constraints: {_display(context.get('constraint'))}",
                ]
            )
        else:
            lines.append(f"- Context: {_display(context)}")
        lines.extend(
            [
                f"- Quantified evidence/strength (mu): {_display(row.get('μ', row.get('mu')))}",
                f"- Applicability condition or mechanism (r): {_display(row.get('r'))}",
                f"- Evidence for r (mu_r): {_display(row.get('μ_r', row.get('mu_r')))}",
                f"- Mechanism depth: {_display(row.get('r_depth'))}",
                f"- Detailed empirical/causal experience:\n{_display(row.get('narrative'))}",
                f"- Source section: {_display(row.get('source_section'))}",
                f"- Supporting quote: {_display(row.get('source_quote'))}",
            ]
        )
        sections.append("\n".join(lines))
    return "\n\n".join(sections) or "No L2 experiences were extracted from this paper."


def format_experience_output(
    paper_id: str,
    l1_payload: Mapping[str, Any],
    l2_payload: Mapping[str, Any],
    *,
    cache_status: str | None = None,
) -> str:
    """Format parsed experience records for insertion into agent context."""
    cache_line = f"\n**Session cache:** {cache_status}\n" if cache_status else ""
    return f"""# Extracted Research Experiences

**Paper ID:** {paper_identifier(paper_id)}
{cache_line}
## L1 Practical Experiences

{format_l1_experiences(l1_payload, paper_identifier(paper_id))}

## L2 Inductive Experiences

{format_l2_experiences(l2_payload, paper_identifier(paper_id))}
"""


def _experience_dir(memory_dir: str | Path, session_id: str, paper_id: str) -> Path:
    session_key = _safe_storage_segment(session_id, prefix="session")
    paper_key = _safe_storage_segment(_canonical_paper_key(paper_id), prefix="paper")
    return Path(memory_dir) / "paper_experiences" / "sessions" / session_key / paper_key


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _extraction_lock(experience_dir: Path) -> asyncio.Lock:
    """Return the process-local lock for one session/paper cache entry.

    Paper-navigator invokes extraction through the in-process runtime tool, so
    this prevents duplicate LLM calls for concurrent reads without holding a
    synchronous filesystem lock across asynchronous provider requests.
    """
    key = str(experience_dir)
    with _EXTRACTION_LOCKS_GUARD:
        return _EXTRACTION_LOCKS.setdefault(key, asyncio.Lock())


def _load_cached_level(
    experience_dir: Path,
    metadata: Mapping[str, Any],
    *,
    level: ExperienceLevel,
    paper_hash: str,
    prompt_hash: str,
) -> dict[str, Any] | None:
    levels = metadata.get("levels")
    if not isinstance(levels, Mapping):
        return None
    level_meta = levels.get(level)
    if not isinstance(level_meta, Mapping):
        return None
    if (
        level_meta.get("paper_sha256") != paper_hash
        or level_meta.get("prompt_sha256") != prompt_hash
    ):
        return None
    payload = _read_json_object(experience_dir / f"{level}.json")
    if payload is None:
        return None
    try:
        # Revalidate on read so malformed/manual cache edits never enter context.
        return parse_experience_json(json.dumps(payload), level=level)
    except ExperienceOutputError:
        return None


async def extract_and_store_paper_experiences(
    *,
    paper_id: str,
    paper_markdown: str,
    session_id: str,
    memory_dir: str | Path,
    model: Any | None = None,
    model_getter: Callable[[], Any] | None = None,
    prompts: Mapping[ExperienceLevel, str] | None = None,
) -> str:
    """Return readable experiences, reusing only the current session's cache."""
    if not session_id.strip():
        raise ValueError("A non-empty session_id is required for experience memory")
    loaded_prompts = dict(prompts or load_experience_prompts())
    normalized_id = paper_identifier(paper_id)
    paper_hash = _sha256_text(paper_markdown)
    prompt_hashes = {
        level: _sha256_text(loaded_prompts[level]) for level in ("l1", "l2")
    }
    experience_dir = _experience_dir(memory_dir, session_id, normalized_id)
    experience_dir.mkdir(parents=True, exist_ok=True)
    lock = _extraction_lock(experience_dir)
    async with lock:
        metadata_path = experience_dir / "metadata.json"
        metadata = _read_json_object(metadata_path) or {}
        cached: dict[ExperienceLevel, dict[str, Any]] = {}
        missing: list[ExperienceLevel] = []
        for level in ("l1", "l2"):
            payload = _load_cached_level(
                experience_dir,
                metadata,
                level=level,
                paper_hash=paper_hash,
                prompt_hash=prompt_hashes[level],
            )
            if payload is None:
                missing.append(level)
            else:
                cached[level] = payload

        extracted: dict[ExperienceLevel, dict[str, Any]] = {}
        failures: dict[ExperienceLevel, Exception] = {}
        if missing:
            extraction_model = model
            if extraction_model is None and model_getter is not None:
                extraction_model = model_getter()
            results = await run_experience_extraction(
                normalized_id,
                paper_markdown,
                levels=tuple(missing),
                prompts=loaded_prompts,
                model=extraction_model,
            )
            for level, value in results.items():
                if isinstance(value, Exception):
                    failures[level] = value
                else:
                    # The requested ID is authoritative for cache identity.
                    value = dict(value)
                    value["paper_id"] = normalized_id
                    extracted[level] = value

        now = datetime.now(UTC).isoformat()
        level_metadata = metadata.get("levels")
        if not isinstance(level_metadata, dict):
            level_metadata = {}
        for level, payload in extracted.items():
            _atomic_write_json(experience_dir / f"{level}.json", payload)
            level_metadata[level] = {
                "experience_count": len(payload["experiences"]),
                "extracted_at": now,
                "paper_sha256": paper_hash,
                "prompt_sha256": prompt_hashes[level],
            }

        metadata = {
            "store_version": _STORE_VERSION,
            "session_id": session_id,
            "paper_id": normalized_id,
            "canonical_paper_key": _canonical_paper_key(normalized_id),
            "updated_at": now,
            "levels": level_metadata,
        }
        if extracted or not metadata_path.exists():
            _atomic_write_json(metadata_path, metadata)

        complete = {**cached, **extracted}
        if "l1" in complete and "l2" in complete:
            cache_status = "hit (no LLM calls)" if not missing else "updated"
            rendered = format_experience_output(
                normalized_id,
                complete["l1"],
                complete["l2"],
                cache_status=cache_status,
            )
            _atomic_write_text(experience_dir / "rendered.md", rendered)
            if not failures:
                return rendered

        if failures:
            detail = "; ".join(
                f"{level.upper()}: {error}" for level, error in failures.items()
            )
            persisted = ", ".join(level.upper() for level in extracted) or "none"
            raise ExperienceOutputError(
                f"Experience extraction failed ({detail}). "
                f"Successfully parsed levels persisted this session: {persisted}."
            )
        raise ExperienceOutputError("L1/L2 experience cache is incomplete")


def _runtime_session_id(runtime: ToolRuntime | None) -> str | None:
    if runtime is None:
        return None
    try:
        if runtime.execution_info and runtime.execution_info.thread_id:
            return str(runtime.execution_info.thread_id)
    except Exception:
        pass
    config = runtime.config or {}
    if isinstance(config, Mapping):
        configurable = config.get("configurable", {})
        if isinstance(configurable, Mapping):
            value = configurable.get("thread_id")
            if isinstance(value, str) and value:
                return value
    return None


def _resolve_workspace_paper_file(paper_file: str) -> Path:
    from .. import paths

    workspace_root = paths.resolve_virtual_path("/").resolve()
    requested = Path(paper_file).expanduser()
    if requested.is_absolute() and requested.is_relative_to(workspace_root):
        resolved = requested.resolve()
    else:
        resolved = paths.resolve_virtual_path(paper_file).resolve()
    if not resolved.is_relative_to(workspace_root):
        raise ValueError("paper_file must remain inside the active workspace")
    if not resolved.is_file():
        raise FileNotFoundError(f"Paper Markdown file does not exist: {paper_file}")
    return resolved


def create_paper_experience_tool(
    *,
    memory_dir: str | Path,
    model_getter: Callable[[], Any] | None = None,
) -> BaseTool:
    """Build the main-agent tool used only by the paper-navigator skill."""

    async def _extract_paper_experiences(
        paper_file: str,
        paper_id: str,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> str:
        session_id = _runtime_session_id(runtime)
        if session_id is None:
            return (
                "Experience extraction failed: the current session/thread ID "
                "was unavailable, so no session-isolated memory was written."
            )
        try:
            path = _resolve_workspace_paper_file(paper_file)
            paper_markdown = path.read_text(encoding="utf-8")
            return await extract_and_store_paper_experiences(
                paper_id=paper_id,
                paper_markdown=paper_markdown,
                session_id=session_id,
                memory_dir=memory_dir,
                model_getter=model_getter,
            )
        except Exception as exc:
            return f"Experience extraction failed: {exc}"

    return StructuredTool.from_function(
        coroutine=_extract_paper_experiences,
        name="extract_paper_experiences",
        description=(
            "Extract L1 practical and L2 inductive experiences from a full-paper "
            "Markdown file fetched by the paper-navigator skill. The tool uses "
            "the auxiliary LLM when configured, caches parsed JSON only within "
            "the current session, and returns readable Markdown rather than raw "
            "JSON. Use only after paper-navigator fetches a final paper's full text."
        ),
        args_schema=PaperExperienceArgs,
        infer_schema=False,
    )


def create_paper_experience_batch_tool(
    *,
    memory_dir: str | Path,
    model_getter: Callable[[], Any] | None = None,
) -> BaseTool:
    """Build the concurrent batch enrichment tool for paper-navigator."""

    async def _extract_paper_experiences_batch(
        papers: list[PaperExperienceBatchItem],
        max_concurrency: int | None = None,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> str:
        session_id = _runtime_session_id(runtime)
        if session_id is None:
            return (
                "Experience extraction failed: the current session/thread ID "
                "was unavailable, so no session-isolated memory was written."
            )
        return await extract_and_store_paper_experiences_batch(
            papers=[item.model_dump() for item in papers],
            session_id=session_id,
            memory_dir=memory_dir,
            model_getter=model_getter,
            max_concurrency=max_concurrency,
        )

    return StructuredTool.from_function(
        coroutine=_extract_paper_experiences_batch,
        name="extract_paper_experiences_batch",
        description=(
            "Extract L1/L2 experiences for a batch of final paper Markdown files "
            "concurrently. Use this once after fetching all final papers instead "
            "of calling extract_paper_experiences repeatedly. max_concurrency "
            "defaults to PAPER_NAV_EXPERIENCE_CONCURRENCY or 4. The result includes "
            "the batch experience_extraction_elapsed_seconds timing."
        ),
        args_schema=PaperExperienceBatchArgs,
        infer_schema=False,
    )


async def _run_cli(
    paper_file: Path,
    paper_id: str,
    session_id: str,
    memory_dir: Path | None = None,
) -> str:
    from .. import paths

    paper_markdown = paper_file.read_text(encoding="utf-8")
    return await extract_and_store_paper_experiences(
        paper_id=paper_id,
        paper_markdown=paper_markdown,
        session_id=session_id,
        memory_dir=memory_dir or paths.MEMORIES_DIR,
    )


async def extract_and_store_paper_experiences_batch(
    *,
    papers: list[Mapping[str, str]],
    session_id: str,
    memory_dir: str | Path,
    model_getter: Callable[[], Any] | None = None,
    max_concurrency: int | None = None,
) -> str:
    """Extract experiences for multiple papers concurrently with a hard limit."""
    if not papers:
        raise ValueError("At least one paper is required")
    limit = _experience_concurrency_limit(max_concurrency)
    semaphore = asyncio.Semaphore(limit)
    started = time.perf_counter()

    async def _one(item: Mapping[str, str]) -> tuple[str, str]:
        paper_id = str(item["paper_id"])
        async with semaphore:
            try:
                path = _resolve_workspace_paper_file(str(item["paper_file"]))
                result = await extract_and_store_paper_experiences(
                    paper_id=paper_id,
                    paper_markdown=path.read_text(encoding="utf-8"),
                    session_id=session_id,
                    memory_dir=memory_dir,
                    model_getter=model_getter,
                )
                return paper_id, result
            except Exception as exc:
                return paper_id, f"Experience extraction unavailable: {exc}"

    results = await asyncio.gather(*(_one(item) for item in papers))
    elapsed = time.perf_counter() - started
    sections = [
        "# Paper Experience Extraction",
        f"- Papers: {len(results)}",
        f"- Concurrency limit: {limit}",
        f"- experience_extraction_elapsed_seconds: {elapsed:.3f}",
        "",
    ]
    for paper_id, result in results:
        sections.extend([f"## {paper_id}", result, ""])
    return "\n".join(sections).rstrip()


def main() -> None:
    """Offline/debug CLI; the runtime tool is the normal paper-navigator path."""
    parser = argparse.ArgumentParser(
        description="Extract and session-cache L1/L2 experiences from paper Markdown"
    )
    parser.add_argument("--paper-file", type=Path, required=True)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument(
        "--session-id",
        required=True,
        help="Explicit cache namespace; the agent tool injects this automatically.",
    )
    parser.add_argument("--memory-dir", type=Path)
    args = parser.parse_args()

    try:
        print(
            asyncio.run(
                _run_cli(
                    args.paper_file,
                    args.paper_id,
                    args.session_id,
                    args.memory_dir,
                )
            )
        )
    except Exception as exc:
        print(f"Experience extraction failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
