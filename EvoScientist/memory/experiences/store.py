"""File-backed, project-isolated paper experience storage."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import OrderedDict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .._atomic import atomic_write_json, atomic_write_text, read_json
from ..types import (
    ExperienceLevel,
    MemoryScope,
    MemoryType,
    ObservationReadResult,
    ObservationSearchDocument,
)
from .taxonomy import resolve_discipline

EXPERIENCE_DIR = "experiences/projects"
EXPERIENCE_CATALOG_FILENAME = "PAPER_EXPERIENCES.md"
STORE_VERSION = 2


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_paper_identifier(value: str) -> str:
    """Normalize common paper URLs just enough for stable deduplication."""
    normalized = value.strip()
    normalized = re.sub(r"v\d+(?=(?:\.pdf)?(?:[?#].*)?$)", "", normalized)
    normalized = normalized.removesuffix(".pdf").rstrip("/")
    return normalized.casefold()


def paper_storage_key(paper_id: str, url: str) -> str:
    canonical = canonical_paper_identifier(paper_id or url)
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", canonical).strip("._-") or "paper"
    return f"{slug[:48]}-{_sha256(canonical)[:12]}"


def _paper_dir(
    memory_dir: str | Path, *, project_id: str, paper_id: str, url: str
) -> Path:
    return (
        Path(memory_dir).expanduser()
        / EXPERIENCE_DIR
        / project_id
        / paper_storage_key(paper_id, url)
    )


def experience_catalog_path(*, memory_dir: str | Path, project_id: str) -> Path:
    """Return the WebUI-visible derived catalog for one project's experiences."""
    return (
        Path(memory_dir).expanduser()
        / "profile"
        / "projects"
        / project_id
        / EXPERIENCE_CATALOG_FILENAME
    )


def _catalog_scalar(value: Any, *, fallback: str = "") -> str:
    if not isinstance(value, str):
        return fallback
    compact = " ".join(value.split())
    return compact.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def refresh_experience_catalog(*, memory_dir: str | Path, project_id: str) -> Path:
    """Rebuild the read-only Markdown projection shown by the existing WebUI."""
    root = Path(memory_dir).expanduser()
    project_root = root / EXPERIENCE_DIR / project_id
    papers: list[tuple[str, Path, dict[str, Any]]] = []
    try:
        paper_dirs = sorted(path for path in project_root.iterdir() if path.is_dir())
    except OSError:
        paper_dirs = []

    for directory in paper_dirs:
        metadata = read_json(directory / "metadata.json")
        if metadata is None or metadata.get("project_id") != project_id:
            continue
        extracted_at = str(metadata.get("extracted_at") or "")
        papers.append((extracted_at, directory, metadata))
    papers.sort(key=lambda item: (item[0], item[1].name), reverse=True)

    lines = [
        "# Paper experiences",
        "",
        "Extracted paper experience saved for this project. This file is a "
        "read-only catalog generated from the isolated experience store.",
        "",
        f"Papers: {len(papers)}",
        "",
    ]
    for extracted_at, directory, metadata in papers:
        title = _catalog_scalar(
            metadata.get("title"),
            fallback=_catalog_scalar(metadata.get("paper_id"), fallback=directory.name),
        )
        paper_id = _catalog_scalar(metadata.get("paper_id"))
        url = _catalog_scalar(metadata.get("url"))
        lines.extend([f"## {title}", ""])
        if paper_id:
            lines.append(f"- Paper ID: `{paper_id.replace('`', '')}`")
        if url:
            lines.append(f"- URL: {url}")
        if extracted_at:
            lines.append(f"- Extracted: {extracted_at}")
        lines.append("")

        found_experiences = False
        for level in ("l1", "l2"):
            payload = read_json(directory / f"{level}.json")
            experiences = payload.get("experiences") if payload else None
            items = experiences if isinstance(experiences, list) else []
            lines.extend([f"### {level.upper()} experiences ({len(items)})", ""])
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                found_experiences = True
                experience_id = _experience_id(
                    project_id=project_id,
                    paper_key=directory.name,
                    level=level,
                    index=index,
                    item=item,
                )
                summary = _catalog_scalar(
                    _summary(item, title=title, level=level),
                    fallback=f"{level.upper()} experience",
                )
                lines.append(f"- `{experience_id}`: {summary}")
            if not items:
                lines.append("- No experience records.")
            lines.append("")
        if not found_experiences:
            lines.extend(["No valid experience records were parsed.", ""])

    target = experience_catalog_path(memory_dir=root, project_id=project_id)
    atomic_write_text(target, "\n".join(lines).rstrip() + "\n")
    return target


def refresh_all_experience_catalogs(*, memory_dir: str | Path) -> list[Path]:
    """Backfill WebUI catalogs for every project already holding experiences."""
    root = Path(memory_dir).expanduser()
    projects_root = root / EXPERIENCE_DIR
    try:
        project_dirs = sorted(
            path
            for path in projects_root.iterdir()
            if path.is_dir() and not path.is_symlink()
        )
    except OSError:
        return []

    catalogs: list[Path] = []
    for project_dir in project_dirs:
        try:
            catalogs.append(
                refresh_experience_catalog(
                    memory_dir=root,
                    project_id=project_dir.name,
                )
            )
        except OSError:
            continue
    return catalogs


def store_paper_experiences(
    *,
    memory_dir: str | Path,
    project_id: str,
    paper_id: str,
    url: str,
    title: str,
    paper_text: str,
    prompts: Mapping[ExperienceLevel, str],
    payloads: Mapping[ExperienceLevel, Mapping[str, Any]],
    domain_arxiv: str | None = None,
) -> Path:
    """Atomically persist prompt-compatible L1 and L2 JSON payloads."""
    directory = _paper_dir(
        memory_dir, project_id=project_id, paper_id=paper_id, url=url
    )
    now = datetime.now(UTC).isoformat()
    for level in ("l1", "l2"):
        atomic_write_json(directory / f"{level}.json", payloads[level])
        # The parse cache validates on (mtime_ns, size). A re-extraction that
        # lands in the same nanosecond tick and produces the same byte count
        # would otherwise keep serving the old parse, so drop these two keys
        # explicitly on the one path that writes them.
        with _cache_lock:
            _level_parse_cache.pop(str(directory / f"{level}.json"), None)
    atomic_write_json(
        directory / "metadata.json",
        {
            "store_version": STORE_VERSION,
            "project_id": project_id,
            "paper_id": paper_id,
            "canonical_paper_id": canonical_paper_identifier(paper_id or url),
            "title": title,
            "url": url,
            "domain_arxiv": domain_arxiv,
            "paper_sha256": _sha256(paper_text),
            "prompt_sha256": {level: _sha256(prompts[level]) for level in ("l1", "l2")},
            "confidence_policy": "single_paper_evidence_v1",
            "supporting_papers": [canonical_paper_identifier(paper_id or url)],
            "contradicting_papers": [],
            "extracted_at": now,
        },
    )
    refresh_experience_catalog(memory_dir=memory_dir, project_id=project_id)
    return directory


def load_paper_experiences(
    *, memory_dir: str | Path, project_id: str, paper_id: str, url: str
) -> dict[str, Any] | None:
    """Load one complete, prompt-compatible L1/L2 extraction if it exists."""
    directory = _paper_dir(
        memory_dir, project_id=project_id, paper_id=paper_id, url=url
    )
    metadata = read_json(directory / "metadata.json")
    l1 = read_json(directory / "l1.json")
    l2 = read_json(directory / "l2.json")
    if (
        metadata is None
        or metadata.get("project_id") != project_id
        or l1 is None
        or l2 is None
    ):
        return None
    return {"metadata": metadata, "l1": l1, "l2": l2}


def _has_full_text(root: Path, *, project_id: str, paper_key: str) -> bool:
    """Report whether this paper's full text is stored alongside its experiences.

    Imported lazily: ``papers.store`` depends on this module for
    ``paper_storage_key()``, so a module-level import here would be circular.
    """
    from ..papers.store import has_paper_text

    return has_paper_text(root, project_id=project_id, paper_key=paper_key)


def _experience_id(
    *,
    project_id: str,
    paper_key: str,
    level: ExperienceLevel,
    index: int,
    item: Mapping[str, Any] | None = None,
) -> str:
    """Derive one record's stable ID, preferring its content over its position.

    The ID used to hash the record's *index* in its level's array. That made it
    positional: re-extracting a paper reorders or re-counts the records, so the
    same finding gets a different ID and a different finding inherits the old
    one. Two things depend on an `E-*` ID meaning one specific claim -- the
    policy cache, keyed on `(task, sorted(selected_ids))`, and the audit trail
    from a policy line back to the record that supports it. Under drift the
    cache silently returns a policy synthesized from other records, and a cited
    ID resolves to something the policy never read.

    Hashing the record's own identifying text fixes both: the ID follows the
    claim. `statement` is the field every layer carries and the one that
    actually distinguishes records; `domain`/`task` join it so two records
    sharing boilerplate still separate. The index remains the fallback for
    records with no usable text, where position is all there is.
    """
    identity = ""
    if item is not None:
        identity = "\x1f".join(
            " ".join(str(item.get(key) or "").split())
            for key in ("statement", "declaration", "domain", "task")
        ).strip("\x1f")
    if identity:
        return f"E-{_sha256(f'{project_id}:{paper_key}:{level}:{identity}')[:16]}"
    return f"E-{_sha256(f'{project_id}:{paper_key}:{level}:{index}')[:16]}"


def _facet(item: Mapping[str, Any], key: str) -> str | None:
    """Read one free-text facet off a record, normalized, or None."""
    value = item.get(key)
    if not isinstance(value, str):
        return None
    compact = " ".join(value.split())
    return compact or None


def _summary(item: Mapping[str, Any], *, title: str, level: ExperienceLevel) -> str:
    """Build the ranking summary for one record.

    ``summary`` carries triple the ranking weight of ``body``, so what goes in
    it decides what the record is findable by. Leading with ``domain`` and
    ``task`` puts the record's own subject-matter vocabulary in the high-weight
    field -- previously both lived only in the JSON body at weight 1.0, while
    the summary held the opening 240 characters of ``statement``, which is
    prose framing the problem rather than naming the subject.

    ``transferable_core`` is preferred over ``statement`` for the descriptive
    tail because it is the paper-agnostic rephrasing: it keeps the causal claim
    and drops the dataset and model names that make a record match queries
    about the source paper rather than about the technique.
    """
    facets = [value for key in ("domain", "task") if (value := _facet(item, key))]
    candidates = (
        item.get("transferable_core"),
        item.get("statement"),
        item.get("declaration"),
        item.get("keywords_summary"),
        item.get("narrative"),
    )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            facets.append(" ".join(value.split()))
            break
    if not facets:
        return f"{level.upper()} experience from {title or 'paper'}"
    return " · ".join(facets)[:240]


def _experience_text(
    *,
    experience_id: str,
    level: ExperienceLevel,
    metadata: Mapping[str, Any],
    item: Mapping[str, Any],
    paper_key: str = "",
    full_text_available: bool = False,
) -> str:
    # `paper_key` names both this paper's experience directory and its
    # full-text directory, so advertising it here plus whether paper.md exists
    # tells a reader it can go from this judgement to the evidence behind it
    # with `search_paper_text` -- no extra lookup round trip.
    return json.dumps(
        {
            "id": experience_id,
            "record_kind": "experience",
            "experience_level": level,
            "scope": "project",
            "project_id": metadata.get("project_id"),
            "paper": {
                "paper_id": metadata.get("paper_id"),
                "title": metadata.get("title"),
                "url": metadata.get("url"),
                "paper_key": paper_key,
                "full_text_available": full_text_available,
            },
            "experience": item,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


# ── Parsed-document cache ─────────────────────────────────────────────
#
# One level file (`l1.json` / `l2.json`) parses to a list of documents, keyed on
# its path and validated against the file's ``(st_mtime_ns, st_size)``
# signature, so a re-extraction invalidates exactly that paper's entry.
#
# Every caller here re-read and re-parsed the whole library: ranking, the
# prompt-facing index block rebuilt on every model request, reading a single
# record by ID, and `apply_experience` twice per call. At 98 records that was
# ~20ms per pass with the JSON parsed identically each time. The work is pure
# and the inputs are files on disk, so it is memoizable as-is.
#
# Mirrors ``observations/store.py``, including the working-set trim rule: every
# entry a call touches is moved to the most-recent end, and trimming stops at
# ``len(touched)``, so a library larger than the cap degrades to no-caching
# instead of thrashing.
_LevelParseValue = tuple[tuple[int, int], list[ObservationSearchDocument]]

_level_parse_cache: OrderedDict[str, _LevelParseValue] = OrderedDict()

# Serializes cache transactions so a concurrent call cannot evict a key between
# another call's lookup and its recency update. Parsing stays outside the lock.
_cache_lock = threading.Lock()

_cached_max_files: int | None = None


def _max_cached_files() -> int:
    """Return the configured cache cap, read once from config on first call."""
    global _cached_max_files
    if _cached_max_files is None:
        from ...config.settings import get_effective_config

        _cached_max_files = int(
            getattr(get_effective_config(), "memory_observation_cache_max_files", 2048)
        )
    return _cached_max_files


def _trim_parse_cache(touched: set[str]) -> None:
    """Trim to ``max(cap, len(touched))``, never evicting this call's own set."""
    target = max(_max_cached_files(), len(touched))
    with _cache_lock:
        while len(_level_parse_cache) > target:
            _level_parse_cache.popitem(last=False)


def reset_experience_parse_cache() -> None:
    """Drop the parse cache. For tests that rewrite files within one mtime tick."""
    with _cache_lock:
        _level_parse_cache.clear()


def _parse_level_documents(
    *,
    root: Path,
    directory: Path,
    level: ExperienceLevel,
    project_id: str,
    metadata: Mapping[str, Any],
    full_text: bool,
) -> list[ObservationSearchDocument]:
    """Parse one level file into documents, with no caching of its own."""
    payload = read_json(directory / f"{level}.json")
    experiences = payload.get("experiences") if payload else None
    if not isinstance(experiences, list):
        return []
    relative = (directory / f"{level}.json").relative_to(root).as_posix()
    documents: list[ObservationSearchDocument] = []
    for index, item in enumerate(experiences):
        if not isinstance(item, dict):
            continue
        experience_id = _experience_id(
            project_id=project_id,
            paper_key=directory.name,
            level=level,
            index=index,
            item=item,
        )
        text = _experience_text(
            experience_id=experience_id,
            level=level,
            metadata=metadata,
            item=item,
            paper_key=directory.name,
            full_text_available=full_text,
        )
        documents.append(
            ObservationSearchDocument(
                observation_id=experience_id,
                path=f"/memories/{relative}",
                memory_type=MemoryType.SEMANTIC,
                scope=MemoryScope.PROJECT,
                summary=_summary(
                    item,
                    title=str(metadata.get("title") or ""),
                    level=level,
                ),
                body=text,
                text=text,
                record_kind="experience",
                experience_level=level,
                # Resolved per record, not per paper: an interdisciplinary paper
                # can carry records the extraction model files under different
                # disciplines, and the record's own value wins over the paper's
                # arXiv category.
                discipline=resolve_discipline(
                    discipline=item.get("discipline"),
                    domain_arxiv=metadata.get("domain_arxiv"),
                ),
                domain=_facet(item, "domain"),
                task=_facet(item, "task"),
                paper_key=directory.name,
            )
        )
    return documents


def _level_documents_with_cache(
    *,
    root: Path,
    directory: Path,
    level: ExperienceLevel,
    project_id: str,
    metadata: Mapping[str, Any],
    full_text: bool,
    touched: set[str],
) -> list[ObservationSearchDocument]:
    """Return one level file's documents, memoized on its mtime/size signature.

    The cached list is returned without copying: ``ObservationSearchDocument`` is
    a frozen dataclass and callers only iterate it or build new ones through
    ``replace``.
    """
    path = directory / f"{level}.json"
    key = str(path)
    try:
        st = path.stat()
    except OSError:
        return []
    signature = (st.st_mtime_ns, st.st_size)
    with _cache_lock:
        cached = _level_parse_cache.get(key)
        if cached is not None and cached[0] == signature:
            _level_parse_cache.move_to_end(key)
            touched.add(key)
            return cached[1]
    documents = _parse_level_documents(
        root=root,
        directory=directory,
        level=level,
        project_id=project_id,
        metadata=metadata,
        full_text=full_text,
    )
    with _cache_lock:
        _level_parse_cache[key] = (signature, documents)
        _level_parse_cache.move_to_end(key)
        touched.add(key)
    return documents


def list_experience_documents(
    *, memory_dir: str | Path, project_id: str
) -> list[ObservationSearchDocument]:
    """Expose project experiences as documents consumable by EvoMemory search."""
    root = Path(memory_dir).expanduser()
    project_root = root / EXPERIENCE_DIR / project_id
    documents: list[ObservationSearchDocument] = []
    touched: set[str] = set()
    try:
        paper_dirs = sorted(path for path in project_root.iterdir() if path.is_dir())
    except OSError:
        return []
    for directory in paper_dirs:
        metadata = read_json(directory / "metadata.json")
        if metadata is None or metadata.get("project_id") != project_id:
            continue
        # One stat per paper, not per experience: every record from this paper
        # shares the same answer.
        full_text = _has_full_text(
            root, project_id=project_id, paper_key=directory.name
        )
        for level in ("l1", "l2"):
            documents.extend(
                _level_documents_with_cache(
                    root=root,
                    directory=directory,
                    level=level,
                    project_id=project_id,
                    metadata=metadata,
                    full_text=full_text,
                    touched=touched,
                )
            )
    _trim_parse_cache(touched)
    return documents


def read_experience_file(
    *, memory_dir: str | Path, project_id: str, experience_id: str
) -> ObservationReadResult | None:
    """Read one stable experience record in the active project."""
    requested = experience_id.strip()
    if not requested:
        return None
    for document in list_experience_documents(
        memory_dir=memory_dir, project_id=project_id
    ):
        if document.observation_id != requested:
            continue
        return {
            "observation_id": document.observation_id,
            "path": document.path,
            "memory_type": document.memory_type,
            "scope": document.scope,
            "summary": document.summary,
            "text": document.text,
            "record_kind": "experience",
            "experience_level": document.experience_level or "l1",
        }
    return None
