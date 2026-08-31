"""File-backed, project-isolated storage for paper full text.

Mirrors ``memory/experiences/store.py`` deliberately: the same
``paper_storage_key()`` names the directory, so one paper's experience
directory and its full-text directory share a name and either side can
derive the other with no lookup. That symmetry is what lets an ``E-*``
record advertise "full text available" and hand the agent a ``paper_id``
it can search.

Layout, per project::

    papers/projects/<project_id>/<paper_key>/
        paper.md        # references-stripped Markdown, exactly as chunked
        chunks.jsonl    # one chunk per line, offsets into paper.md
        metadata.json   # paper identity, sha256, chunk/section counts
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .._atomic import atomic_write_json, atomic_write_text, read_json
from ..experiences.store import canonical_paper_identifier, paper_storage_key
from ..types import (
    MemoryScope,
    MemoryType,
    ObservationSearchDocument,
)
from .chunking import (
    CHUNKING_VERSION,
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_OVERLAP_CHARS,
    PaperChunk,
    chunk_paper_text,
)

PAPER_DIR = "papers/projects"
STORE_VERSION = 1

# A search summary is what the TF-IDF ranker weights x3, so it holds the
# section path plus a short lead-in rather than the whole chunk (weighted x1
# as the body).
_SUMMARY_LEAD_CHARS = 160


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def paper_dir(
    memory_dir: str | Path, *, project_id: str, paper_id: str, url: str
) -> Path:
    """Return the full-text directory for one paper in one project."""
    return (
        Path(memory_dir).expanduser()
        / PAPER_DIR
        / project_id
        / paper_storage_key(paper_id, url)
    )


def paper_dir_for_key(
    memory_dir: str | Path, *, project_id: str, paper_key: str
) -> Path:
    """Return the full-text directory addressed by an already-computed key.

    Used by the experience store, which knows only its own directory name --
    which is the same key by construction.
    """
    return Path(memory_dir).expanduser() / PAPER_DIR / project_id / paper_key


def has_paper_text(memory_dir: str | Path, *, project_id: str, paper_key: str) -> bool:
    """Report whether one paper's full text is on disk, cheaply.

    One ``stat`` per call: this runs once per experience record while building
    search documents, so it must not read or parse anything.
    """
    return (
        paper_dir_for_key(memory_dir, project_id=project_id, paper_key=paper_key)
        / "paper.md"
    ).is_file()


def _chunk_summary(chunk: PaperChunk) -> str:
    lead = " ".join(chunk.text.split())[:_SUMMARY_LEAD_CHARS]
    if chunk.section_path:
        return f"{chunk.section_path} -- {lead}"
    return lead


def _write_chunks(path: Path, chunks: list[PaperChunk]) -> None:
    """Write chunks as JSONL through the same atomic tmp+replace path."""
    atomic_write_text(
        path,
        "".join(
            json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for chunk in chunks
        ),
    )


def store_paper_text(
    *,
    memory_dir: str | Path,
    project_id: str,
    paper_id: str,
    url: str,
    title: str,
    paper_text: str,
    domain_arxiv: str | None = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> Path:
    """Persist one paper's full text plus its chunk index, atomically.

    The signature intentionally tracks ``store_paper_experiences()`` so both
    can be called from the same place with the same arguments.
    """
    directory = paper_dir(memory_dir, project_id=project_id, paper_id=paper_id, url=url)
    key = paper_storage_key(paper_id, url)
    chunks = chunk_paper_text(
        paper_text,
        project_id=project_id,
        paper_key=key,
        max_chunk_chars=max_chunk_chars,
        overlap_chars=overlap_chars,
    )
    # paper.md lands before chunks.jsonl: offsets in a chunk record are only
    # meaningful once the text they index exists.
    atomic_write_text(directory / "paper.md", paper_text)
    _write_chunks(directory / "chunks.jsonl", chunks)
    atomic_write_json(
        directory / "metadata.json",
        {
            "store_version": STORE_VERSION,
            "chunking_version": CHUNKING_VERSION,
            "project_id": project_id,
            "paper_key": key,
            "paper_id": paper_id,
            "canonical_paper_id": canonical_paper_identifier(paper_id or url),
            "title": title,
            "url": url,
            "domain_arxiv": domain_arxiv,
            "paper_sha256": _sha256(paper_text),
            "char_count": len(paper_text),
            "chunk_count": len(chunks),
            "section_count": len({chunk.section_path for chunk in chunks}),
            "max_chunk_chars": max_chunk_chars,
            "overlap_chars": overlap_chars,
        },
    )
    return directory


def _load_chunks(directory: Path) -> list[dict[str, Any]]:
    """Read chunks.jsonl, skipping malformed lines rather than failing.

    A truncated final line can only come from a write that never completed,
    in which case the previous file was never replaced -- but tolerating it
    keeps a corrupted store searchable instead of raising into a tool call.
    """
    try:
        raw = (directory / "chunks.jsonl").read_text(encoding="utf-8")
    except OSError:
        return []
    chunks: list[dict[str, Any]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            chunks.append(payload)
    return chunks


def _read_metadata(directory: Path, *, project_id: str) -> dict[str, Any] | None:
    metadata = read_json(directory / "metadata.json")
    if metadata is None or metadata.get("project_id") != project_id:
        return None
    return metadata


def load_paper_text(
    *, memory_dir: str | Path, project_id: str, paper_id: str, url: str
) -> dict[str, Any] | None:
    """Load one stored paper's text, metadata, and chunks, if all are present."""
    directory = paper_dir(memory_dir, project_id=project_id, paper_id=paper_id, url=url)
    metadata = _read_metadata(directory, project_id=project_id)
    if metadata is None:
        return None
    try:
        text = (directory / "paper.md").read_text(encoding="utf-8")
    except OSError:
        return None
    return {
        "metadata": metadata,
        "text": text,
        "chunks": _load_chunks(directory),
    }


def _project_paper_dirs(root: Path, *, project_id: str) -> list[Path]:
    project_root = root / PAPER_DIR / project_id
    try:
        return sorted(
            path
            for path in project_root.iterdir()
            if path.is_dir() and not path.is_symlink()
        )
    except OSError:
        return []


def list_paper_chunk_documents(
    *, memory_dir: str | Path, project_id: str, paper_id: str | None = None
) -> list[ObservationSearchDocument]:
    """Expose stored chunks in one project as search documents.

    These feed the dedicated paper search entry point only. They are
    deliberately absent from both the observation and the experience retriever.

    ``paper_id`` restricts the set to one paper. Filtering here rather than
    after ranking keeps the ranker's IDF computed over exactly the candidates
    that can be returned.
    """
    root = Path(memory_dir).expanduser()
    if paper_id and paper_id.strip():
        found = find_paper_directory(root, project_id=project_id, paper_id=paper_id)
        directories = [found[0]] if found else []
    else:
        directories = _project_paper_dirs(root, project_id=project_id)

    documents: list[ObservationSearchDocument] = []
    for directory in directories:
        metadata = _read_metadata(directory, project_id=project_id)
        if metadata is None:
            continue
        relative = (directory / "paper.md").relative_to(root).as_posix()
        for chunk in _load_chunks(directory):
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            if not isinstance(chunk_id, str) or not isinstance(text, str):
                continue
            section_path = (
                chunk["section_path"]
                if isinstance(chunk.get("section_path"), str)
                else ""
            )
            summary = _chunk_summary(
                PaperChunk(
                    chunk_id=chunk_id,
                    chunk_index=int(chunk.get("chunk_index") or 0),
                    section=str(chunk.get("section") or ""),
                    section_path=section_path,
                    char_start=int(chunk.get("char_start") or 0),
                    char_end=int(chunk.get("char_end") or 0),
                    text=text,
                )
            )
            documents.append(
                ObservationSearchDocument(
                    observation_id=chunk_id,
                    path=f"/memories/{relative}",
                    memory_type=MemoryType.SEMANTIC,
                    scope=MemoryScope.PROJECT,
                    summary=summary,
                    body=text,
                    text=text,
                    record_kind="paper_chunk",
                )
            )
    return documents


def _locate_chunk(
    root: Path, *, project_id: str, chunk_id: str
) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    wanted = chunk_id.strip()
    for directory in _project_paper_dirs(root, project_id=project_id):
        metadata = _read_metadata(directory, project_id=project_id)
        if metadata is None:
            continue
        for chunk in _load_chunks(directory):
            if chunk.get("chunk_id") == wanted:
                return directory, metadata, chunk
    return None


def _paper_reference(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": metadata.get("paper_id"),
        "paper_key": metadata.get("paper_key"),
        "title": metadata.get("title"),
        "url": metadata.get("url"),
    }


def describe_chunks(
    *, memory_dir: str | Path, project_id: str, chunk_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Resolve chunk ids to their paper and section, in one pass over the store.

    Search returns ids; a caller rendering hits needs each id's paper and
    section without re-deriving them from the search summary. Requested ids are
    resolved together so a page of hits costs one traversal, not one per hit.
    """
    wanted = {value.strip() for value in chunk_ids if value.strip()}
    if not wanted:
        return {}
    root = Path(memory_dir).expanduser()
    described: dict[str, dict[str, Any]] = {}
    for directory in _project_paper_dirs(root, project_id=project_id):
        metadata = _read_metadata(directory, project_id=project_id)
        if metadata is None:
            continue
        for chunk in _load_chunks(directory):
            chunk_id = chunk.get("chunk_id")
            if not isinstance(chunk_id, str) or chunk_id not in wanted:
                continue
            described[chunk_id] = {
                "paper": _paper_reference(metadata),
                "section": chunk.get("section"),
                "section_path": chunk.get("section_path"),
                "char_start": chunk.get("char_start"),
                "char_end": chunk.get("char_end"),
            }
        if len(described) == len(wanted):
            break
    return described


def read_paper_chunk(
    *,
    memory_dir: str | Path,
    project_id: str,
    chunk_id: str,
    expand: str = "section",
    max_chars: int | None = None,
) -> dict[str, Any] | None:
    """Read one chunk, or the whole section containing it.

    ``expand="section"`` reconstructs the section by slicing ``paper.md``
    across the offsets of every chunk sharing the same ``section_path``, so an
    over-long section that was split into overlapping windows is returned once,
    whole, with no duplicated overlap.
    """
    root = Path(memory_dir).expanduser()
    located = _locate_chunk(root, project_id=project_id, chunk_id=chunk_id)
    if located is None:
        return None
    directory, metadata, chunk = located
    try:
        text = (directory / "paper.md").read_text(encoding="utf-8")
    except OSError:
        return None

    start = int(chunk.get("char_start") or 0)
    end = int(chunk.get("char_end") or 0)
    section_path = (
        chunk.get("section_path") if isinstance(chunk.get("section_path"), str) else ""
    )
    if expand == "section":
        siblings = [
            row
            for row in _load_chunks(directory)
            if row.get("section_path") == chunk.get("section_path")
        ]
        if siblings:
            start = min(int(row.get("char_start") or 0) for row in siblings)
            end = max(int(row.get("char_end") or 0) for row in siblings)

    body = text[start:end]
    truncated = False
    if max_chars is not None and len(body) > max_chars:
        body = body[:max_chars]
        truncated = True
    return {
        "chunk_id": chunk.get("chunk_id"),
        "expand": "section" if expand == "section" else "chunk",
        "paper": _paper_reference(metadata),
        "section": chunk.get("section"),
        "section_path": section_path,
        "char_start": start,
        "char_end": start + len(body) if truncated else end,
        "truncated": truncated,
        "text": body,
    }


def read_paper_full(
    *,
    memory_dir: str | Path,
    project_id: str,
    paper_id: str,
    url: str = "",
    max_chars: int | None = None,
) -> dict[str, Any] | None:
    """Read one paper's whole stored text, optionally truncated.

    Kept as a first-class path because the finding that motivated this store
    is that a whole paper outperforms its distilled experience; deep single-paper
    reading must stay available.
    """
    root = Path(memory_dir).expanduser()
    directory = paper_dir(root, project_id=project_id, paper_id=paper_id, url=url)
    metadata = _read_metadata(directory, project_id=project_id)
    if metadata is None:
        found = find_paper_directory(root, project_id=project_id, paper_id=paper_id)
        if found is None:
            return None
        directory, metadata = found
    try:
        text = (directory / "paper.md").read_text(encoding="utf-8")
    except OSError:
        return None

    total = len(text)
    truncated = max_chars is not None and total > max_chars
    if truncated:
        text = text[:max_chars]
    return {
        "expand": "full",
        "paper": _paper_reference(metadata),
        "char_count": total,
        "returned_chars": len(text),
        "truncated": truncated,
        "chunk_count": metadata.get("chunk_count"),
        "text": text,
    }


def find_paper_directory(
    root: str | Path, *, project_id: str, paper_id: str
) -> tuple[Path, dict[str, Any]] | None:
    """Resolve a paper by id when the caller has no URL to key on.

    ``paper_storage_key()`` folds paper id and URL into one canonical name, so
    a caller holding only the id (an agent quoting a search hit) may compute a
    different key than the writer did. Matching on the stored canonical id and
    the raw id covers both.
    """
    wanted = paper_id.strip()
    if not wanted:
        return None
    canonical = canonical_paper_identifier(wanted)
    for directory in _project_paper_dirs(
        Path(root).expanduser(), project_id=project_id
    ):
        metadata = _read_metadata(directory, project_id=project_id)
        if metadata is None:
            continue
        if (
            metadata.get("paper_id") == wanted
            or metadata.get("canonical_paper_id") == canonical
            or metadata.get("paper_key") == wanted
            or canonical_paper_identifier(str(metadata.get("url") or "")) == canonical
        ):
            return directory, metadata
    return None


def list_paper_projects(memory_dir: str | Path) -> list[str]:
    """List project ids that have at least one stored paper.

    Used by the debug inspector, which has no workspace to resolve a project
    from and must show whatever is actually on disk.
    """
    root = Path(memory_dir).expanduser() / PAPER_DIR
    try:
        return sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and not path.is_symlink()
        )
    except OSError:
        return []


def list_paper_chunks(
    *, memory_dir: str | Path, project_id: str, paper_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Return one paper's metadata and every stored chunk row, in order.

    Unlike ``read_paper_chunk``, this exposes the raw index rather than reading
    text by offset -- the inspector needs to show what was stored, including
    rows whose offsets might disagree with ``paper.md``.
    """
    found = find_paper_directory(
        Path(memory_dir).expanduser(), project_id=project_id, paper_id=paper_id
    )
    if found is None:
        return None
    directory, metadata = found
    chunks = sorted(
        _load_chunks(directory), key=lambda row: int(row.get("chunk_index") or 0)
    )
    return metadata, chunks


def list_papers(*, memory_dir: str | Path, project_id: str) -> list[dict[str, Any]]:
    """List paper-level metadata for one project, newest chunking first.

    Feeds the compact ``<paper_fulltext_memory>`` index block: one line per
    paper, never one per chunk.
    """
    root = Path(memory_dir).expanduser()
    papers: list[dict[str, Any]] = []
    for directory in _project_paper_dirs(root, project_id=project_id):
        metadata = _read_metadata(directory, project_id=project_id)
        if metadata is None or not (directory / "paper.md").is_file():
            continue
        papers.append(metadata)
    papers.sort(key=lambda item: str(item.get("title") or item.get("paper_id") or ""))
    return papers
