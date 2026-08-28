"""Prompt-facing observation memory indexes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from ..experiences.store import list_experience_documents
from ..types import MemoryScope, MemoryType, ObservationSearchDocument
from .store import list_observation_documents

DEFAULT_MAX_INLINE_OBSERVATION_INDEX_CHARS = 12_000


def build_observation_index_context(
    *,
    memory_dir: str | Path,
    project_id: str,
    max_inline_chars: int = DEFAULT_MAX_INLINE_OBSERVATION_INDEX_CHARS,
) -> str:
    """Build a compact observation-memory index for prompts."""
    observation_context = _format_observation_index_context(
        _observation_documents(memory_dir=memory_dir, project_id=project_id),
        include_counts=True,
        include_paths=True,
        include_search_hints=True,
        empty_context=True,
        intro="Indexed observations:",
        max_inline_chars=max_inline_chars,
    )
    experiences = sorted(
        list_experience_documents(memory_dir=memory_dir, project_id=project_id),
        key=lambda document: document.observation_id,
    )
    if not experiences:
        # Full text can exist without experiences -- extraction is persisted
        # after the text and may have failed -- so still offer the paper block.
        fulltext_only = _build_paper_fulltext_index(
            memory_dir=memory_dir,
            project_id=project_id,
            remaining=max_inline_chars - len(observation_context) - 2,
        )
        if not fulltext_only:
            return observation_context
        return f"{observation_context}\n\n{fulltext_only}"
    experience_lines = [
        (
            f"- {document.observation_id} "
            f"[{document.experience_level}/project] {document.path}: "
            f"{document.summary}"
        )
        for document in experiences
    ]
    remaining = max_inline_chars - len(observation_context) - 2
    prefix = [
        "<paper_experience_memory>",
        f"Indexed project paper experiences: total={len(experiences)}.",
    ]
    suffix = [
        "Use `search_observations` and `read_memory` with E-* IDs.",
        "</paper_experience_memory>",
    ]
    experience_context = _fit_block(
        prefix=prefix,
        lines=experience_lines,
        suffix=suffix,
        remaining=remaining,
        truncation_note="Experience index truncated to entries that fit.",
    )
    if not experience_context:
        return observation_context
    combined = f"{observation_context}\n\n{experience_context}"
    fulltext_context = _build_paper_fulltext_index(
        memory_dir=memory_dir,
        project_id=project_id,
        remaining=max_inline_chars - len(combined) - 2,
    )
    if not fulltext_context:
        return combined
    return f"{combined}\n\n{fulltext_context}"


def _fit_block(
    *,
    prefix: list[str],
    lines: list[str],
    suffix: list[str],
    remaining: int,
    truncation_note: str,
) -> str:
    """Render a bounded index block, or "" when even its framing will not fit.

    The truncation note is part of the prefix while lines are selected, not
    appended afterwards: adding it later would push an already-exactly-fitting
    block past ``remaining``.
    """
    if remaining <= 0:
        return ""
    fits = _fit_lines(prefix=prefix, lines=lines, suffix=suffix, remaining=remaining)
    if fits is None:
        return ""
    selected = fits
    if len(selected) == len(lines):
        return "\n".join([*prefix, *selected, *suffix])

    noted_prefix = [*prefix, truncation_note]
    fits = _fit_lines(
        prefix=noted_prefix, lines=lines, suffix=suffix, remaining=remaining
    )
    if fits is None:
        # The note itself does not fit; keep the entries and drop the note
        # rather than dropping the block.
        return "\n".join([*prefix, *selected, *suffix])
    return "\n".join([*noted_prefix, *fits, *suffix])


def _fit_lines(
    *,
    prefix: list[str],
    lines: list[str],
    suffix: list[str],
    remaining: int,
) -> list[str] | None:
    """Select the lines that fit, or None when the framing alone is too large."""
    if len("\n".join([*prefix, *suffix])) > remaining:
        return None
    selected: list[str] = []
    for line in lines:
        candidate = "\n".join([*prefix, *selected, line, *suffix])
        if len(candidate) <= remaining:
            selected.append(line)
    return selected


def _build_paper_fulltext_index(
    *, memory_dir: str | Path, project_id: str, remaining: int
) -> str:
    """Build the paper-level full-text index block.

    Deliberately one line per *paper*, never per chunk: a single paper yields
    tens of chunks, so listing chunks would exhaust the shared inline budget
    after a few papers. Papers are named here; passages are found by searching.
    """
    from ..papers.store import list_papers

    papers = list_papers(memory_dir=memory_dir, project_id=project_id)
    if not papers or remaining <= 0:
        return ""
    lines = [
        (
            f"- {paper.get('paper_id') or paper.get('paper_key')}: "
            f"{' '.join(str(paper.get('title') or 'untitled').split())} "
            f"({paper.get('chunk_count', 0)} passages, "
            f"{paper.get('section_count', 0)} sections)"
        )
        for paper in papers
    ]
    prefix = [
        "<paper_fulltext_memory>",
        f"Papers with stored full text: total={len(papers)}.",
    ]
    suffix = [
        (
            "Search passages with `search_paper_text`, then read them with "
            "`read_paper`. Experiences (E-*) give judgements; this text gives "
            "the evidence, numbers, and wording behind them."
        ),
        "</paper_fulltext_memory>",
    ]
    return _fit_block(
        prefix=prefix,
        lines=lines,
        suffix=suffix,
        remaining=remaining,
        truncation_note="Paper index truncated to entries that fit.",
    )


def build_observation_linker_index_context(
    *,
    memory_dir: str | Path,
    project_id: str,
    exclude_ids: Iterable[str],
    max_inline_chars: int = DEFAULT_MAX_INLINE_OBSERVATION_INDEX_CHARS,
) -> str:
    """Build the existing-observation index included in linker launches."""
    return _format_observation_index_context(
        _observation_documents(
            memory_dir=memory_dir,
            project_id=project_id,
            exclude_ids=exclude_ids,
        ),
        include_counts=False,
        include_paths=False,
        include_search_hints=False,
        empty_context=False,
        intro=(
            "Stored observation snapshot excluding the current batch "
            "(id [type/scope]: summary). Read before linking when needed."
        ),
        max_inline_chars=max_inline_chars,
    )


def _observation_documents(
    *,
    memory_dir: str | Path,
    project_id: str,
    exclude_ids: Iterable[str] = (),
) -> list[ObservationSearchDocument]:
    excluded = set(exclude_ids)
    return sorted(
        (
            document
            for document in list_observation_documents(
                memory_dir=memory_dir,
                project_id=project_id,
            )
            if document.observation_id not in excluded
        ),
        key=lambda document: document.observation_id,
    )


def _format_observation_index_context(
    documents: Sequence[ObservationSearchDocument],
    *,
    include_counts: bool = True,
    include_paths: bool = True,
    include_search_hints: bool = True,
    empty_context: bool = True,
    intro: str = "Indexed observations:",
    max_inline_chars: int = DEFAULT_MAX_INLINE_OBSERVATION_INDEX_CHARS,
) -> str:
    """Format parsed observation documents as a prompt index."""
    if not documents and not empty_context:
        return ""

    header = ["<observation_memory>"]
    if include_counts:
        header.append(_observation_index_count_line(documents))

    footer = [_observation_search_hints()] if include_search_hints else []
    if not documents:
        return "\n".join([*header, *footer, "</observation_memory>"])

    lines = [
        _observation_index_line(document, include_paths=include_paths)
        for document in documents
    ]
    full = "\n".join(
        [
            *header,
            intro,
            *lines,
            *footer,
            "</observation_memory>",
        ]
    )
    if len(full) <= max_inline_chars:
        return full

    return _truncated_observation_index_context(
        header=header,
        intro=intro,
        lines=lines,
        footer=footer,
        max_inline_chars=max_inline_chars,
    )


def _truncated_observation_index_context(
    *,
    header: Sequence[str],
    intro: str,
    lines: Sequence[str],
    footer: Sequence[str],
    max_inline_chars: int,
) -> str:
    prefix = [
        *header,
        "Observation index truncated to entries that fit.",
        intro,
    ]
    suffix = [*footer, "</observation_memory>"]
    selected: list[str] = []
    for line in lines:
        candidate = "\n".join([*prefix, *selected, line, *suffix])
        if len(candidate) <= max_inline_chars:
            selected.append(line)
    if selected:
        return "\n".join([*prefix, *selected, *suffix])

    return "\n".join(
        [
            *header,
            "Observation summaries are too large to inline; search on demand.",
            *footer,
            "</observation_memory>",
        ]
    )


def _observation_index_line(
    document: ObservationSearchDocument,
    *,
    include_paths: bool,
) -> str:
    typed_scope = f"[{document.memory_type.value}/{document.scope.value}]"
    if include_paths:
        return (
            f"- {document.observation_id} {typed_scope} "
            f"{document.path}: {document.summary}"
        )
    return f"- {document.observation_id} {typed_scope}: {document.summary}"


def _observation_index_count_line(
    documents: Sequence[ObservationSearchDocument],
) -> str:
    """Return compact observation counts by scope and memory type."""
    scope_counts = dict.fromkeys(MemoryScope, 0)
    type_counts = dict.fromkeys(MemoryType, 0)
    for document in documents:
        scope_counts[document.scope] += 1
        type_counts[document.memory_type] += 1
    return (
        f"Counts: total={len(documents)}; "
        f"scope global={scope_counts[MemoryScope.GLOBAL]}, "
        f"project={scope_counts[MemoryScope.PROJECT]}; "
        f"type semantic={type_counts[MemoryType.SEMANTIC]}, "
        f"procedural={type_counts[MemoryType.PROCEDURAL]}, "
        f"episodic={type_counts[MemoryType.EPISODIC]}."
    )


def _observation_search_hints() -> str:
    """Return stable search hints for observation memory."""
    return "\n".join(
        [
            "Search hints:",
            "- Each line gives id, type/scope, path, and summary.",
            (
                "- Use `search_observations` for ranked keyword search "
                "and `read_memory` for known observation IDs."
            ),
            "- Use `mode=regex` only when exact grep-like matching is required.",
            "- Search by id when you already know it from the index.",
            (
                "- Filter by type when appropriate: "
                "`memory_type: procedural`, `memory_type: semantic`, or "
                "`memory_type: episodic`."
            ),
            (
                "- Filter by scope when appropriate: "
                "`scope: project` or `scope: global`."
            ),
            (
                "- Search with a few distinctive words or phrases from "
                "the current work that describe the issue, constraint, "
                "procedure, or prior result to find."
            ),
        ]
    )
