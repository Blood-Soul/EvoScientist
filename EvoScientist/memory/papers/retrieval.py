"""Retrieval entry point for stored paper full text.

A single choke point on purpose: the whole store is behind one function. The
ranking is lexical, fused with an embedding ranking when the embedding service
is configured (see ``memory.dense``).

Chunks are searched *here*, never through the observation or experience
retrievers. One paper yields tens of chunks and a project holds dozens of
papers, so folding them into either ranking would let raw text crowd out the
distilled ``O-*`` and ``E-*`` records those entry points exist to return.
"""

from __future__ import annotations

from pathlib import Path

from ..search import search_documents
from ..types import ObservationSearchHit, ObservationSearchMode
from .store import list_paper_chunk_documents


def search_paper_chunks(
    *,
    memory_dir: str | Path,
    project_id: str,
    query: str,
    limit: int = 8,
    mode: ObservationSearchMode = ObservationSearchMode.RANKED,
    paper_id: str | None = None,
) -> list[ObservationSearchHit]:
    """Rank stored paper chunks against ``query`` within one project."""
    query_text = query.strip()
    if not query_text:
        return []
    documents = list_paper_chunk_documents(
        memory_dir=memory_dir,
        project_id=project_id,
        paper_id=paper_id,
    )
    search_mode = ObservationSearchMode(mode)
    lexical = search_documents(documents=documents, query=query_text, limit=limit, mode=search_mode)
    if search_mode == ObservationSearchMode.REGEX:
        return lexical
    # With the embedding service configured, fused with the embedding ranking (memory.dense); lexical otherwise.
    from ..dense import hybrid
    from ..embedding import RETRIEVE_PASSAGE

    return hybrid(documents, query_text, lexical, task=RETRIEVE_PASSAGE, memory_dir=memory_dir, limit=limit)
