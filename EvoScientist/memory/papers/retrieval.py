"""Retrieval entry point for stored paper full text.

A single choke point on purpose: the whole store is behind one function, so
swapping lexical TF-IDF for embeddings or a hybrid ranker later touches this
file and nothing else.

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
    return search_documents(
        documents=list_paper_chunk_documents(
            memory_dir=memory_dir,
            project_id=project_id,
            paper_id=paper_id,
        ),
        query=query_text,
        limit=limit,
        mode=ObservationSearchMode(mode),
    )
