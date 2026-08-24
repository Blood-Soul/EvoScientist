"""Aggregation of isolated experience records into EvoMemory retrieval."""

from __future__ import annotations

from pathlib import Path

from ..search import search_documents
from ..types import (
    MemoryScope,
    MemoryType,
    ObservationReadResult,
    ObservationSearchDocument,
    ObservationSearchHit,
    ObservationSearchMode,
)
from .store import list_experience_documents, read_experience_file


def list_memory_documents(
    *,
    memory_dir: str | Path,
    project_id: str,
    scope: MemoryScope | None = None,
    memory_type: MemoryType | None = None,
) -> list[ObservationSearchDocument]:
    """List observations plus project experiences without merging their stores."""
    from ..observations.store import list_observation_documents

    documents = list_observation_documents(
        memory_dir=memory_dir,
        project_id=project_id,
        scope=scope,
        memory_type=memory_type,
    )
    if scope == MemoryScope.GLOBAL:
        return documents
    if memory_type not in {None, MemoryType.SEMANTIC}:
        return documents
    return [
        *documents,
        *list_experience_documents(memory_dir=memory_dir, project_id=project_id),
    ]


def search_memory_files(
    *,
    memory_dir: str | Path,
    project_id: str,
    query: str,
    scope: MemoryScope | None = None,
    memory_type: MemoryType | None = None,
    limit: int = 8,
    mode: ObservationSearchMode = ObservationSearchMode.RANKED,
) -> list[ObservationSearchHit]:
    query_text = query.strip()
    if not query_text:
        return []
    return search_documents(
        documents=list_memory_documents(
            memory_dir=memory_dir,
            project_id=project_id,
            scope=scope,
            memory_type=memory_type,
        ),
        query=query_text,
        limit=limit,
        mode=ObservationSearchMode(mode),
    )


def read_memory_file(
    *, memory_dir: str | Path, project_id: str, record_id: str
) -> ObservationReadResult | None:
    """Read either an observation or a project experience by stable ID."""
    from ..observations.store import read_observation_file

    requested = record_id.strip()
    if requested.startswith("E-"):
        return read_experience_file(
            memory_dir=memory_dir,
            project_id=project_id,
            experience_id=requested,
        )
    return read_observation_file(
        memory_dir=memory_dir,
        project_id=project_id,
        observation_id=requested,
    )
