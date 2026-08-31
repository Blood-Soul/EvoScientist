"""Shared types for EvoMemory observation storage and search."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, NotRequired, TypedDict

ExperienceLevel = Literal["l1", "l2"]

# What a search document represents. "paper_chunk" records are addressed by
# `C-*` ids and are searched by the dedicated paper full-text tools, never by
# `search_observations` -- mixing thousands of chunks into that ranking would
# bury the `E-*` and `O-*` records it exists to surface.
MemoryRecordKind = Literal["observation", "experience", "paper_chunk"]


class MemoryType(StrEnum):
    """Kinds of reusable memory an observation can represent."""

    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"


class MemoryScope(StrEnum):
    """Whether an observation is global or tied to the active project."""

    GLOBAL = "global"
    PROJECT = "project"


class MemorySourceType(StrEnum):
    """Where a memory observation originated."""

    SUBAGENT = "subagent"
    TURN = "turn"


class ObservationSearchMode(StrEnum):
    """Search modes supported by `search_observations`."""

    RANKED = "ranked"
    REGEX = "regex"


class ObservationRelation(StrEnum):
    """Allowed relationship labels between observations."""

    COMPLEMENTS = "complements"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"


class ObservationRecordResult(TypedDict):
    """Result returned by `record_observation`."""

    observation_id: str
    path: str
    created: bool
    memory_type: MemoryType
    scope: MemoryScope
    project_id: NotRequired[str]


class RelatedObservationResult(TypedDict):
    """One resolved observation relationship exposed to memory tools."""

    observation_id: str
    path: str
    memory_type: MemoryType
    scope: MemoryScope
    summary: str
    relation: NotRequired[ObservationRelation]
    reason: NotRequired[str]


@dataclass(frozen=True)
class DocumentTokenIndex:
    """One document's tokens, split by the field they came from.

    Ranking asks two questions per query token -- "is it in this field" and
    "how rare is it in the corpus" -- and both need the same per-field token
    sets. Deriving them is regex work over the document's full text, so they
    are computed once and reused for every later query against the same
    document. `all_tokens` is the union, used for corpus IDF.
    """

    id_tokens: frozenset[str]
    summary_tokens: frozenset[str]
    body_tokens: frozenset[str]
    metadata_tokens: frozenset[str]
    all_tokens: frozenset[str]


@dataclass(frozen=True)
class ObservationSearchDocument:
    """Parsed observation document ready for search."""

    observation_id: str
    path: str
    memory_type: MemoryType
    scope: MemoryScope
    summary: str
    body: str
    text: str
    related_observations: tuple[RelatedObservationResult, ...] = ()
    record_kind: MemoryRecordKind = "observation"
    experience_level: ExperienceLevel | None = None
    # Facets carried only by `E-*` records. They are structured metadata, not
    # ranking text, and exist so experience retrieval can narrow by structure
    # before ranking -- a caller who does not know the library's vocabulary
    # cannot phrase a lexical query that reaches it.
    discipline: str | None = None
    domain: str | None = None
    task: str | None = None
    paper_key: str | None = None
    # Lazily filled by `search.document_token_index`. Tokenizing is the
    # dominant cost of a ranked query and it runs over the whole corpus, once
    # per facet pass; the store-level parse caches hand back the same document
    # objects across queries, so the tokens only have to be derived once.
    # Excluded from `__init__`, equality and `repr`: it is a memo of the other
    # fields rather than part of the record.
    token_index: DocumentTokenIndex | None = field(
        default=None, init=False, compare=False, repr=False
    )


class ObservationSearchHit(TypedDict):
    """One result returned by `search_observations`."""

    observation_id: str
    path: str
    memory_type: MemoryType
    scope: MemoryScope
    summary: str
    matches: list[str]
    record_kind: NotRequired[MemoryRecordKind]
    experience_level: NotRequired[ExperienceLevel]
    related_observations: NotRequired[list[RelatedObservationResult]]
    score: NotRequired[float]


class ObservationReadResult(TypedDict):
    """Full observation document returned by `read_memory`."""

    observation_id: str
    path: str
    memory_type: MemoryType
    scope: MemoryScope
    summary: str
    text: str
    record_kind: NotRequired[MemoryRecordKind]
    experience_level: NotRequired[ExperienceLevel]
    related_observations: NotRequired[list[RelatedObservationResult]]
