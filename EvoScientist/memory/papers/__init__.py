"""Project-scoped storage and retrieval of paper full text for RAG."""

from .chunking import (
    CHUNKING_VERSION,
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_OVERLAP_CHARS,
    MIN_CHUNK_CHARS,
    PaperChunk,
    chunk_id_for,
    chunk_paper_text,
)
from .persist import (
    PaperFulltextSettings,
    paper_fulltext_settings,
    persist_paper_fulltext,
    reset_paper_fulltext_settings_cache,
)
from .retrieval import search_paper_chunks
from .store import (
    PAPER_DIR,
    describe_chunks,
    find_paper_directory,
    has_paper_text,
    list_paper_chunk_documents,
    list_paper_chunks,
    list_paper_projects,
    list_papers,
    load_paper_text,
    paper_dir,
    paper_dir_for_key,
    read_paper_chunk,
    read_paper_full,
    store_paper_text,
)

__all__ = [
    "CHUNKING_VERSION",
    "DEFAULT_MAX_CHUNK_CHARS",
    "DEFAULT_OVERLAP_CHARS",
    "MIN_CHUNK_CHARS",
    "PAPER_DIR",
    "PaperChunk",
    "PaperFulltextSettings",
    "chunk_id_for",
    "chunk_paper_text",
    "describe_chunks",
    "find_paper_directory",
    "has_paper_text",
    "list_paper_chunk_documents",
    "list_paper_chunks",
    "list_paper_projects",
    "list_papers",
    "load_paper_text",
    "paper_dir",
    "paper_dir_for_key",
    "paper_fulltext_settings",
    "persist_paper_fulltext",
    "read_paper_chunk",
    "read_paper_full",
    "reset_paper_fulltext_settings_cache",
    "search_paper_chunks",
    "store_paper_text",
]
