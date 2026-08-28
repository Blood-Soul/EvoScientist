"""The single persistence entry point used by both extraction call sites.

The background worker and the foreground `extract_paper_experiences` tool both
hold a paper's downloaded text and must persist it under the same rules:
honour the feature switch, use the configured chunk geometry, and never let a
storage failure fail the extraction that was actually asked for. Keeping those
rules here means the two call sites cannot drift apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .chunking import DEFAULT_MAX_CHUNK_CHARS, DEFAULT_OVERLAP_CHARS
from .store import store_paper_text

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperFulltextSettings:
    """Resolved full-text storage switches and chunk geometry."""

    enabled: bool
    max_chunk_chars: int
    overlap_chars: int


_cached_settings: PaperFulltextSettings | None = None


def paper_fulltext_settings() -> PaperFulltextSettings:
    """Resolve full-text settings once per process.

    Mirrors ``observations/store._max_cached_files()``: read lazily on first
    use and cached at module level, so a config change needs a restart.
    """
    global _cached_settings
    if _cached_settings is None:
        try:
            from ...config import get_effective_config

            config = get_effective_config()
            _cached_settings = PaperFulltextSettings(
                enabled=bool(config.memory_paper_fulltext_enabled),
                max_chunk_chars=int(config.memory_paper_chunk_max_chars),
                overlap_chars=int(config.memory_paper_chunk_overlap_chars),
            )
        except Exception:  # pragma: no cover - defensive, config must not block storage
            _logger.debug(
                "Falling back to default paper full-text settings", exc_info=True
            )
            _cached_settings = PaperFulltextSettings(
                enabled=True,
                max_chunk_chars=DEFAULT_MAX_CHUNK_CHARS,
                overlap_chars=DEFAULT_OVERLAP_CHARS,
            )
    return _cached_settings


def reset_paper_fulltext_settings_cache() -> None:
    """Drop the cached settings so a test can vary configuration."""
    global _cached_settings
    _cached_settings = None


def persist_paper_fulltext(
    *,
    memory_dir: str | Path,
    project_id: str,
    paper_id: str,
    url: str,
    title: str,
    paper_text: str,
    domain_arxiv: str | None = None,
) -> Path | None:
    """Persist one paper's full text, returning None when it was not stored.

    Never raises. Full text is a complement to experience extraction, not a
    precondition for it, so a disabled switch or a write failure degrades to
    "no full text for this paper" rather than failing the queued task.
    """
    settings = paper_fulltext_settings()
    if not settings.enabled:
        return None
    try:
        return store_paper_text(
            memory_dir=memory_dir,
            project_id=project_id,
            paper_id=paper_id,
            url=url,
            title=title,
            paper_text=paper_text,
            domain_arxiv=domain_arxiv,
            max_chunk_chars=settings.max_chunk_chars,
            overlap_chars=settings.overlap_chars,
        )
    except Exception as exc:
        _logger.warning("Failed to persist paper full text for %s: %s", paper_id, exc)
        return None
