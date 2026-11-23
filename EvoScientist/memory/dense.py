"""Dense ranking over search documents, fused with the lexical ranking.

`search_experience` and `search_paper_text` (and plan review, which retrieves through the former) rank lexically:
a query only reaches a record that shares its words. With the embedding service configured (:mod:`.embedding`), each
query is also ranked by cosine between its Qwen3 query embedding and the documents' embeddings, and the two rankings
are fused by reciprocal rank (RRF): a record that either ranking puts high surfaces, and agreement promotes it.
Without the service, or when it fails, the lexical ranking is returned unchanged.

Document vectors are cached by text (see :mod:`.embedding`), so a library is embedded once and a query costs one
embedding call plus a matrix product.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from . import embedding
from .types import ObservationSearchDocument, ObservationSearchHit

RRF_K = 60  # as in experiences.retrieval
DENSE_DEPTH = 4  # the dense ranking contributes this many times `limit` candidates
MIN_COSINE = 0.2  # below this a dense match is noise for Qwen3 (unrelated pairs sit near 0.3-0.4 at most)


def document_text(document: ObservationSearchDocument) -> str:
    return (document.text or document.body or document.summary or "").strip()


_memo: dict[tuple[str, str], np.ndarray] = {}  # (cache file, digest of the texts) -> matrix; the last few only


def _matrix(documents: Sequence[ObservationSearchDocument], memory_dir: str | Path) -> np.ndarray | None:
    """The documents' vectors. The store hands back the same documents across queries, so the stacked matrix is
    kept for the last few document sets: a query then costs one query embedding and one matrix product."""
    import hashlib

    texts = [document_text(doc) for doc in documents]
    key = (str(embedding._cache_path(memory_dir)), hashlib.sha256("\x00".join(texts).encode()).hexdigest())
    if key in _memo:
        return _memo[key]
    m = embedding.embed_documents(texts, memory_dir=memory_dir)
    if m is not None:
        if len(_memo) >= 8:
            _memo.pop(next(iter(_memo)))
        _memo[key] = m
    return m


def dense_rank(
    documents: Sequence[ObservationSearchDocument], query: str, *, task: str, memory_dir: str | Path, limit: int
) -> list[tuple[ObservationSearchDocument, float]] | None:
    """The `limit` documents nearest `query`, with their cosines, or None without embeddings."""
    if not documents or not query.strip() or not embedding.configured():
        return None
    q = embedding.embed_queries([query.strip()], task=task, memory_dir=memory_dir)
    if q is None:
        return None
    d = _matrix(documents, memory_dir)
    if d is None:
        return None
    scores = d @ q[0]
    order = np.argsort(-scores)[:limit]
    return [(documents[i], float(scores[i])) for i in order if scores[i] >= MIN_COSINE]


def fuse(
    lexical: list[ObservationSearchHit],
    dense: list[tuple[ObservationSearchDocument, float]] | None,
    *,
    limit: int,
    make_hit: Callable[[ObservationSearchDocument, float], ObservationSearchHit],
) -> list[ObservationSearchHit]:
    """Reciprocal-rank fusion of the lexical hits and the dense ranking. The lexical hits alone without dense."""
    if not dense:
        return lexical[:limit]
    scores: dict[str, float] = {}
    best: dict[str, ObservationSearchHit] = {}
    for rank, hit in enumerate(lexical):
        key = hit["observation_id"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        best.setdefault(key, hit)
    for rank, (doc, cos) in enumerate(dense):
        key = doc.observation_id
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        if key not in best:
            best[key] = make_hit(doc, cos)
    ordered = sorted(best, key=lambda k: (-scores[k], k))[:limit]
    out: list[ObservationSearchHit] = []
    for key in ordered:
        hit = dict(best[key])
        hit["score"] = round(scores[key], 5)
        out.append(hit)  # type: ignore[arg-type]
    return out


def hit_from_document(document: ObservationSearchDocument, cosine: float) -> ObservationSearchHit:
    """A hit for a document only the dense ranking found: no matching lines, the summary and an opening excerpt."""
    body = " ".join((document.body or "").split())
    hit: ObservationSearchHit = {
        "observation_id": document.observation_id,
        "path": document.path,
        "memory_type": document.memory_type,
        "scope": document.scope,
        "summary": document.summary,
        "matches": [body[:240]] if body else [],
        "score": round(cosine, 3),
    }
    if document.record_kind != "observation":
        hit["record_kind"] = document.record_kind
    if document.experience_level is not None:
        hit["experience_level"] = document.experience_level
    return hit


def hybrid(
    documents: list[ObservationSearchDocument],
    query: str,
    lexical: list[ObservationSearchHit],
    *,
    task: str,
    memory_dir: str | Path,
    limit: int,
) -> list[ObservationSearchHit]:
    """The lexical hits fused with the dense ranking of the same documents (the lexical hits alone without it)."""
    dense = dense_rank(documents, query, task=task, memory_dir=memory_dir, limit=max(limit * DENSE_DEPTH, limit))
    return fuse(lexical, dense, limit=limit, make_hit=hit_from_document)
