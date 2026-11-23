"""Text embeddings for clustering L1 records and for retrieval, from one embedding service.

The model is Qwen3-Embedding-0.6B by default, served by `scripts/embedding_server.py` (an OpenAI-compatible
`/v1/embeddings` endpoint) or by any server that speaks that API (vLLM, TEI, a hosted endpoint). This module is the
client: it batches requests, keeps every vector it has computed in a cache on disk, and never raises to its callers.

Configuration (environment, so the experiment machine and the Mac can differ without a config change):

- ``EVOSCIENTIST_EMBEDDING_URL``: the service's base URL, for example ``http://127.0.0.1:18765/v1``. Unset means no
  embeddings: clustering and retrieval stay lexical, exactly as before.
- ``EVOSCIENTIST_EMBEDDING_MODEL``: the model name sent to the service (default ``Qwen/Qwen3-Embedding-0.6B``).
  It also keys the cache, so vectors of two models never mix.
- ``EVOSCIENTIST_EMBEDDING_API_KEY``: sent as a bearer token when set.

Qwen3-Embedding is asymmetric: a query is embedded with a one-line instruction in front of it, a document without.
:func:`embed_queries` and :func:`embed_documents` do that; vectors come back L2-normalised, so a dot product is the
cosine.

The cache is one SQLite file per model under the memory directory (``embeddings/<model>.sqlite``), keyed by the
SHA-256 of the exact text sent (instruction included). A record's vector is computed once, whichever path asks for it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

URL_ENV = "EVOSCIENTIST_EMBEDDING_URL"
MODEL_ENV = "EVOSCIENTIST_EMBEDDING_MODEL"
KEY_ENV = "EVOSCIENTIST_EMBEDDING_API_KEY"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# The instruction Qwen3-Embedding expects in front of a query (its model card: "Instruct: {task}\nQuery:{query}").
# One task per use, written for what the query looks for.
RETRIEVE_EXPERIENCE = (
    "Given a research question or a design decision, retrieve experience records from papers that bear on it"
)
RETRIEVE_PASSAGE = "Given a question about a paper, retrieve the passages of the paper that answer it"

BATCH = 32
TIMEOUT_S = 120.0
MAX_CHARS = 8000  # longer text is cut: a record or a chunk is far shorter, and the cost grows with length
RETRY_AFTER_S = 60.0  # after a failed call the service is not asked again for this long

_lock = threading.Lock()
_down_until = 0.0
_warned = False


def configured() -> bool:
    return bool(os.environ.get(URL_ENV, "").strip())


def model_name() -> str:
    return os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL


def query_text(query: str, task: str) -> str:
    return f"Instruct: {task}\nQuery:{query}"


# ----------------------------------------------------------------------------- the cache


def _cache_path(memory_dir: str | Path) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name())
    return Path(memory_dir).expanduser() / "embeddings" / f"{safe}.sqlite"


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Cache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS vec (k TEXT PRIMARY KEY, dim INTEGER, v BLOB)")
        self.db.commit()

    def get(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):
            part = list(keys[i : i + 500])
            rows = self.db.execute(
                f"SELECT k, v FROM vec WHERE k IN ({','.join('?' * len(part))})", part
            ).fetchall()
            out.update((k, np.frombuffer(v, dtype=np.float32)) for k, v in rows)
        return out

    def put(self, items: dict[str, np.ndarray]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO vec (k, dim, v) VALUES (?, ?, ?)",
            [(k, int(v.shape[0]), v.astype(np.float32).tobytes()) for k, v in items.items()],
        )
        self.db.commit()


_caches: dict[Path, _Cache] = {}


def _cache(memory_dir: str | Path) -> _Cache:
    path = _cache_path(memory_dir)
    with _lock:
        if path not in _caches:
            _caches[path] = _Cache(path)
        return _caches[path]


# ----------------------------------------------------------------------------- the service


def _request(texts: list[str]) -> list[list[float]]:
    import httpx

    base = os.environ[URL_ENV].strip().rstrip("/")
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get(KEY_ENV, "").strip():
        headers["Authorization"] = f"Bearer {key}"
    resp = httpx.post(f"{base}/embeddings", json={"model": model_name(), "input": texts}, headers=headers,
                      timeout=TIMEOUT_S)
    resp.raise_for_status()
    data = sorted(resp.json()["data"], key=lambda d: d["index"])
    if len(data) != len(texts):
        raise ValueError(f"asked for {len(texts)} embeddings, got {len(data)}")
    return [d["embedding"] for d in data]


def _unit(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


def embed(texts: Sequence[str], *, memory_dir: str | Path) -> np.ndarray | None:
    """Unit vectors for `texts` (one row each), or None when no service is configured or it failed.

    Cached vectors are reused; only the texts not seen before are sent. Never raises.
    """
    global _down_until, _warned
    if not configured() or not texts:
        return None
    if time.monotonic() < _down_until:
        return None
    sent = [t[:MAX_CHARS] for t in texts]
    keys = [_key(t) for t in sent]
    try:
        cache = _cache(memory_dir)
        have = cache.get(list(dict.fromkeys(keys)))
        todo = list(dict.fromkeys(k for k in keys if k not in have))
        text_of = dict(zip(keys, sent, strict=True))
        for i in range(0, len(todo), BATCH):
            part = todo[i : i + BATCH]
            vecs = _unit(np.asarray(_request([text_of[k] for k in part]), dtype=np.float32))
            new = dict(zip(part, vecs, strict=True))
            cache.put(new)
            have.update(new)
        return _unit(np.stack([have[k] for k in keys]))
    except Exception as exc:  # retrieval and clustering fall back to lexical; the run goes on
        _down_until = time.monotonic() + RETRY_AFTER_S
        if not _warned:
            logger.warning("embedding service unavailable (%s); using lexical ranking", exc)
            _warned = True
        else:
            logger.debug("embedding call failed: %s", exc)
        return None


def embed_documents(texts: Sequence[str], *, memory_dir: str | Path) -> np.ndarray | None:
    return embed(texts, memory_dir=memory_dir)


def embed_queries(queries: Sequence[str], *, task: str, memory_dir: str | Path) -> np.ndarray | None:
    return embed([query_text(q, task) for q in queries], memory_dir=memory_dir)


def _reset_for_tests() -> None:
    global _down_until, _warned
    with _lock:
        for c in _caches.values():
            c.db.close()
        _caches.clear()
    _down_until, _warned = 0.0, False


__all__ = [
    "DEFAULT_MODEL",
    "RETRIEVE_EXPERIENCE",
    "RETRIEVE_PASSAGE",
    "configured",
    "embed",
    "embed_documents",
    "embed_queries",
    "model_name",
    "query_text",
]
