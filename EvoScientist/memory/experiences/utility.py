"""Specificity, abstractness and utility of an experience record, against a fixed set of reference tasks.

For a record e and reference tasks t_1..t_N, with c_i = cos(E(e), E(t_i)) from the embedding service:

* **Specificity** S(e) = the share of task pairs (i != j) whose cosine cos(E(t_i), E(t_j)) is at most max_i c_i.
  Does the record come at least as close to some task as tasks come to each other?
* **Abstractness** A(e) = exp(H(p)) / N, with p = softmax(c / tau) and H its entropy (tau = 0.1). The effective
  number of tasks the record is about, as a share of N: low when it bears on one task, high when it bears on many.
* **Utility** U(e) = S(e) * A(e).

Each score depends on the record and on the reference tasks only, never on the other records in the library. So
the library can grow and change without anything being recomputed: a record is scored once, when it is first seen,
and its score stays valid as long as the reference set and the embedding model do not change. Both are part of the
score's key (``ref``), so changing either rescores every record and never mixes scores of two references.

The reference set is a file of tasks (JSONL: ``{"id", "text", ...}``). The default is the packaged
``utility_tasks/v1.jsonl`` (the 20 AIRS tasks and 60 CIE research goals judgelab used); a deployment can point
``EVOSCIENTIST_UTILITY_TASKS`` at its own. Its task embeddings and the sorted task-pair cosines are computed once and
kept with the scores.

The scores are written to ``experiences/projects/<P>/_utility/scores.json`` as ``{record id: {S, A, U, c_max,
nearest}}``. This is a measurement only: nothing in retrieval or clustering reads it. On the judgelab library
(7,334 records) S and A correlate at -0.70, U ranked L1 slightly above L2 although L2 was measured more useful, and U
did not track the plan-chain utility labels (Spearman -0.18 to -0.04; those labels are themselves unreliable). See
docs/cross_paper_l2.zh-CN.md §11.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .._atomic import atomic_write_json, read_json
from .store import EXPERIENCE_DIR

logger = logging.getLogger(__name__)

TAU = 0.1
TASKS_ENV = "EVOSCIENTIST_UTILITY_TASKS"
DEFAULT_TASKS = Path(__file__).parent / "utility_tasks" / "v1.jsonl"
UTILITY_DIR = "_utility"
MAX_TASK_CHARS = 6000


# ----------------------------------------------------------------------------- the reference


def tasks_path() -> Path:
    raw = os.environ.get(TASKS_ENV, "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_TASKS


@lru_cache(maxsize=4)
def _load_tasks(path: str, mtime: float) -> tuple[tuple[str, str], ...]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out.append((str(row["id"]), str(row["text"])[:MAX_TASK_CHARS]))
    return tuple(out)


def load_tasks(path: Path | None = None) -> list[tuple[str, str]]:
    """The reference tasks as (id, text)."""
    p = path or tasks_path()
    return list(_load_tasks(str(p), p.stat().st_mtime))


def reference_key(tasks: Sequence[tuple[str, str]], model: str) -> str:
    """What a score was computed against: the reference tasks' texts and the embedding model."""
    h = hashlib.sha256(model.encode())
    for tid, text in tasks:
        h.update(b"\x00" + tid.encode() + b"\x01" + text.encode())
    return h.hexdigest()[:16]


class Reference:
    """Task embeddings and the sorted cosines between distinct tasks; built once per reference key."""

    def __init__(self, ids: list[str], vectors: np.ndarray, key: str) -> None:
        self.ids, self.vectors, self.key = ids, vectors, key
        sims = vectors @ vectors.T
        self.pairs = np.sort(sims[np.triu_indices(len(ids), 1)])

    def score(self, records: np.ndarray, *, tau: float = TAU) -> dict[str, np.ndarray]:
        """S, A, U (and the nearest task, its cosine) for unit record vectors, one row each."""
        c = records @ self.vectors.T
        cmax = c.max(axis=1)
        s = np.searchsorted(self.pairs, cmax, side="right") / max(len(self.pairs), 1)
        z = np.exp((c - cmax[:, None]) / tau)
        p = z / z.sum(axis=1, keepdims=True)
        h = -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=1)
        a = np.exp(h) / c.shape[1]
        return {"S": s, "A": a, "U": s * a, "c_max": cmax, "nearest": c.argmax(axis=1)}


_references: dict[str, Reference] = {}


def reference(memory_dir: str | Path, tasks: Sequence[tuple[str, str]] | None = None) -> Reference | None:
    """The reference for the current task file and embedding model, or None without the embedding service."""
    from .. import embedding

    if not embedding.configured():
        return None
    tasks = list(tasks) if tasks is not None else load_tasks()
    if len(tasks) < 2:
        raise ValueError("the reference needs at least two tasks")
    key = reference_key(tasks, embedding.model_name())
    if key not in _references:
        vecs = embedding.embed_documents([text for _, text in tasks], memory_dir=memory_dir)
        if vecs is None:
            return None
        _references[key] = Reference([tid for tid, _ in tasks], vecs, key)
    return _references[key]


# ----------------------------------------------------------------------------- records


def record_text(item: Mapping[str, Any]) -> str:
    """The text a record is embedded as, the same view for L1 and L2 (and for clustering)."""
    from .cross_paper import _view_text

    return _view_text(item)


def _scores_path(memory_dir: str | Path, project_id: str) -> Path:
    return Path(memory_dir).expanduser() / EXPERIENCE_DIR / project_id / UTILITY_DIR / "scores.json"


def load_scores(memory_dir: str | Path, project_id: str) -> dict[str, Any]:
    data = read_json(_scores_path(memory_dir, project_id))
    return data if isinstance(data, dict) and isinstance(data.get("scores"), dict) else {"ref": "", "scores": {}}


def _library(memory_dir: str | Path, project_id: str) -> list[tuple[str, str, dict[str, Any]]]:
    """(record id, level, item) for every current-shape record of the project, L1 and L2."""
    from .store import list_experience_documents

    out = []
    for doc in list_experience_documents(memory_dir=memory_dir, project_id=project_id):
        try:
            item = json.loads(doc.text).get("experience")
        except (ValueError, AttributeError):
            continue
        if isinstance(item, Mapping) and item.get("problem"):
            out.append((doc.observation_id, doc.experience_level or "l1", dict(item)))
    return out


def update_scores(memory_dir: str | Path, project_id: str, *, tasks: Sequence[tuple[str, str]] | None = None,
                  records: Sequence[tuple[str, str, Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    """Score the records that have no score for the current reference yet; drop the scores of records that left.

    Records already scored are not touched: a score depends on the record and the reference only. Never raises.
    """
    try:
        ref = reference(memory_dir, tasks)
        if ref is None:
            return {"status": "no_embeddings"}
        lib = list(records) if records is not None else _library(memory_dir, project_id)
        saved = load_scores(memory_dir, project_id)
        scores = saved["scores"] if saved.get("ref") == ref.key else {}
        live = {rid for rid, _, _ in lib}
        dropped = [rid for rid in scores if rid not in live]
        for rid in dropped:
            scores.pop(rid)
        todo = [(rid, lvl, item) for rid, lvl, item in lib if rid not in scores]
        if todo:
            from .. import embedding

            vecs = embedding.embed_documents([record_text(item) for _, _, item in todo], memory_dir=memory_dir)
            if vecs is None:
                return {"status": "no_embeddings"}
            out = ref.score(vecs)
            for k, (rid, lvl, _) in enumerate(todo):
                scores[rid] = {"S": round(float(out["S"][k]), 4), "A": round(float(out["A"][k]), 4),
                               "U": round(float(out["U"][k]), 4), "c_max": round(float(out["c_max"][k]), 4),
                               "nearest": ref.ids[int(out["nearest"][k])], "level": lvl}
        if todo or dropped or saved.get("ref") != ref.key:
            path = _scores_path(memory_dir, project_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, {"ref": ref.key, "tasks": str(tasks_path()) if tasks is None else "(given)",
                                     "n_tasks": len(ref.ids), "tau": TAU, "scores": scores})
        return {"status": "ok", "scored": len(todo), "dropped": len(dropped), "total": len(scores), "ref": ref.key}
    except Exception as exc:  # a measurement; it must never take extraction down
        logger.warning("experience utility scoring failed", exc_info=True)
        return {"status": "error", "error": str(exc)}


def score_of(memory_dir: str | Path, project_id: str, record_id: str) -> dict[str, Any] | None:
    return load_scores(memory_dir, project_id)["scores"].get(record_id)


__all__ = [
    "TAU",
    "Reference",
    "load_scores",
    "load_tasks",
    "record_text",
    "reference",
    "reference_key",
    "score_of",
    "tasks_path",
    "update_scores",
]
