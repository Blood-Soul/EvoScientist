"""Embeddings: the client and its cache, clustering on dense vectors, and hybrid retrieval. No real model: a fake
OpenAI-compatible service maps words to fixed directions."""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from EvoScientist.memory import dense, embedding
from EvoScientist.memory.experiences import cross_paper as C
from EvoScientist.memory.types import MemoryScope, MemoryType, ObservationSearchDocument

DIM = 16
TOPICS = {"span": 0, "abstain": 1, "unanswerable": 1, "warmup": 2, "learning": 2, "graph": 3, "molecule": 3}


def fake_vector(text: str) -> list[float]:
    """Words of one topic point the same way; anything else adds a small text-specific direction."""
    v = np.zeros(DIM)
    body = text.split("Query:", 1)[-1].lower()
    for word, axis in TOPICS.items():
        if word in body:
            v[axis] += 1.0
    seed = int(hashlib.sha256(body.encode()).hexdigest()[:8], 16)
    v[4 + seed % (DIM - 4)] += 0.15
    return (v / np.linalg.norm(v)).tolist()


class FakeService:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail = False
        svc = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                svc.calls.append(body["input"])
                if svc.fail:
                    self.send_response(500)
                    self.end_headers()
                    return
                data = json.dumps({"data": [{"index": i, "embedding": fake_vector(t)}
                                            for i, t in enumerate(body["input"])]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def service(monkeypatch):
    embedding._reset_for_tests()
    svc = FakeService()
    monkeypatch.setenv(embedding.URL_ENV, svc.url)
    yield svc
    svc.close()
    embedding._reset_for_tests()


@pytest.fixture(autouse=True)
def _lexical_backend():
    yield
    C._backend = "lexical"


# ---------------------------------------------------------------------- the client


def test_without_a_service_nothing_is_embedded(tmp_path, monkeypatch):
    monkeypatch.delenv(embedding.URL_ENV, raising=False)
    assert embedding.embed_documents(["a"], memory_dir=tmp_path) is None
    assert dense.dense_rank([], "q", task="t", memory_dir=tmp_path, limit=3) is None


def test_vectors_are_unit_cached_and_queries_carry_the_instruction(service, tmp_path):
    v = embedding.embed_documents(["span head", "warmup schedule"], memory_dir=tmp_path)
    assert v.shape == (2, DIM) and np.allclose(np.linalg.norm(v, axis=1), 1)
    again = embedding.embed_documents(["warmup schedule", "span head"], memory_dir=tmp_path)
    assert np.allclose(again[0], v[1]) and len(service.calls) == 1  # the second call came from the cache
    embedding.embed_queries(["abstain?"], task="Find records", memory_dir=tmp_path)
    assert service.calls[-1] == ["Instruct: Find records\nQuery:abstain?"]
    assert (tmp_path / "embeddings" / "Qwen_Qwen3-Embedding-0.6B.sqlite").is_file()


def test_a_failing_service_returns_none_and_is_not_asked_again_at_once(service, tmp_path):
    service.fail = True
    assert embedding.embed_documents(["x"], memory_dir=tmp_path) is None
    service.fail = False
    assert embedding.embed_documents(["y"], memory_dir=tmp_path) is None  # backing off
    assert len(service.calls) == 1


def test_concurrent_calls_share_one_cache_without_sqlite_misuse(service, tmp_path):
    """The main agent and a node agent embed into the same memory directory from different threads.

    They share one SQLite connection; unsynchronised use used to raise ``ProgrammingError: bad parameter
    or other API misuse``, which made the client fall back to lexical ranking for every thread at once.
    """
    threads, barrier = 24, threading.Barrier(24)
    results: list[np.ndarray | None] = []
    lock = threading.Lock()

    def one(k: int) -> None:
        barrier.wait()
        v = embedding.embed_documents(
            [f"graph descriptor {k}-{i} for molecule {i % 7}" for i in range(32)], memory_dir=tmp_path)
        with lock:
            results.append(v)

    ts = [threading.Thread(target=one, args=(k,)) for k in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(results) == threads and all(v is not None and v.shape == (32, DIM) for v in results)


# ---------------------------------------------------------------------- clustering


def _rec(rid: str, problem: str, method: str) -> dict:
    return {"id": rid, "paper": rid.split("#")[0], "item": {"problem": problem, "method": [method], "task": "qa"}}


def test_clustering_uses_embeddings_when_the_service_answers(service, tmp_path):
    recs = [_rec("p1#0", "span extraction", "span head"), _rec("p2#0", "span boundaries", "span decoder"),
            _rec("p3#0", "unanswerable questions", "abstain head"), _rec("p4#0", "warmup instability", "warmup")]
    vecs, backend = C.vectors(recs, tmp_path)
    assert backend == "embedding" and isinstance(vecs[0], np.ndarray)
    assert C.join_min() == C.EMBED_JOIN and C.min_cohesion() == C.EMBED_COHESION
    state = {"clusters": {}, "next_cluster": 0}
    C.assign(state, [r["id"] for r in recs], dict(zip([r["id"] for r in recs], vecs, strict=True)))
    groups = sorted(sorted(c["members"]) for c in state["clusters"].values())
    assert ["p1#0", "p2#0"] in groups and ["p3#0"] in groups and ["p4#0"] in groups


def test_clustering_falls_back_to_tfidf(tmp_path, monkeypatch):
    monkeypatch.delenv(embedding.URL_ENV, raising=False)
    vecs, backend = C.vectors([_rec("p1#0", "span", "span head")], tmp_path)
    assert backend == "lexical" and isinstance(vecs[0], dict) and C.join_min() == C.LEXICAL_JOIN


def test_the_embedding_view_labels_each_field_once():
    text = C._view_text({"problem": "p", "method": ["m1", "m2"], "task": "t", "action": "a"})
    assert text == "Problem: p\nMethod: m1; m2\nTask: t\nAction: a"


def test_a_backend_change_reclusters_and_hands_findings_over():
    state = {"clusters": {"c1": {"members": ["a", "b", "c"], "findings": ["F1"]}},
             "findings": {"F1": {"record": {"source_l1_ids": ["a", "b", "c"]}}}, "next_cluster": 1}
    old, state["clusters"] = state["clusters"], {}
    state["clusters"] = {"c2": {"members": ["a", "b"], "findings": []}, "c3": {"members": ["c"], "findings": []}}
    C._hand_over(state, old)
    assert state["clusters"]["c2"]["findings"] == ["F1"] and state["clusters"]["c2"]["split"] is True
    assert state["clusters"]["c3"]["findings"] == []


# ---------------------------------------------------------------------- retrieval


def _doc(rid: str, text: str) -> ObservationSearchDocument:
    return ObservationSearchDocument(observation_id=rid, path=f"/m/{rid}", memory_type=MemoryType.SEMANTIC,
                                     scope=MemoryScope.PROJECT, summary=text[:40], body=text, text=text,
                                     record_kind="experience", experience_level="l1")


def test_dense_retrieval_finds_what_shares_no_word_with_the_query(service, tmp_path):
    docs = [_doc("E-1", "train an abstain classifier for questions without an answer"),
            _doc("E-2", "linear warmup of the learning rate"), _doc("E-3", "graph networks for molecule property")]
    lexical: list = []  # the query shares no word with E-1
    out = dense.hybrid(docs, "how to handle unanswerable items", lexical, task="t", memory_dir=tmp_path, limit=2)
    assert out[0]["observation_id"] == "E-1" and out[0]["matches"]


def test_fusion_keeps_lexical_hits_and_promotes_agreement():
    a, b, c = _doc("E-a", "x"), _doc("E-b", "y"), _doc("E-c", "z")
    lexical = [dense.hit_from_document(b, 0.0), dense.hit_from_document(a, 0.0)]
    out = dense.fuse(lexical, [(a, 0.9), (c, 0.8)], limit=3, make_hit=dense.hit_from_document)
    assert [h["observation_id"] for h in out] == ["E-a", "E-b", "E-c"]  # E-a is in both rankings
    assert dense.fuse(lexical, None, limit=1, make_hit=dense.hit_from_document) == lexical[:1]


def test_search_experience_records_fuses_when_configured(service, tmp_path, monkeypatch):
    from EvoScientist.memory.experiences import retrieval

    docs = [_doc("E-1", "an abstain head for unanswerable questions"), _doc("E-2", "warmup of the learning rate")]
    monkeypatch.setattr(retrieval, "list_experience_documents", lambda **_: docs)
    hits = retrieval.search_experience_records(memory_dir=tmp_path, project_id="P", topic="questions with no answer")
    assert hits[0]["observation_id"] == "E-1"
    monkeypatch.delenv(embedding.URL_ENV)
    lexical = retrieval.search_experience_records(memory_dir=tmp_path, project_id="P", topic="questions with no answer")
    assert [h["observation_id"] for h in lexical] == ["E-1"]  # the word "questions" still matches lexically


def test_a_library_clustered_lexically_moves_to_embeddings_and_keeps_its_findings(tmp_path, monkeypatch):
    """One trigger on TF-IDF makes a finding; the next, with the service on, re-clusters on embeddings, hands the
    finding to the cluster holding its support and revises it there instead of losing it."""
    import asyncio
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    import test_cross_paper as T

    monkeypatch.delenv(embedding.URL_ENV, raising=False)
    mem = T.library(tmp_path)
    model = T.Scripted(T.induce_all)
    first = asyncio.run(C.run_cross_paper_induction(memory_dir=mem, project_id=T.PROJECT, model=model, force=True))
    assert first["status"] == "ok" and first["vectors"] == "lexical" and first["findings"] == 1
    fid = next(iter(C.load_state(mem, T.PROJECT)["findings"]))

    embedding._reset_for_tests()
    svc = FakeService()
    monkeypatch.setenv(embedding.URL_ENV, svc.url)
    # the fake service puts the three empty-window records in one direction ("abstain") and every other apart
    monkeypatch.setitem(TOPICS, "window", 1)
    try:
        second = asyncio.run(C.run_cross_paper_induction(memory_dir=mem, project_id=T.PROJECT, model=model, force=True))
    finally:
        svc.close()
        embedding._reset_for_tests()
    state = C.load_state(mem, T.PROJECT)
    assert second["status"] == "ok" and second["vectors"] == "embedding" and state["backend"] == "embedding"
    assert second["retired"] == 0 and len(state["findings"]) == 1
    # the cluster was revised with the finding it inherited, not induced from scratch
    assert "current_findings" in json.loads(model.asked[-1])
    assert fid in state["findings"] or any(f.get("lineage") == fid for f in state["findings"].values())
